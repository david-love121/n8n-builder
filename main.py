import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import mermaid as md
import yaml
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_project_dotenv() -> None:
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)


def require_env(variable: str) -> str:
    value = os.getenv(variable)
    if not value:
        raise RuntimeError(
            f"Environment variable '{variable}' is required but missing."
        )
    return value


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://api.openai.com/v1"
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name


    def create_completion(self, messages: list, model:str = ""):
        if model:
            return self.client.chat.completions.create(
                model=model, messages=messages
            )
        return self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )


def extract_mermaid_diagram(text: str) -> Tuple[Optional[str], str]:
    pattern = re.compile(r"```mermaid\s*(.*?)```(.*)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None, ""

    mermaid_code = match.group(1).strip()
    description = match.group(2).strip()
    return mermaid_code, description


def convert_mermaid_to_image(
    mermaid_code: str,
    output_path: Union[str, Path],
    kroki_base_url: str = "https://kroki.io",
) -> Path:

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    md.Mermaid(mermaid_code).to_png(output)
    return output


def sanitize_directory_name(name: str) -> str:
    sanitized = name.strip()
    sanitized = sanitized.replace(os.sep, "-")
    sanitized = sanitized.replace("/", "-")
    sanitized = sanitized.replace("\\", "-")
    sanitized = re.sub(r"[^A-Za-z0-9._\- ]+", "-", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized.replace(" ", "-")
    return sanitized or "automation"


def ensure_output_directory(automation_name: str) -> Path:
    charts_root = PROJECT_ROOT / "charts-bin"
    charts_root.mkdir(parents=True, exist_ok=True)
    directory = charts_root / sanitize_directory_name(automation_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_metadata(
    directory: Path,
    automation_name: str,
    user_prompt: str,

    mermaid_code: str,
    description: str,
) -> Path:
    metadata = {
        "automation_name": automation_name,
        "user_prompt": user_prompt,
        "mermaid_code": mermaid_code,
        "description": description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = directory / "metadata.yaml"
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return metadata_path


def collect_user_inputs() -> Tuple[str, str]:
    if sys.stdin.isatty():
        automation_name = ""
        while not automation_name:
            automation_name = input("Enter the automation name: ").strip()
            if not automation_name:
                print("Automation name cannot be empty. Please try again.")

        user_prompt = input("Enter the prompt: ").strip()
        return automation_name, user_prompt

    stdin_data = sys.stdin.read()
    automation_line, _, remainder = stdin_data.partition("\n")
    automation_name = automation_line.strip()
    user_prompt = remainder.strip()

    if not automation_name:
        automation_name = "automation"

    return automation_name, user_prompt


# Creates a mermaid diagram based on the user prompt, to be fed into automation builder
def planner(client: OpenAIClient, documentation: str = ""):
    automation_name, user_prompt = collect_user_inputs()
    output_directory = ensure_output_directory(automation_name)
    model_prompt = f"""Create a mermaid flowchart for the following automation: {user_prompt} \n\n Your output should contain only ONE mermaid diagram. 
    You should follow your diagram with a brief explanation of how the automation works. 
    Mermaid documentation: {documentation}
    Do not add semicolons at the end of lines in the mermaid diagram. Only use single quotes '' for strings in the diagram.
    
    Format your response like: ```mermaid\n<your diagram here>\n```\n<your explanation here>"""
    completion = client.create_completion(
        messages=[{"role": "user", "content": model_prompt}],
    )

    message_content = ""
    if getattr(completion, "choices", None):
        try:
            message_content = completion.choices[0].message.content or ""
        except AttributeError:
            pass

    if not message_content:
        message_content = getattr(completion, "content", "")

    print(message_content)

    mermaid_diagram, description = extract_mermaid_diagram(message_content)
    if not mermaid_diagram:
        print("No mermaid diagram detected in the response.")
        return

    output_file = convert_mermaid_to_image(
        mermaid_diagram, output_directory / "flowchart.png"
    )

    metadata_file = write_metadata(
        output_directory,
        automation_name,
        user_prompt,
        mermaid_diagram,
        description,
    )

    print(f"Mermaid diagram saved to {output_file.resolve()}")
    print(f"Metadata saved to {metadata_file.resolve()}")


# Creates the automation based on the user prompt and the generated mermaid diagram
def builder(client: OpenAIClient, documentation: str):
    automation_name = input("Enter the automation name: ").strip()
    output_directory = ensure_output_directory(automation_name)
    # Retrieve metadata to prompt model
    metadata_file = output_directory / "metadata.yaml"
    if not metadata_file.exists():
        print(f"Metadata file not found at {metadata_file.resolve()}")
        return
    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    user_prompt = metadata.get("user_prompt", "")
    mermaid_code = metadata.get("mermaid_code", "")
    description = metadata.get("description", "")
    model_prompt = f"""You must build an n8n automation based on the following user prompt and mermaid diagram. Ensure each connection in the mermaid diagram is represented in n8n nodes. The names in the diagram
    Won't be exact matches, you need to string together multiple nodes at times to create the intended effect. Make sure nodes you use actually exist in n8n.
    User prompt: {user_prompt} \n
    mermaid diagram: ```mermaid\n{mermaid_code}\n``` 
    description: {description}
    n8n documentation: {documentation}\n
    You can serach the web for a list of the n8n integrations / nodes here: https://docs.n8n.io/integrations/
    Your output should be a valid n8n workflow JSON, enclosed in triple backticks like: ```json\n<your json here>\n```"""
    initial_messages = [{"role": "user", "content": model_prompt}]
    completion = client.create_completion(
        messages=initial_messages
    )
    # Extract the JSON workflow from the response
    message_content = ""
    if getattr(completion, "choices", None):
        try:
            message_content = completion.choices[0].message.content or ""
        except AttributeError:
            pass
    if not message_content:
        message_content = getattr(completion, "content", "")
    print(message_content)
    pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(message_content)
    if not match:
        print("No valid JSON workflow found.")
        return
    json_workflow = match.group(1).strip()

    workflow_file = output_directory / "workflow.json"
    workflow_file.write_text(
        json_workflow,
        encoding="utf-8",
    )
    print(f"n8n workflow saved to {workflow_file.resolve()}")


async def main():
    load_project_dotenv()
    n8n_content = ""
    with open("./n8n-information.txt", 'r') as file:
        n8n_content = file.read()
    mermaid_content = ""
    with open("./mermaid-information.txt", 'r') as file:
        mermaid_content = file.read()


    client = OpenAIClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=require_env("OPENROUTER_KEY"),
        model_name="google/gemini-2.5-flash-preview-09-2025:online",
            )
    planner(client, mermaid_content)
    builder(client, n8n_content)


if __name__ == "__main__":
    asyncio.run(main())
