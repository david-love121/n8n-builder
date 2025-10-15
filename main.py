import asyncio
import json
import os
import re
import sys


from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import httpx
import yaml
from dotenv import load_dotenv

from openai import OpenAI
from MCPStreamableHttp import MCPStreamableHttp


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
        base_url: str = "https://api.openai.com/v1",
        tools: list[str] = [],
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.tools = tools

    def create_completion(self, messages: list):
        return self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=False, tools=self.tools
        )


def extract_mermaid_diagram(text: str) -> Optional[str]:
    pattern = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def convert_mermaid_to_image(
    mermaid_code: str,
    output_path: Union[str, Path],
    image_format: str = "png",
    kroki_base_url: str = "https://kroki.io",
) -> Path:
    supported_formats = {"png", "svg", "pdf"}
    format_lower = image_format.lower()
    if format_lower not in supported_formats:
        raise ValueError(
            f"Unsupported image format '{image_format}'. Choose from {sorted(supported_formats)}."
        )

    api_url = f"{kroki_base_url.rstrip('/')}/mermaid/{format_lower}"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Content-Type": "text/plain"}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            api_url, content=mermaid_code.encode("utf-8"), headers=headers
        )
        response.raise_for_status()

    output.write_bytes(response.content)
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
    model_prompt: str,
    mermaid_code: str,
) -> Path:
    metadata = {
        "automation_name": automation_name,
        "user_prompt": user_prompt,
        "model_prompt": model_prompt,
        "mermaid_code": mermaid_code,
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
def planner(mcp_manager: MCPStreamableHttp):
    mcp_info = mcp_manager.describe_mcp_tools()
    if mcp_info:
        print(mcp_info)
    client = OpenAIClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=require_env("OPENROUTER_KEY"),
    )
    automation_name, user_prompt = collect_user_inputs()
    output_directory = ensure_output_directory(automation_name)
    model_prompt = f"""Create a mermaid flowchart for the following automation: {user_prompt} \n\n Your output should contain only ONE mermaid diagram. 
    You should follow your diagram with a brief explanation of how the automation works. 
    Format your response like: ```mermaid\n<your diagram here>\n```\n<your explanation here>"""
    completion = client.create_completion(
        messages=[{"role": "user", "content": model_prompt}],
        model="deepseek/deepseek-chat-v3.1:free",
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

    mermaid_diagram = extract_mermaid_diagram(message_content)
    if not mermaid_diagram:
        print("No mermaid diagram detected in the response.")
        return

    try:
        output_file = convert_mermaid_to_image(
            mermaid_diagram, output_directory / "flowchart.png"
        )
    except httpx.HTTPError as error:
        print(f"Failed to render mermaid diagram: {error}")
        return

    metadata_file = write_metadata(
        output_directory,
        automation_name,
        user_prompt,
        model_prompt,
        mermaid_diagram,
    )

    print(f"Mermaid diagram saved to {output_file.resolve()}")
    print(f"Metadata saved to {metadata_file.resolve()}")


# Creates the automation based on the user prompt and the generated mermaid diagram
def builder(mcp_manager: MCPStreamableHttp, client: OpenAIClient):
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
    model_prompt = f"""You must build an n8n automation based on the following user prompt and mermaid diagram. Ensure each connection in the mermaid diagram is represented in n8n. 
    User prompt: {user_prompt} \n
    mermaid diagram: ```mermaid\n{mermaid_code}\n``` 
    Your output should be a valid n8n workflow JSON, enclosed in triple backticks like: ```json\n<your json here>\n```"""
    initial_messages = [{"role": "user", "content": model_prompt}]
    completion = client.create_completion(
        messages=initial_messages,
        model="deepseek/deepseek-chat-v3.1:free",
        tools=mcp_manager.tools,
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

    async with MCPStreamableHttp(
        url=require_env("MCP_URL"), bearer=require_env("MCP_BEARER")
    ) as mcp_manager:
        openai_formatted_tools = await mcp_manager.openai_compatible_tools()
        client = OpenAIClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=require_env("OPENROUTER_KEY"),
            model_name="x-ai/grok-code-fast-1",
            tools=openai_formatted_tools,
        )

        messages = [
            {
                "role": "user",
                "content": "Use the resolve-library-id tool to find the context7 specific library-id for n8n",
            }
        ]
        response = client.create_completion(messages=messages)
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
