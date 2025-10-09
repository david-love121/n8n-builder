# n8n Builder

Generate n8n automations from natural language prompts and leverage Model Context Protocol (MCP) servers that run inside Docker containers. The project renders planning diagrams, stores metadata, and can call tools exposed by MCP servers through the [`fastmcp`](https://github.com/jlowin/fastmcp) client library.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for managing the virtual environment (`uv sync` will recreate `.venv`)
- Docker with access to the MCP server images you have built locally

## Installation

```bash
uv sync
```

This command installs Python dependencies into `.venv`. Activate the environment with:

```bash
source .venv/bin/activate
```

## Environment variables

Create a `.env` file in the project root. At minimum provide your OpenRouter key and identify each Dockerised MCP server you want to expose. Two servers are supported out of the box, but you can list any number.

```dotenv
OPENROUTER_KEY=sk-your-openrouter-key

# Comma-separated list of server identifiers
MCP_DOCKER_SERVERS=github,notion

# Configuration for the "github" server
MCP_DOCKER_GITHUB_IMAGE=github-mcp:latest
MCP_DOCKER_GITHUB_CMD=/server/github-mcp-server stdio
MCP_DOCKER_GITHUB_RUN_ARGS=--network host
MCP_DOCKER_GITHUB_ENV=GITHUB_PERSONAL_ACCESS_TOKEN=

# Configuration for the "notion" server
MCP_DOCKER_NOTION_IMAGE=notion-mcp:latest
MCP_DOCKER_NOTION_CMD=/server/notion-mcp-server stdio

# Configuration for the "context7" server served over HTTP
MCP_DOCKER_CONTEXT7_IMAGE=context7-mcp:latest
MCP_DOCKER_CONTEXT7_CMD=/server/context7-mcp-server http --port 8211
MCP_DOCKER_CONTEXT7_RUN_ARGS=-p 127.0.0.1:8211:8211
MCP_DOCKER_CONTEXT7_TRANSPORT=http
MCP_DOCKER_CONTEXT7_URL=http://127.0.0.1:8211/mcp
MCP_DOCKER_CONTEXT7_STARTUP_TIMEOUT=45
```

Configuration keys:

- `MCP_DOCKER_SERVERS`: Comma-separated identifiers. Each identifier is used when calling tools (e.g. `github`).
- `MCP_DOCKER_<ID>_IMAGE`: Docker image to run. Match the output of `docker image list`. If you want to add a namespace (for example `local/`), retag the image first: `docker tag github-mcp:latest local/github-mcp:latest`.
- `MCP_DOCKER_<ID>_CMD`: (optional) Command executed inside the container after the image name. Use it to invoke the MCP server binary with `stdio` transport if the entrypoint does not already do so.
- `MCP_DOCKER_<ID>_RUN_ARGS`: (optional) Additional arguments passed to `docker run` (e.g. `--network host`, volume mounts, etc.).
- `MCP_DOCKER_<ID>_ENV`: (optional) Semicolon-separated list of `KEY=VALUE` pairs forwarded into the container as environment variables (for example, `TOKEN=abc;DEBUG=1`). Leave the value empty (`TOKEN=`) to copy the value from your host environment variable of the same name.
- `MCP_DOCKER_<ID>_NAME`: (optional) Friendly display name for status output.
- `MCP_DOCKER_<ID>_TRANSPORT`: (optional) Transport for the MCP client. Defaults to `stdio`. Set to `http` to connect with an HTTP transport.
- `MCP_DOCKER_<ID>_URL`: Required when transport is `http`. URL that the client should call (e.g. `http://127.0.0.1:8211/mcp`).
- `MCP_DOCKER_<ID>_STARTUP_TIMEOUT`: (optional) Seconds to wait for the HTTP endpoint to become responsive (default 30).

## Usage

```bash
uv run python main.py
```

The script loads `.env`, prints the available tools from each configured MCP server, prompts for automation details, generates a Mermaid diagram, converts it to an image, and writes metadata and an n8n workflow JSON file to `charts-bin/`.

### Debugging

If the script appears to hang, enable verbose logging to see the exact Docker commands being invoked:

```bash
export MCP_DEBUG=1
uv run python main.py
```

This prints the `docker run` invocation for each MCP server and surfaces any connection errors raised by `fastmcp`.

### Calling MCP tools in code

The application exposes a `MCP_CLIENT_MANAGER` with helper methods. For example:

```python
from main import MCP_CLIENT_MANAGER

result = MCP_CLIENT_MANAGER.call_tool(
	"github",
	"list_open_pull_requests",
	{"repository": "david-love121/n8n-builder"},
)
print(result)
```

For asynchronous workflows, call `await MCP_CLIENT_MANAGER._call_tool_async(...)` directly instead of the synchronous wrapper.

## Testing

Currently the project has no automated test suite. Run the script end-to-end to verify changes.
