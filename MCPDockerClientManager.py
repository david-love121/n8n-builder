import asyncio
import json
import logging
import os
import re
import shlex

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from httpx import AsyncClient


@dataclass
class DockerMCPServerConfig:
    identifier: str
    image: str
    display_name: str
    docker_run_args: list[str]
    command: list[str]
    environment: Dict[str, str]
    transport: str = "stdio"
    http_url: Optional[str] = None
    startup_timeout: float = 30.0


# private helpers

_MCP_DEBUG = os.getenv("MCP_DEBUG", "").lower() in {"1", "true", "yes", "on"}
logging.basicConfig(
    level=logging.DEBUG if _MCP_DEBUG else logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("mcp.docker")


def _split_args(raw: str | None) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw)


def load_docker_mcp_servers() -> list[DockerMCPServerConfig]:
    servers_raw = os.getenv("MCP_DOCKER_SERVERS", "")
    identifiers = [item.strip() for item in servers_raw.split(",") if item.strip()]
    configs: list[DockerMCPServerConfig] = []

    for identifier in identifiers:
        normalized = _normalize_identifier(identifier)
        image = os.getenv(f"MCP_DOCKER_{normalized}_IMAGE")
        if not image:
            continue

        display_name = os.getenv(f"MCP_DOCKER_{normalized}_NAME", identifier)
        docker_args = _split_args(os.getenv(f"MCP_DOCKER_{normalized}_RUN_ARGS"))
        command = _split_args(os.getenv(f"MCP_DOCKER_{normalized}_CMD"))
        env_map = _parse_env_mapping(os.getenv(f"MCP_DOCKER_{normalized}_ENV"))
        transport = (
            os.getenv(f"MCP_DOCKER_{normalized}_TRANSPORT", "stdio").strip().lower()
            or "stdio"
        )
        http_url = os.getenv(f"MCP_DOCKER_{normalized}_URL")
        try:
            startup_timeout = float(
                os.getenv(f"MCP_DOCKER_{normalized}_STARTUP_TIMEOUT", "30")
            )
        except ValueError:
            startup_timeout = 30.0

        if transport == "http" and not http_url:
            LOGGER.warning(
                "Skipping MCP server '%s' because HTTP transport requires MCP_DOCKER_%s_URL",
                identifier,
                normalized,
            )
            continue

        configs.append(
            DockerMCPServerConfig(
                identifier=identifier,
                image=image,
                display_name=display_name,
                docker_run_args=docker_args,
                command=command,
                environment=env_map,
                transport=transport,
                http_url=http_url,
                startup_timeout=startup_timeout,
            )
        )

    return configs


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", value.upper())


def _parse_env_mapping(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    pairs = {}
    for item in raw.split(";"):
        key, sep, val = item.partition("=")
        key = key.strip()
        if not key:
            continue

        value = val.strip() if sep else ""
        if not value:
            env_value = os.getenv(key)
            if env_value is None:
                LOGGER.warning(
                    "Environment variable '%s' not found while configuring MCP docker env",
                    key,
                )
                continue
            value = env_value
        pairs[key] = value
    return pairs


def _build_transport_args(config: DockerMCPServerConfig) -> list[str]:
    args: list[str] = ["run", "--rm"]
    if config.transport == "stdio":
        args.append("--interactive")
    for key, value in config.environment.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend(config.docker_run_args)
    args.append(config.image)
    args.extend(config.command)
    if _MCP_DEBUG:
        LOGGER.debug(
            "Launching MCP server '%s' with command: %s",
            config.identifier,
            shlex.join(["docker", *args]),
        )
    return args


class MCPDockerClientManager:
    def __init__(self):
        self._configs = load_docker_mcp_servers()

    def refresh(self) -> None:
        self._configs = load_docker_mcp_servers()

    def describe_tools(self) -> Optional[str]:
        if not self._configs:
            return None
        try:
            return asyncio.run(self._describe_tools_async())
        except RuntimeError as exc:
            raise RuntimeError(
                "describe_tools() must be awaited when used inside an active event loop"
            ) from exc

    async def _describe_tools_async(self) -> str:
        sections: list[str] = []
        for config in self._configs:
            section = await self._collect_server_summary(config)
            sections.append(section)
        return "\n\n".join(sections)

    async def _collect_server_summary(self, config: DockerMCPServerConfig) -> str:
        header = f"MCP Server: {config.display_name}"

        try:
            tools = await self._list_tools_for_config(config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to collect tools for '%s'", config.identifier)
            return f"{header}\n  Error: {exc}"

        if not tools:
            return f"{header}\n  No tools reported."

        lines = [header]
        for tool in tools:
            description = getattr(tool, "description", "") or "No description provided."
            lines.append(f"  - {tool.name}: {description}")
        return "\n".join(lines)

    async def _list_tools_for_config(self, config: DockerMCPServerConfig) -> list[Any]:
        if config.transport == "http":
            async with self._run_http_container(config) as transport:
                client = Client(transport)
                async with client:
                    return await client.list_tools()

        transport_args = _build_transport_args(config)
        transport = StdioTransport(
            command="docker",
            args=transport_args,
            keep_alive=False,
        )
        client = Client(transport)
        async with client:
            return await client.list_tools()

    def call_tool(
        self, server_identifier: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        config = self._get_config(server_identifier)
        if not config:
            raise ValueError(f"Unknown MCP server '{server_identifier}'.")
        try:
            return asyncio.run(self._call_tool_async(config, tool_name, arguments))
        except RuntimeError as exc:
            raise RuntimeError(
                "call_tool() must be awaited when used inside an active event loop"
            ) from exc

    def _get_config(self, identifier: str) -> Optional[DockerMCPServerConfig]:
        for config in self._configs:
            if config.identifier == identifier:
                return config
        return None

    async def _call_tool_async(
        self,
        config: DockerMCPServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        if config.transport == "http":
            return await self._call_tool_http(config, tool_name, arguments)

        transport_args = _build_transport_args(config)
        transport = StdioTransport(
            command="docker",
            args=transport_args,
            keep_alive=False,
        )
        client = Client(transport, timeout=60.0)

        async with client:
            result = await client.call_tool(tool_name, arguments)
        return getattr(result, "content", result)

    async def _collect_http_server_summary(self, config: DockerMCPServerConfig) -> str:
        header = f"MCP Server: {config.display_name}"
        try:
            tools = await self._list_tools_for_config(config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to collect tools for '%s'", config.identifier)
            return f"{header}\n  Error: {exc}"

        if not tools:
            return f"{header}\n  No tools reported."

        lines = [header]
        for tool in tools:
            description = getattr(tool, "description", "") or "No description provided."
            lines.append(f"  - {tool.name}: {description}")
        return "\n".join(lines)

    def describe_mcp_tools_json(self) -> Optional[str]:
        self.refresh()
        try:
            summaries = asyncio.run(self._describe_tools_json_async())
        except RuntimeError as exc:
            raise RuntimeError(
                "describe_mcp_tools_json() must be awaited when used inside an active event loop"
            ) from exc
        if not summaries:
            return None
        return json.dumps({"servers": summaries}, indent=2)

    async def _describe_tools_json_async(self) -> list[Dict[str, Any]]:
        summaries: list[Dict[str, Any]] = []
        for config in self._configs:
            summaries.append(await self._collect_server_summary_dict(config))
        return summaries

    async def _collect_server_summary_dict(
        self, config: DockerMCPServerConfig
    ) -> Dict[str, Any]:
        try:
            tools = await self._list_tools_for_config(config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to collect tools for '%s'", config.identifier)
            return {
                "identifier": config.identifier,
                "display_name": config.display_name,
                "transport": config.transport,
                "tools": [],
                "error": str(exc),
            }

        return {
            "identifier": config.identifier,
            "display_name": config.display_name,
            "transport": config.transport,
            "tools": [
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", "")
                    or "No description provided.",
                    "input_schema": getattr(tool, "input_schema", None),
                }
                for tool in tools
            ],
        }

    def tool_definitions_for_openai(self) -> list[Dict[str, Any]]:
        self.refresh()
        try:
            return asyncio.run(self._tool_definitions_for_openai_async())
        except RuntimeError as exc:
            raise RuntimeError(
                "tool_definitions_for_openai() must be awaited when used inside an active event loop"
            ) from exc

    async def _tool_definitions_for_openai_async(self) -> list[Dict[str, Any]]:
        specifications: list[Dict[str, Any]] = []
        for config in self._configs:
            try:
                tools = await self._list_tools_for_config(config)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to collect tools for '%s'", config.identifier)
                continue

            for tool in tools:
                schema = getattr(tool, "input_schema", None)
                if not schema:
                    schema = {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    }
                specifications.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{config.identifier}::{tool.name}",
                            "description": getattr(tool, "description", "")
                            or "No description provided.",
                            "parameters": schema,
                        },
                    }
                )
        return specifications

    def call_tool_by_qualified_name(
        self, qualified_name: str, arguments: Dict[str, Any]
    ) -> Any:
        server_identifier, sep, tool_name = qualified_name.partition("::")
        if not sep:
            raise ValueError(
                "Tool name must be qualified as '<server_identifier>::<tool_name>'."
            )
        return self.call_tool(server_identifier, tool_name, arguments)

    def execute_openai_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """Execute a tool call returned by the OpenAI-compatible API."""

        if tool_call is None:
            raise ValueError("tool_call cannot be None")

        if isinstance(tool_call, dict):
            payload = tool_call
        else:
            payload = {
                "id": getattr(tool_call, "id", None),
                "type": getattr(tool_call, "type", None),
                "function": {
                    "name": getattr(getattr(tool_call, "function", None), "name", None),
                    "arguments": getattr(
                        getattr(tool_call, "function", None), "arguments", ""
                    ),
                },
            }

        function = payload.get("function", {})
        name = function.get("name")
        arguments_raw = function.get("arguments", "{}")
        if not name:
            raise ValueError("Tool call missing function name")

        try:
            arguments = (
                json.loads(arguments_raw)
                if isinstance(arguments_raw, str)
                else arguments_raw
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON arguments for tool '{name}': {exc}"
            ) from exc

        result = self.call_tool_by_qualified_name(name, arguments or {})
        return {
            "tool_call_id": payload.get("id"),
            "qualified_name": name,
            "arguments": arguments or {},
            "result": result,
        }

    async def _call_tool_http(
        self,
        config: DockerMCPServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        async with self._run_http_container(config) as transport:
            client = Client(transport, timeout=60.0)
            async with client:
                result = await client.call_tool(tool_name, arguments)
        return getattr(result, "content", result)

    @asynccontextmanager
    async def _run_http_container(self, config: DockerMCPServerConfig):
        assert config.http_url, "HTTP transport requires a target URL"
        command = ["docker", *_build_transport_args(config)]
        LOGGER.debug("Starting HTTP MCP container for '%s'", config.identifier)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stderr_task = asyncio.create_task(
            self._drain_stream(process.stderr, config.identifier)
        )

        try:
            await self._wait_for_http_ready(config.http_url, config.startup_timeout)
            transport = StreamableHttpTransport(url=config.http_url)
            yield transport
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

            await stderr_task

    async def _wait_for_http_ready(self, url: str, timeout: float) -> None:
        deadline = asyncio.get_event_loop().time() + max(timeout, 1.0)
        async with AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    response = await client.get(url)
                    if response.status_code < 400:
                        return
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(0.5)

                if asyncio.get_event_loop().time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for MCP HTTP server at {url}"
                    )

    async def _drain_stream(
        self, stream: asyncio.StreamReader | None, identifier: str
    ) -> None:
        if not stream:
            return
        while not stream.at_eof():
            line = await stream.readline()
            if not line:
                break
            LOGGER.debug(
                "[%s stderr] %s", identifier, line.decode(errors="ignore").rstrip()
            )

    def describe_mcp_tools(self) -> Optional[str]:
        self.refresh()
        summary = self.describe_tools()
        if not summary:
            return None
        return "Available MCP tools:\n" + summary
