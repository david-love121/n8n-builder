import json

from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.tools import Tool


def get_tool_json(tool: Tool):
    tool_data_dict = {
        "type": "function",
        "name": tool.name,
        "descrption": tool.description,
        "parameters": tool.inputSchema,
    }
    text = json.dumps(tool_data_dict)
    return text


class MCPStreamableHttp:
    def __init__(self, url: str, bearer: str | None = None):
        self.url = url
        self.bearer_token = BearerAuth(bearer) if bearer else None
        self.tools = None
        self._client = None

    async def __aenter__(self):
        self._client = Client(self.url, auth=self.bearer_token)
        await self._client.__aenter__()
        await self.fetch_tools()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    async def connect(self):
        if self._client is None:
            async with Client(self.url, auth=self.bearer_token) as client:
                await client.ping()
            return

    async def fetch_tools(self) -> dict:
        async with Client(self.url, auth=self.bearer_token) as client:
            self.tools = await client.list_tools()
        return self.tools

    async def print_tools(self):
        if self.tools is None:
            self.tools = await self.fetch_tools()
        for tool in self.tools:
            print(tool)

    async def openai_compatible_tools(self):
        final_json: str = ""
        for tool in self.tools:
            final_json += get_tool_json(tool)
        return final_json
