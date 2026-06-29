import asyncio
import os

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def test_stdio_tool_list_includes_read_only_annotations() -> None:
    async def run() -> None:
        params = StdioServerParameters(
            command="uv",
            args=["run", "stocker-mcp"],
            cwd=os.getcwd(),
            env={"STOCKER_HOME": os.path.expanduser("~/StockerLocal")},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = {tool.name: tool for tool in result.tools}

        assert "search" in tools
        assert "fetch" in tools
        for tool in tools.values():
            assert tool.annotations is not None
            assert tool.annotations.readOnlyHint is True
            assert tool.annotations.destructiveHint is False
            assert tool.outputSchema is not None

    asyncio.run(run())
