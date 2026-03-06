import asyncio
from rich.console import Console
from rich.prompt import Prompt
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.workflow import Context
from llama_index.core import Settings




# LLM Configuration
llm = BedrockConverse(model='us.amazon.nova-pro-v1:0', region_name="us-west-2", max_tokens=10000)
Settings.llm = llm



async def initialize_mcp_async(mcp_servers: dict):
    """Initialize MCP clients with better error handling"""

    SYSTEM_PROMPT = """\
    You are an AI assistant for Tool Calling. Here are the instructions for you to follow.
    1. Before you help a user, you need to work with tools to interact with Our Backend systems. 
    2. Follow a chain of thought reasoning. 
    3. If the user's question requires a tabulated response, then do your best to format it as a table or chart.
    """
    all_tools = []
# But do NOT include the details of your thinking in your response.
    for server_name, server_url in mcp_servers.items():

        client = BasicMCPClient(server_url)
        tool_spec = McpToolSpec(client=client)

        # Get tools
        tools = await tool_spec.to_tool_list_async()

        all_tools.extend(tools)

        # Brief pause
        await asyncio.sleep(0.3)


    """Create agent with tools"""
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with a variety of MCP servers including chart generation.",
        tools=all_tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent



async def main():
    # Server configurations
    MCP_SERVERS = {
        "NSE Server": "http://127.0.0.1:4201/mcp",
        "Email server": "http://127.0.0.1:4202/sse",
    }

    agent = await initialize_mcp_async(MCP_SERVERS)

    console = Console()
    BOTSTYLE = "bold blue"


    console.clear()
    messages = []
    console.print("Welcome to the Terminal Chatbot! Type 'exit' to quit.", style=BOTSTYLE)

    while True:
        # Always prompt at the bottom
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        if user_input.strip().lower() == "exit":
            console.print("[bold magenta]Goodbye![/bold magenta]")
            break

        messages.append(("user", user_input))
        agent_context = Context(agent)

        with console.status("Thinking...", spinner="dots"):
            handler = agent.run(user_input, ctx=agent_context)
            response = await handler

        console.print(f"Bot:{response}", justify="right", style=BOTSTYLE)
        messages.append(("bot", response))


if __name__ == "__main__":
    asyncio.run(main())