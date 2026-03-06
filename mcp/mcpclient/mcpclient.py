from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.client.transports import PythonStdioTransport

from fastmcp import Client
import asyncio

async def main():
    nse_client = Client[StreamableHttpTransport]("http://127.0.0.1:4201/mcp")
    email_client = Client[PythonStdioTransport]("http://127.0.0.1:4202/sse")

    # Basic server interaction
    async with nse_client:
        await nse_client.ping()        
        tools_list = await nse_client.list_tools()
        resources_list = await nse_client.list_resources()
        prompts_list = await nse_client.list_prompts()


    async with email_client:
        await email_client.ping()
        tools_list += await email_client.list_tools()
        resources_list += await email_client.list_resources()
        prompts_list += await email_client.list_prompts()



    for tool in tools_list:
        print(tool,"\n\n\n")
    print("--------------------------------")
    for resource in resources_list:
        print(resource)
    print("--------------------------------")
    for prompt in prompts_list:
        print(prompt)
    print("--------------------------------")


    async with nse_client:
        result = await nse_client.call_tool("get_highest_opening_date", {"ticker": "TCS"})
        print(result)
    # async with email_client:
    #     result = await email_client.call_tool("send_email", {"strToEmailAddress": "gopi.nuti@autodesk.com", "strSubject": "Test Email", "strBody": str(result)})
    #     print(result)

    # async with nse_client:
    #     # This is not good for demonstrating elicitation. Invoke it from Cursor
    #     result = await nse_client.call_tool("get_highest_closing_date", {"ticker": "TCS", "bOpen": True, "bClose": True})
    #     print(result)

if __name__ == "__main__":
    asyncio.run(main())