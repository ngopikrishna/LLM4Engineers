# import boto3
import json
import datetime as datetime
from dataclasses import dataclass

# server.py
from fastmcp import FastMCP, Context
import pandas as pd
# from pypdf import PdfReader



@dataclass
class GetPricesRequest:
    ticker: str
    bOpen: bool
    bClose: bool


@dataclass
class GetAuthenticationInformation:
    strUserName: str
    strPassword: str

server = FastMCP("workshop-server")

@server.tool
async def get_prices(ctx:Context) -> list[dict]:
    """
        Return opening and closing prices for given ticker.
        If no ticker is provided, return opening and closing prices for all tickers.
    """
    lst_columns = ["Date","NSESYMBOL"]
    nsedata_df = pd.read_excel("../data/NSE Data YTD.xlsx")


    # result = await ctx.elicit(
    #         message="Please provide the ticker you want to get the prices for, and whether you want to get the opening or closing prices",
    #         response_type=GetPricesRequest
    #     )

    result = await ctx.elicit(
        message="Please provide the username and password to login to the NSE website",
        response_type=GetAuthenticationInformation
    )



    if result.action == "accept":
        prices_request = result.data
        ticker = prices_request.ticker
        bOpen = prices_request.bOpen
        bClose = prices_request.bClose
    elif result.action == "decline":
        return [{"error":"User declined to provide the information"}]
    else:  # cancel
        return [{"error":"Operation cancelled"}]



    if bOpen:
        lst_columns.append("OPEN")

    if bClose:
        lst_columns.append("CLOSE")




    if ticker:
        filtered_df = nsedata_df[nsedata_df["NSESYMBOL"] == ticker]
    else:
        filtered_df = nsedata_df


    if bClose:
        filtered_df = filtered_df[lst_columns]
    return nsedata_df.to_dict(orient="records")

if __name__ == "__main__":
    try:
        server.run(transport="streamable-http",
                host="127.0.0.1",
                port=4201,
                log_level="debug")
        print("Server started successfully")
    except Exception as e:
        print(f"********** Failed to start server: {str(e)}", exc_info=True)
        raise




########################################################
# Questions:
"""
1. Get prices
2. Get opening prices for TCS
3. Get closing prices for INFY
4. Get Closing prices for TCS, INFY and opening prices for WIPRO
"""