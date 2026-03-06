# import boto3
import json
import datetime as datetime


# server.py
from fastmcp import FastMCP
import pandas as pd
# from pypdf import PdfReader

"""
This is a foolish MCP server which will return nonsensical data.
"""



server = FastMCP("workshop-nse-server")


# @server.tool
def get_opening_prices(ticker:str=None) -> list[dict]:
    """
        Return opening prices for a given ticker.
        If no ticker is provided, return all opening prices.
    """

    nsedata_df = pd.read_excel("../data/NSE Data YTD.xlsx")

    if ticker:
        filtered_df = nsedata_df[nsedata_df["NSESYMBOL"] == ticker]
    else:
        filtered_df = nsedata_df


    return filtered_df[["NSESYMBOL","OPEN"]].to_dict(orient="records")


@server.tool
def get_closing_prices(ticker:str=None) -> list[dict]:
    """
        Return closing prices for a given ticker.
        If no ticker is provided, return all closing prices.
    """

    nsedata_df = pd.read_excel("../data/NSE Data YTD.xlsx")

    if ticker:
        filtered_df = nsedata_df[nsedata_df["NSESYMBOL"] == ticker]
    else:
        filtered_df = nsedata_df


    return filtered_df[["NSESYMBOL","CLOSE"]].to_dict(orient="records")


@server.tool
def get_highest_opening_date(ticker:str) -> tuple[str, float]:
    """
        Return highest opening date and price for a given ticker.
    """
    nsedata_df = pd.read_excel("../data/NSE Data YTD.xlsx")
    filtered_df = nsedata_df[nsedata_df["NSESYMBOL"] == ticker]
    return str(filtered_df["Date"].max()), filtered_df["OPEN"].max()


@server.tool
def get_lowest_opening_date(ticker:str) -> tuple[str, float]:
    """
        Return lowest opening date and price for a given ticker.
    """
    nsedata_df = pd.read_excel("../data/NSE Data YTD.xlsx")
    filtered_df = nsedata_df[nsedata_df["NSESYMBOL"] == ticker]
    return str(filtered_df["Date"].min()), filtered_df["OPEN"].min()


@server.tool
def get_highest_closing_date(ticker:str) -> tuple[str, float]:
    """
        Return highest closing date and price for a given ticker.
    """
    nsedata_df = pd.read_excel("../data/NSE Data YTD.xlsx")
    filtered_df = nsedata_df[nsedata_df["NSESYMBOL"] == ticker]
    return str(filtered_df["Date"].max()), filtered_df["CLOSE"].max()


@server.tool
def get_lowest_closing_date(ticker:str) -> tuple[str, float]:
    """
        Return lowest closing date and price for a given ticker.
    """
    nsedata_df = pd.read_excel("../data/NSE Data YTD.xlsx")
    filtered_df = nsedata_df[nsedata_df["NSESYMBOL"] == ticker]
    return str(filtered_df["Date"].min()), filtered_df["CLOSE"].min()







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
1. What was the closing price of TCS when INFY closed lowest?
2. Get opening prices
3. Get closing prices
4. Get lowest closing date and price for TCS
5. Email lowest closing date and price for INFY
6. 
"""