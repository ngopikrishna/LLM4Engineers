"""Data agent engine: LlamaIndex ReActAgent over an NSE stock Excel workbook.

The Excel file is loaded once at startup into a pandas DataFrame.  A set of
focused FunctionTools expose common stock-market queries (price look-up, range
history, comparison, volume, gainers/losers, returns).  A ReActAgent selects
and chains these tools to answer arbitrary natural-language questions.

Public API
----------
load_data(xlsx_path)    — call once at startup; populates _workflow
data_info()             — returns a summary dict for the UI header
ask_stream(question)    — sync generator; yields (event_type, payload) tuples
reset()                 — clear the conversation history

Event types yielded by ask_stream
----------------------------------
("status",       str)          — status bar text (e.g. "Calling get_price_on_date()")
("tool_results", list[dict])   — accumulated tool call results so far
("text",         str)          — a token of the final answer (streamed)
"""

import asyncio
import queue
import threading

import pandas as pd
from llama_index.core import Settings
from llama_index.core.agent import AgentWorkflow, ReActAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentStream,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

import config

# ── Module-level state ────────────────────────────────────────────────────────

_df: pd.DataFrame | None = None
_data_meta: dict = {}
_workflow: AgentWorkflow | None = None
_chat_history: list[ChatMessage] = []

# Stores (tool_name, result_str) for the last agent call — shown in the UI panel.
_last_tool_results: list[dict] = []


# ── Pandas tools ──────────────────────────────────────────────────────────────

def _record_result(tool_name: str, result: str) -> str:
    """Store the tool result so the UI panel can display it, then return it."""
    _last_tool_results.append({"tool": tool_name, "result": result})
    return result


def list_available_symbols() -> str:
    """Return all stock ticker symbols available in the dataset and the date range covered."""
    symbols    = sorted(_df["NSESYMBOL"].unique().tolist())
    date_min   = _df["Date"].min().strftime("%Y-%m-%d")
    date_max   = _df["Date"].max().strftime("%Y-%m-%d")
    result = (
        f"Available symbols: {', '.join(symbols)}\n"
        f"Date range: {date_min} to {date_max}\n"
        f"Total trading days per symbol: {_df.groupby('NSESYMBOL').size().to_dict()}"
    )
    return _record_result("list_available_symbols", result)


def get_price_on_date(symbol: str, date: str) -> str:
    """Get the OHLC prices, volume, and closing price for a stock symbol on a specific date.

    Args:
        symbol: NSE ticker symbol (e.g. 'TCS', 'INFY', 'WIPRO')
        date: date in YYYY-MM-DD format
    """
    try:
        dt  = pd.to_datetime(date)
        row = _df[(_df["NSESYMBOL"] == symbol.upper()) & (_df["Date"] == dt)]
        if row.empty:
            result = f"No data found for {symbol} on {date}. The market may have been closed or the date is outside the dataset range."
        else:
            r = row.iloc[0]
            result = (
                f"{symbol} on {date}:\n"
                f"  Open:        {r['OPEN']:.2f}\n"
                f"  High:        {r['HIGH']:.2f}\n"
                f"  Low:         {r['LOW']:.2f}\n"
                f"  Close:       {r['CLOSE']:.2f}\n"
                f"  Prev. Close: {r['PREV. CLOSE']:.2f}\n"
                f"  VWAP:        {r['vwap']:.2f}\n"
                f"  Volume:      {r['VOLUME']:,}\n"
                f"  Trades:      {r['No of trades']:,}"
            )
    except Exception as e:
        result = f"Error: {e}"
    return _record_result("get_price_on_date", result)


def get_price_history(symbol: str, start_date: str, end_date: str) -> str:
    """Get daily closing prices for a stock between two dates (inclusive).

    Args:
        symbol: NSE ticker symbol
        start_date: start date in YYYY-MM-DD format
        end_date: end date in YYYY-MM-DD format
    """
    try:
        mask = (
            (_df["NSESYMBOL"] == symbol.upper())
            & (_df["Date"] >= pd.to_datetime(start_date))
            & (_df["Date"] <= pd.to_datetime(end_date))
        )
        sub = _df[mask].sort_values("Date")[["Date", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        if sub.empty:
            result = f"No data for {symbol} between {start_date} and {end_date}."
        else:
            sub = sub.copy()
            sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
            result = f"{symbol} price history ({start_date} to {end_date}):\n{sub.to_string(index=False)}"
    except Exception as e:
        result = f"Error: {e}"
    return _record_result("get_price_history", result)


def get_summary_stats(symbol: str) -> str:
    """Get summary statistics for a stock: average, max, min closing prices, total volume, etc.

    Args:
        symbol: NSE ticker symbol
    """
    try:
        sub = _df[_df["NSESYMBOL"] == symbol.upper()]
        if sub.empty:
            result = f"No data found for symbol '{symbol}'."
        else:
            result = (
                f"{symbol} summary statistics:\n"
                f"  Trading days:  {len(sub)}\n"
                f"  Avg close:     {sub['CLOSE'].mean():.2f}\n"
                f"  Max close:     {sub['CLOSE'].max():.2f}  on {sub.loc[sub['CLOSE'].idxmax(), 'Date'].strftime('%Y-%m-%d')}\n"
                f"  Min close:     {sub['CLOSE'].min():.2f}  on {sub.loc[sub['CLOSE'].idxmin(), 'Date'].strftime('%Y-%m-%d')}\n"
                f"  52W High:      {sub['52W H'].iloc[-1]:.2f}\n"
                f"  52W Low:       {sub['52W L'].iloc[-1]:.2f}\n"
                f"  Avg volume:    {sub['VOLUME'].mean():,.0f}\n"
                f"  Total volume:  {sub['VOLUME'].sum():,}"
            )
    except Exception as e:
        result = f"Error: {e}"
    return _record_result("get_summary_stats", result)


def compare_stocks_on_date(date: str) -> str:
    """Compare all available stocks on a given trading date — closing price, change, and volume.

    Args:
        date: date in YYYY-MM-DD format
    """
    try:
        dt  = pd.to_datetime(date)
        sub = _df[_df["Date"] == dt].copy()
        if sub.empty:
            result = f"No trading data for {date}. The market may have been closed."
        else:
            sub["Change %"] = ((sub["CLOSE"] - sub["PREV. CLOSE"]) / sub["PREV. CLOSE"] * 100).round(2)
            cols = ["NSESYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "Change %", "VOLUME"]
            sub = sub[cols].sort_values("Change %", ascending=False)
            result = f"All stocks on {date}:\n{sub.to_string(index=False)}"
    except Exception as e:
        result = f"Error: {e}"
    return _record_result("compare_stocks_on_date", result)


def get_top_gainers_losers(date: str, n: int = 3) -> str:
    """Get the top N gaining and top N losing stocks on a specific trading date.

    Args:
        date: date in YYYY-MM-DD format
        n: number of top gainers/losers to return (default 3)
    """
    try:
        dt  = pd.to_datetime(date)
        sub = _df[_df["Date"] == dt].copy()
        if sub.empty:
            result = f"No trading data for {date}."
        else:
            sub["Change %"] = ((sub["CLOSE"] - sub["PREV. CLOSE"]) / sub["PREV. CLOSE"] * 100).round(2)
            gainers = sub.nlargest(n, "Change %")[["NSESYMBOL", "CLOSE", "Change %"]]
            losers  = sub.nsmallest(n, "Change %")[["NSESYMBOL", "CLOSE", "Change %"]]
            result = (
                f"Top {n} gainers on {date}:\n{gainers.to_string(index=False)}\n\n"
                f"Top {n} losers on {date}:\n{losers.to_string(index=False)}"
            )
    except Exception as e:
        result = f"Error: {e}"
    return _record_result("get_top_gainers_losers", result)


def calculate_return(symbol: str, start_date: str, end_date: str) -> str:
    """Calculate the price return (absolute and percentage) for a stock between two dates.

    Args:
        symbol: NSE ticker symbol
        start_date: start date in YYYY-MM-DD format
        end_date: end date in YYYY-MM-DD format
    """
    try:
        sub = _df[_df["NSESYMBOL"] == symbol.upper()].sort_values("Date")
        start = sub[sub["Date"] >= pd.to_datetime(start_date)].head(1)
        end   = sub[sub["Date"] <= pd.to_datetime(end_date)].tail(1)
        if start.empty or end.empty:
            result = f"Insufficient data to calculate return for {symbol} between {start_date} and {end_date}."
        else:
            p0      = start.iloc[0]["CLOSE"]
            p1      = end.iloc[0]["CLOSE"]
            d0      = start.iloc[0]["Date"].strftime("%Y-%m-%d")
            d1      = end.iloc[0]["Date"].strftime("%Y-%m-%d")
            change  = p1 - p0
            pct     = change / p0 * 100
            result = (
                f"{symbol} return from {d0} to {d1}:\n"
                f"  Start price: {p0:.2f}\n"
                f"  End price:   {p1:.2f}\n"
                f"  Change:      {change:+.2f}\n"
                f"  Return:      {pct:+.2f}%"
            )
    except Exception as e:
        result = f"Error: {e}"
    return _record_result("calculate_return", result)


def get_volume_leaders(date: str) -> str:
    """Rank all stocks by trading volume on a specific date.

    Args:
        date: date in YYYY-MM-DD format
    """
    try:
        dt  = pd.to_datetime(date)
        sub = _df[_df["Date"] == dt][["NSESYMBOL", "VOLUME", "No of trades"]].sort_values("VOLUME", ascending=False)
        if sub.empty:
            result = f"No trading data for {date}."
        else:
            result = f"Volume leaders on {date}:\n{sub.to_string(index=False)}"
    except Exception as e:
        result = f"Error: {e}"
    return _record_result("get_volume_leaders", result)


# ── Initialisation ────────────────────────────────────────────────────────────

def load_data(xlsx_path: str) -> None:
    """Load the Excel workbook and build the ReActAgent workflow."""
    global _df, _data_meta, _workflow

    print(f"[1/2] Loading Excel data: {xlsx_path}")
    _df = pd.read_excel(xlsx_path)
    _df["Date"] = pd.to_datetime(_df["Date"])
    symbols   = sorted(_df["NSESYMBOL"].unique().tolist())
    date_min  = _df["Date"].min().strftime("%Y-%m-%d")
    date_max  = _df["Date"].max().strftime("%Y-%m-%d")
    print(f"      {len(_df)} rows | symbols: {', '.join(symbols)} | {date_min} → {date_max}")

    print(f"[2/2] Building ReActAgent workflow ({config.MODEL})")
    llm = Ollama(model=config.MODEL, base_url=config.OLLAMA_HOST, request_timeout=180.0)
    Settings.embed_model = None

    tools = [
        FunctionTool.from_defaults(list_available_symbols),
        FunctionTool.from_defaults(get_price_on_date),
        FunctionTool.from_defaults(get_price_history),
        FunctionTool.from_defaults(get_summary_stats),
        FunctionTool.from_defaults(compare_stocks_on_date),
        FunctionTool.from_defaults(get_top_gainers_losers),
        FunctionTool.from_defaults(calculate_return),
        FunctionTool.from_defaults(get_volume_leaders),
    ]

    system_prompt = (
        "You are a stock market analyst assistant with access to NSE (National Stock Exchange of India) "
        "daily trading data. Use the available tools to look up data before answering. "
        "Always cite specific numbers from the data. If a date falls on a weekend or holiday, "
        "mention that no trading data is available for that date."
    )

    react_agent = ReActAgent(
        name="StockAgent",
        tools=tools,
        llm=llm,
        system_prompt=system_prompt,
        max_iterations=100,
    )

    _workflow = AgentWorkflow(agents=[react_agent], root_agent="StockAgent")

    _data_meta = {
        "symbols":   symbols,
        "date_from": date_min,
        "date_to":   date_max,
        "rows":      len(_df),
    }
    print("Agent ready.")


# ── Public helpers ────────────────────────────────────────────────────────────

def data_info() -> dict:
    return _data_meta


def ask_stream(question: str):
    """Sync generator that streams agent progress as (event_type, payload) tuples.

    Yields
    ------
    ("status",       str)        — status bar text for the current agent step
    ("tool_results", list[dict]) — updated tool call list after each tool completes
    ("text",         str)        — token of the final answer as it streams in
    """
    if _workflow is None:
        raise RuntimeError("Agent not initialised — call load_data() first.")

    _last_tool_results.clear()
    event_queue: queue.Queue = queue.Queue()
    _DONE = object()  # sentinel

    async def _run() -> None:
        handler = _workflow.run(
            user_msg=question,
            chat_history=list(_chat_history),
        )
        final_content = ""
        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                # Format the arguments compactly for the status bar.
                args = ", ".join(f"{k}={v!r}" for k, v in event.tool_kwargs.items())
                event_queue.put(("status", f"Calling {event.tool_name}({args})"))

            elif isinstance(event, ToolCallResult):
                # _record_result() was already called inside the tool function,
                # so _last_tool_results is up to date — just push the snapshot.
                event_queue.put(("tool_results", list(_last_tool_results)))
                event_queue.put(("status", "Synthesising answer…"))

            elif isinstance(event, AgentStream) and event.delta:
                final_content += event.delta
                event_queue.put(("text", event.delta))

        # Persist turns for multi-turn conversation.
        result = await handler
        answer = result.response.content or final_content
        _chat_history.append(ChatMessage(role="user",      content=question))
        _chat_history.append(ChatMessage(role="assistant", content=answer))
        event_queue.put(_DONE)

    thread = threading.Thread(target=lambda: asyncio.run(_run()), daemon=True)
    thread.start()

    while True:
        item = event_queue.get()
        if item is _DONE:
            break
        yield item

    thread.join()


def reset() -> None:
    """Clear conversation history and last tool results."""
    _chat_history.clear()
    _last_tool_results.clear()
