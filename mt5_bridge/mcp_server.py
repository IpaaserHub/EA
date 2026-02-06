#!/usr/bin/env python3
"""
MT5 MCP Server for Claude Code
==============================
This MCP server allows Claude to directly control MetaTrader 5.

Tools provided:
- mt5_get_symbols: List available trading symbols
- mt5_get_rates: Download historical price data
- mt5_get_account: Get account information
- mt5_get_positions: Get open positions
- mt5_download_data: Download and save data to CSV
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# MT5 Client
import rpyc

# Create MCP server
server = Server("mt5-server")

# Global connection
_mt5_conn = None

def get_mt5_connection():
    """Get or create MT5 bridge connection."""
    global _mt5_conn
    if _mt5_conn is None:
        try:
            _mt5_conn = rpyc.connect("localhost", 18812)
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to MT5 bridge: {e}\n"
                "Make sure to run: ./mt5_bridge/start_server.sh"
            )
    return _mt5_conn


@server.list_tools()
async def list_tools():
    """List available MT5 tools."""
    return [
        Tool(
            name="mt5_status",
            description="Check MT5 connection status and terminal info",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="mt5_get_symbols",
            description="Get list of available trading symbols. Optionally filter by pattern (e.g., '*JPY*' for JPY pairs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter pattern (e.g., '*JPY*', '*USD*')",
                    },
                },
            },
        ),
        Tool(
            name="mt5_get_rates",
            description="Get historical OHLC price data for a symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol (e.g., XAUJPY, USDJPY)",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN1",
                        "default": "H1",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of candles to retrieve",
                        "default": 100,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="mt5_download_data",
            description="Download historical data and save to CSV file for backtesting. Downloads specified years of data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol (e.g., XAUJPY)",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe (default: H1)",
                        "default": "H1",
                    },
                    "years": {
                        "type": "number",
                        "description": "Years of historical data to download",
                        "default": 2,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="mt5_get_account",
            description="Get trading account information (balance, equity, margin, etc.)",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="mt5_get_positions",
            description="Get current open positions",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Optional: filter by symbol",
                    },
                },
            },
        ),
        Tool(
            name="mt5_get_tick",
            description="Get current price tick for a symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol",
                    },
                },
                "required": ["symbol"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    try:
        if name == "mt5_status":
            try:
                conn = get_mt5_connection()
                info = conn.root.terminal_info()
                account = conn.root.account_info()
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "connected": True,
                        "terminal": info.get("name") if info else "Unknown",
                        "company": info.get("company") if info else "Unknown",
                        "account": account.get("login") if account else "Unknown",
                        "balance": account.get("balance") if account else 0,
                        "currency": account.get("currency") if account else "Unknown",
                    }, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "connected": False,
                        "error": str(e),
                        "hint": "Run ./mt5_bridge/start_server.sh with MT5 open",
                    }, indent=2)
                )]

        elif name == "mt5_get_symbols":
            conn = get_mt5_connection()
            filter_pattern = arguments.get("filter")
            symbols = conn.root.symbols_get(filter_pattern)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "count": len(symbols),
                    "symbols": list(symbols)[:50],  # Limit to 50
                    "note": "Showing first 50" if len(symbols) > 50 else None,
                }, indent=2)
            )]

        elif name == "mt5_get_rates":
            conn = get_mt5_connection()
            symbol = arguments["symbol"]
            timeframe = arguments.get("timeframe", "H1")
            count = arguments.get("count", 100)

            rates = conn.root.copy_rates_from_pos(symbol, timeframe, 0, count)

            if not rates:
                return [TextContent(type="text", text=f"No data for {symbol}")]

            # Format rates
            formatted = []
            for r in rates[-10:]:  # Show last 10
                formatted.append({
                    "time": datetime.fromtimestamp(r[0]).isoformat(),
                    "open": round(r[1], 5),
                    "high": round(r[2], 5),
                    "low": round(r[3], 5),
                    "close": round(r[4], 5),
                })

            return [TextContent(
                type="text",
                text=json.dumps({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "total_candles": len(rates),
                    "latest_candles": formatted,
                }, indent=2)
            )]

        elif name == "mt5_download_data":
            conn = get_mt5_connection()
            symbol = arguments["symbol"]
            timeframe = arguments.get("timeframe", "H1")
            years = arguments.get("years", 2)

            # Calculate date range
            date_to = datetime.now()
            date_from = date_to - timedelta(days=int(years * 365))

            rates = conn.root.copy_rates_range(
                symbol, timeframe,
                date_from.isoformat(),
                date_to.isoformat()
            )

            if not rates:
                return [TextContent(type="text", text=f"No data returned for {symbol}")]

            # Save to CSV
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            os.makedirs(data_dir, exist_ok=True)

            filename = f"{symbol}_{timeframe}_extended.csv"
            filepath = os.path.join(data_dir, filename)

            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Open", "High", "Low", "Close", "Change", "Change%"])
                for r in rates:
                    dt = datetime.fromtimestamp(r[0])
                    open_p, high, low, close = r[1], r[2], r[3], r[4]
                    change = close - open_p
                    change_pct = (change / open_p * 100) if open_p else 0
                    writer.writerow([
                        dt.strftime("%Y.%m.%d %H:%M"),
                        f"{open_p:.5f}",
                        f"{high:.5f}",
                        f"{low:.5f}",
                        f"{close:.5f}",
                        f"{change:.5f}",
                        f"{change_pct:.2f}",
                    ])

            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles": len(rates),
                    "date_range": f"{date_from.date()} to {date_to.date()}",
                    "saved_to": filepath,
                }, indent=2)
            )]

        elif name == "mt5_get_account":
            conn = get_mt5_connection()
            account = conn.root.account_info()
            if account:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "login": account.get("login"),
                        "server": account.get("server"),
                        "balance": account.get("balance"),
                        "equity": account.get("equity"),
                        "margin": account.get("margin"),
                        "free_margin": account.get("margin_free"),
                        "currency": account.get("currency"),
                        "leverage": account.get("leverage"),
                    }, indent=2)
                )]
            return [TextContent(type="text", text="Could not get account info")]

        elif name == "mt5_get_positions":
            conn = get_mt5_connection()
            symbol = arguments.get("symbol")
            positions = conn.root.positions_get(symbol)

            if not positions:
                return [TextContent(type="text", text="No open positions")]

            formatted = []
            for p in positions:
                formatted.append({
                    "ticket": p.get("ticket"),
                    "symbol": p.get("symbol"),
                    "type": "BUY" if p.get("type") == 0 else "SELL",
                    "volume": p.get("volume"),
                    "price_open": p.get("price_open"),
                    "price_current": p.get("price_current"),
                    "profit": p.get("profit"),
                })

            return [TextContent(
                type="text",
                text=json.dumps({"positions": formatted}, indent=2)
            )]

        elif name == "mt5_get_tick":
            conn = get_mt5_connection()
            symbol = arguments["symbol"]
            tick = conn.root.symbol_info_tick(symbol)

            if tick:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "symbol": symbol,
                        "bid": tick.get("bid"),
                        "ask": tick.get("ask"),
                        "last": tick.get("last"),
                        "time": datetime.fromtimestamp(tick.get("time", 0)).isoformat(),
                    }, indent=2)
                )]
            return [TextContent(type="text", text=f"No tick data for {symbol}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
