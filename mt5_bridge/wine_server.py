"""
MT5 Bridge Server - Runs in Wine Python
========================================
This script runs inside Wine and exposes MT5 functions via rpyc.

Start with: wine python.exe wine_server.py
"""

import rpyc
from rpyc.utils.server import ThreadedServer
import MetaTrader5 as mt5

class MT5Service(rpyc.Service):
    """RPyC service exposing MT5 functions."""

    def on_connect(self, conn):
        print("Client connected")

    def on_disconnect(self, conn):
        print("Client disconnected")

    # === Initialization ===
    def exposed_initialize(self):
        return mt5.initialize()

    def exposed_shutdown(self):
        return mt5.shutdown()

    def exposed_terminal_info(self):
        info = mt5.terminal_info()
        if info:
            return info._asdict()
        return None

    def exposed_account_info(self):
        info = mt5.account_info()
        if info:
            return info._asdict()
        return None

    # === Symbol Info ===
    def exposed_symbols_total(self):
        return mt5.symbols_total()

    def exposed_symbols_get(self, group=None):
        if group:
            symbols = mt5.symbols_get(group=group)
        else:
            symbols = mt5.symbols_get()
        if symbols:
            return [s.name for s in symbols]
        return []

    def exposed_symbol_info(self, symbol):
        info = mt5.symbol_info(symbol)
        if info:
            return info._asdict()
        return None

    def exposed_symbol_info_tick(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return tick._asdict()
        return None

    # === Historical Data ===
    def exposed_copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        """Get historical OHLC data."""
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
        if rates is not None:
            return rates.tolist()
        return None

    def exposed_copy_rates_range(self, symbol, timeframe, date_from, date_to):
        """Get historical data for date range."""
        from datetime import datetime
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

        if isinstance(date_from, str):
            date_from = datetime.fromisoformat(date_from)
        if isinstance(date_to, str):
            date_to = datetime.fromisoformat(date_to)

        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is not None:
            return rates.tolist()
        return None

    # === Trading ===
    def exposed_positions_total(self):
        return mt5.positions_total()

    def exposed_positions_get(self, symbol=None):
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        if positions:
            return [p._asdict() for p in positions]
        return []

    def exposed_orders_total(self):
        return mt5.orders_total()

    def exposed_history_deals_total(self, date_from, date_to):
        from datetime import datetime
        if isinstance(date_from, str):
            date_from = datetime.fromisoformat(date_from)
        if isinstance(date_to, str):
            date_to = datetime.fromisoformat(date_to)
        return mt5.history_deals_total(date_from, date_to)

    # === Error Info ===
    def exposed_last_error(self):
        return mt5.last_error()


if __name__ == "__main__":
    print("=" * 50)
    print("MT5 Bridge Server")
    print("=" * 50)

    # Initialize MT5
    if not mt5.initialize():
        print("ERROR: Failed to initialize MT5")
        print("Make sure MetaTrader 5 is running!")
        exit(1)

    info = mt5.terminal_info()
    print(f"Connected to: {info.name}")
    print(f"Company: {info.company}")
    print()

    # Start server
    PORT = 18812
    print(f"Starting server on port {PORT}...")
    server = ThreadedServer(MT5Service, port=PORT)
    print(f"Server ready! Waiting for connections...")
    print()
    print("Keep this window open while using the optimizer.")
    print("Press Ctrl+C to stop.")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        mt5.shutdown()
