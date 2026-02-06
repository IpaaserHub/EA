"""
MT5 Client - Connects to Wine Bridge Server
============================================
This module connects to the MT5 bridge server running in Wine
and provides a clean API for the optimizer to use.

Usage:
    from mt5.client import MT5Client

    client = MT5Client()
    if client.connect():
        prices = client.get_rates("XAUJPY", "H1", 17520)  # 2 years
        client.save_to_csv(prices, "data/XAUJPY_H1.csv")
"""

import os
import csv
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MT5Client:
    """Client for connecting to MT5 via the Wine bridge."""

    def __init__(self, host: str = "localhost", port: int = 18812):
        """
        Initialize MT5 client.

        Args:
            host: Bridge server host
            port: Bridge server port (default 18812)
        """
        self.host = host
        self.port = port
        self._conn = None

    def connect(self) -> bool:
        """
        Connect to the MT5 bridge server.

        Returns:
            True if connected successfully
        """
        try:
            import rpyc
            self._conn = rpyc.connect(self.host, self.port)
            logger.info(f"Connected to MT5 bridge at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MT5 bridge: {e}")
            logger.error("Make sure the bridge server is running: ./mt5_bridge/start_server.sh")
            return False

    def disconnect(self):
        """Disconnect from the bridge server."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Disconnected from MT5 bridge")

    def is_connected(self) -> bool:
        """Check if connected to bridge."""
        return self._conn is not None

    def _check_connection(self):
        """Raise error if not connected."""
        if not self._conn:
            raise ConnectionError("Not connected to MT5 bridge. Call connect() first.")

    # === Terminal Info ===

    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """Get terminal information."""
        self._check_connection()
        return self._conn.root.terminal_info()

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        self._check_connection()
        return self._conn.root.account_info()

    # === Symbols ===

    def get_symbols(self, group: Optional[str] = None) -> List[str]:
        """
        Get available symbols.

        Args:
            group: Filter by group (e.g., "*JPY*")

        Returns:
            List of symbol names
        """
        self._check_connection()
        return list(self._conn.root.symbols_get(group))

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        self._check_connection()
        return self._conn.root.symbol_info(symbol)

    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick for symbol."""
        self._check_connection()
        return self._conn.root.symbol_info_tick(symbol)

    # === Historical Data ===

    def get_rates(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLC data.

        Args:
            symbol: Trading symbol (e.g., "XAUJPY")
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            count: Number of candles to retrieve

        Returns:
            List of candle dicts with time, open, high, low, close, volume
        """
        self._check_connection()
        rates = self._conn.root.copy_rates_from_pos(symbol, timeframe, 0, count)

        if not rates:
            return []

        # Convert to list of dicts
        result = []
        for r in rates:
            result.append({
                "time": datetime.fromtimestamp(r[0]),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "tick_volume": r[5],
                "spread": r[6],
                "real_volume": r[7],
            })

        return result

    def get_rates_range(
        self,
        symbol: str,
        timeframe: str = "H1",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a date range.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            date_from: Start date (default: 2 years ago)
            date_to: End date (default: now)

        Returns:
            List of candle dicts
        """
        self._check_connection()

        if date_to is None:
            date_to = datetime.now()
        if date_from is None:
            date_from = date_to - timedelta(days=730)  # 2 years

        rates = self._conn.root.copy_rates_range(
            symbol,
            timeframe,
            date_from.isoformat(),
            date_to.isoformat(),
        )

        if not rates:
            return []

        result = []
        for r in rates:
            result.append({
                "time": datetime.fromtimestamp(r[0]),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "tick_volume": r[5],
                "spread": r[6],
                "real_volume": r[7],
            })

        return result

    # === Trading ===

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions."""
        self._check_connection()
        positions = self._conn.root.positions_get(symbol)
        return [dict(p) for p in positions] if positions else []

    # === Utility ===

    def save_to_csv(
        self,
        rates: List[Dict[str, Any]],
        filepath: str,
        format: str = "extended",
    ):
        """
        Save rates to CSV file.

        Args:
            rates: List of rate dicts from get_rates()
            filepath: Output file path
            format: "extended" (for our optimizer) or "mt5" (MT5 native)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            if format == "extended":
                writer = csv.writer(f)
                writer.writerow(["Date", "Open", "High", "Low", "Close", "Change", "Change%"])
                for r in rates:
                    change = r["close"] - r["open"]
                    change_pct = (change / r["open"] * 100) if r["open"] else 0
                    writer.writerow([
                        r["time"].strftime("%Y.%m.%d %H:%M"),
                        f"{r['open']:.5f}",
                        f"{r['high']:.5f}",
                        f"{r['low']:.5f}",
                        f"{r['close']:.5f}",
                        f"{change:.5f}",
                        f"{change_pct:.2f}",
                    ])
            else:  # mt5 format
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["Date", "Time", "Open", "High", "Low", "Close", "Volume"])
                for r in rates:
                    writer.writerow([
                        r["time"].strftime("%Y.%m.%d"),
                        r["time"].strftime("%H:%M"),
                        r["open"],
                        r["high"],
                        r["low"],
                        r["close"],
                        r["tick_volume"],
                    ])

        logger.info(f"Saved {len(rates)} candles to {filepath}")


def download_symbol_data(
    symbol: str,
    timeframe: str = "H1",
    years: int = 2,
    output_dir: str = "data",
) -> Optional[str]:
    """
    Convenience function to download symbol data.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe (default H1)
        years: Years of data to download
        output_dir: Output directory

    Returns:
        Path to saved file, or None if failed
    """
    client = MT5Client()

    if not client.connect():
        print("Failed to connect to MT5 bridge.")
        print("1. Make sure MT5 is running in Wine")
        print("2. Start the bridge: ./mt5_bridge/start_server.sh")
        return None

    try:
        print(f"Downloading {years} years of {timeframe} data for {symbol}...")

        date_to = datetime.now()
        date_from = date_to - timedelta(days=years * 365)

        rates = client.get_rates_range(symbol, timeframe, date_from, date_to)

        if not rates:
            print(f"No data returned for {symbol}")
            return None

        print(f"Downloaded {len(rates)} candles")

        # Save to file
        filename = f"{symbol}_{timeframe}_extended.csv"
        filepath = os.path.join(output_dir, filename)
        client.save_to_csv(rates, filepath)

        print(f"Saved to {filepath}")
        return filepath

    finally:
        client.disconnect()


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MT5 data")
    parser.add_argument("symbol", help="Trading symbol (e.g., XAUJPY)")
    parser.add_argument("-t", "--timeframe", default="H1", help="Timeframe (default: H1)")
    parser.add_argument("-y", "--years", type=int, default=2, help="Years of data (default: 2)")
    parser.add_argument("-o", "--output", default="data", help="Output directory")

    args = parser.parse_args()

    download_symbol_data(args.symbol, args.timeframe, args.years, args.output)
