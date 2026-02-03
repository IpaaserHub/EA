"""
Data Loader
===========
Load price data from CSV files exported by MetaTrader 5.

Supports multiple CSV formats:
1. MT5 format: Date Time,Open,High,Low,Close,Volume,Spread
2. Extended format: Date,Open,High,Low,Close,Change,Change%
"""

import os
from typing import List, Dict, Optional
from datetime import datetime


def load_mt5_csv(csv_path: str) -> List[Dict]:
    """
    Load price data from MT5-exported CSV file.

    MT5 format (may have BOM/encoding issues):
    2025.01.01 00:00,147677.00000,147677.00000,146857.00000,147035.00000,739,0

    Args:
        csv_path: Path to CSV file

    Returns:
        List of OHLC dicts sorted by time (oldest first)
    """
    prices = []

    if not os.path.exists(csv_path):
        return []

    # Try different encodings
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'latin-1']

    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Remove null bytes (common in MT5 exports)
            content = content.replace('\x00', '')

            lines = content.strip().split('\n')

            for line in lines:
                # Skip empty lines and headers
                line = line.strip()
                if not line or 'Date' in line or 'Historical' in line:
                    continue

                row = line.split(',')
                if len(row) >= 5:
                    try:
                        prices.append({
                            "datetime": row[0].strip(),
                            "open": float(row[1].strip()),
                            "high": float(row[2].strip()),
                            "low": float(row[3].strip()),
                            "close": float(row[4].strip()),
                        })
                    except (ValueError, IndexError):
                        continue

            if prices:
                break  # Found working encoding

        except (UnicodeDecodeError, UnicodeError):
            continue

    return prices


def load_extended_csv(csv_path: str) -> List[Dict]:
    """
    Load price data from extended format CSV.

    Extended format (with header):
    Date,Open,High,Low,Close,Change(Pips),Change(%)
    01/05/2026 00:00,686434.0,690933.0,686180.0,690550.0,4116.0,0.6

    Note: Data may be in reverse order (newest first)

    Args:
        csv_path: Path to CSV file

    Returns:
        List of OHLC dicts sorted by time (oldest first)
    """
    prices = []

    if not os.path.exists(csv_path):
        return []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Skip header lines
        data_start = 0
        for i, line in enumerate(lines):
            if 'Date' in line and 'Open' in line:
                data_start = i + 1
                break
            elif line.strip() and not any(x in line for x in ['Historical', 'XAUJPY', 'BTCJPY']):
                data_start = i
                break

        for line in lines[data_start:]:
            row = line.strip().split(',')
            if len(row) >= 5:
                try:
                    prices.append({
                        "datetime": row[0].strip(),
                        "open": float(row[1].strip()),
                        "high": float(row[2].strip()),
                        "low": float(row[3].strip()),
                        "close": float(row[4].strip()),
                    })
                except (ValueError, IndexError):
                    continue

    except Exception:
        return []

    # Check if data is in reverse order (newest first) and fix
    if len(prices) >= 2:
        # Simple heuristic: if first date > last date, reverse
        first = prices[0]["datetime"]
        last = prices[-1]["datetime"]
        if first > last:  # String comparison works for most date formats
            prices.reverse()

    return prices


def load_prices(csv_path: str) -> List[Dict]:
    """
    Auto-detect format and load price data.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of OHLC dicts sorted by time (oldest first)
    """
    # Try extended format first (cleaner)
    prices = load_extended_csv(csv_path)

    # Fall back to MT5 format
    if not prices:
        prices = load_mt5_csv(csv_path)

    return prices


def find_data_file(symbol: str, data_dir: str = "data") -> Optional[str]:
    """
    Find the best data file for a symbol.

    Looks for files in order of preference:
    1. {symbol}_H1_extended.csv (cleanest format)
    2. {symbol}H1.csv (MT5 export)
    3. Any file containing the symbol name

    Args:
        symbol: Trading symbol (e.g., "XAUJPY")
        data_dir: Directory containing data files

    Returns:
        Path to data file, or None if not found
    """
    if not os.path.exists(data_dir):
        return None

    # Preferred file patterns
    patterns = [
        f"{symbol}_H1_extended.csv",
        f"{symbol}H1.csv",
        f"{symbol}_H1.csv",
        f"{symbol}.csv",
    ]

    for pattern in patterns:
        path = os.path.join(data_dir, pattern)
        if os.path.exists(path):
            return path

    # Search for any file containing symbol name
    for filename in os.listdir(data_dir):
        if symbol in filename and filename.endswith('.csv'):
            return os.path.join(data_dir, filename)

    return None


def get_available_symbols(data_dir: str = "data") -> List[str]:
    """
    Get list of symbols with available data files.

    Args:
        data_dir: Directory containing data files

    Returns:
        List of symbol names
    """
    if not os.path.exists(data_dir):
        return []

    symbols = set()
    known_symbols = ["XAUJPY", "BTCJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"]

    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        # Extract symbol from filename
        for symbol in known_symbols:
            if symbol in filename.upper():
                symbols.add(symbol)
                break

    return sorted(list(symbols))
