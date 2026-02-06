#!/bin/bash
# Start the MT5 Bridge Server
# Make sure MT5 is running first!

echo "=== MT5 Bridge Server ==="
echo ""
echo "Make sure MetaTrader 5 is running in Wine first!"
echo ""

# Find Wine Python
WINE_PYTHON=$(find ~/.wine -name "python.exe" 2>/dev/null | grep -i "python3" | head -1)

if [ -z "$WINE_PYTHON" ]; then
    echo "ERROR: Wine Python not found. Run setup_wine_python.sh first."
    exit 1
fi

echo "Using: $WINE_PYTHON"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start server
cd "$SCRIPT_DIR"
wine "$WINE_PYTHON" wine_server.py
