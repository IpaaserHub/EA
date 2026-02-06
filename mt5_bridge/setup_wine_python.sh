#!/bin/bash
# Setup Python in Wine for MT5 connection
# Run this once to set up the bridge

set -e

echo "=== MT5 Linux Bridge Setup ==="
echo ""

# Check Wine
if ! command -v wine &> /dev/null; then
    echo "ERROR: Wine is not installed. Install with: sudo pacman -S wine"
    exit 1
fi

echo "[1/4] Downloading Python for Windows..."
PYTHON_URL="https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
PYTHON_INSTALLER="/tmp/python_installer.exe"

if [ ! -f "$PYTHON_INSTALLER" ]; then
    wget -q --show-progress "$PYTHON_URL" -O "$PYTHON_INSTALLER"
fi

echo ""
echo "[2/4] Installing Python in Wine..."
echo "IMPORTANT: Check 'Add Python to PATH' during installation!"
wine "$PYTHON_INSTALLER" /quiet InstallAllUsers=0 PrependPath=1

# Wait for installation
sleep 5

echo ""
echo "[3/4] Finding Wine Python..."
WINE_PYTHON=$(find ~/.wine -name "python.exe" 2>/dev/null | grep -i "python310" | head -1)

if [ -z "$WINE_PYTHON" ]; then
    WINE_PYTHON=$(find ~/.wine -name "python.exe" 2>/dev/null | head -1)
fi

if [ -z "$WINE_PYTHON" ]; then
    echo "ERROR: Could not find Wine Python. Try manual installation."
    exit 1
fi

echo "Found: $WINE_PYTHON"

echo ""
echo "[4/4] Installing MetaTrader5 and rpyc in Wine Python..."
wine "$WINE_PYTHON" -m pip install --upgrade pip
wine "$WINE_PYTHON" -m pip install MetaTrader5 rpyc

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Wine Python path: $WINE_PYTHON"
echo ""
echo "To start the bridge server, run:"
echo "  ./mt5_bridge/start_server.sh"
