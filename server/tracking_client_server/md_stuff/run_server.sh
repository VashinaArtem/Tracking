#!/bin/bash
# Start MiniCPM-V Tracking Server
# Run this on Linux server

echo "========================================"
echo "MiniCPM-V Tracking Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv_server" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please run the following commands first:"
    echo "  python -m venv venv_server"
    echo "  source venv_server/bin/activate"
    echo "  pip install -r server_requirements.txt"
    echo ""
    exit 1
fi

# Activate virtual environment
echo "[1/2] Activating virtual environment..."
source venv_server/bin/activate

# Get IP address
IP=$(hostname -I | awk '{print $1}')
echo "[2/2] Starting server..."
echo ""
echo "Server will be accessible at:"
echo "  http://localhost:8000"
echo "  http://$IP:8000"
echo ""
echo "API docs: http://$IP:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run server
python tracking_server.py --host 0.0.0.0 --port 8000
