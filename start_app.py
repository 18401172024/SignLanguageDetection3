"""
SignSense AI Launcher
Starts the Flask server and opens the web application in your default browser.
"""

import os
import sys
import time
import webbrowser
import threading

# Set temporary directory for Matplotlib cache to avoid warnings
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from app import app

def open_browser():
    time.sleep(1.5)
    url = "http://localhost:5000"
    print(f"\nOpening browser at: {url}\n")
    webbrowser.open(url)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("   SignSense AI - Real-Time Sign Language Detection")
    print(f"   Server running on: http://localhost:{port}")
    print("   Press CTRL+C to stop the server")
    print("=" * 60)

    # Launch browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
