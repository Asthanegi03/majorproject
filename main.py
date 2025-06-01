"""
Main entry point for the Audio Deepfake Detection System

This file imports and runs the Flask application defined in app.py.

Author: DeepfakeSoundShield Team
Date: May 2025
"""

from app import app

if __name__ == "__main__":
    # Run the Flask application
    # Use 0.0.0.0 to make the app accessible externally
    app.run(host="0.0.0.0", port=5000, debug=True)