#!/usr/bin/env python3
"""
Main entry point for running the ViT Image Retrieval Web Application

This script ensures the package can be imported and the working directory is set correctly.
It can be run directly: python run_web_app.py
"""

import os
import sys
from pathlib import Path

# Get the directory where this script is located
script_path = Path(__file__).absolute()
repo_root = script_path.parent

# Set working directory to repository root
os.chdir(repo_root)

# Add src directory to Python path
src_path = str(repo_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import and run the app
from vit_image_retrieval.web.app import create_app

if __name__ == '__main__':
    app = create_app()
    
    # Configure the app
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    # Get host and port from environment or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    
    print(f"Starting ViT Image Retrieval Web Server...")
    print(f"Repository root: {repo_root}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {app.config['DEBUG']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Index folder: {app.config['INDEX_FOLDER']}")
    print()
    print("Open your browser and navigate to: http://localhost:{}".format(port))
    print()
    
    # Run the app
    app.run(host=host, port=port, debug=app.config['DEBUG'])
