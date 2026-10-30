#!/usr/bin/env python3
"""
Run script for ViT Image Retrieval Web Application
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vit_image_retrieval.web.app import create_app

def main():
    """Main entry point for the web application."""
    # Create the Flask app
    app = create_app()
    
    # Configure the app
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    # Get host and port from environment or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    
    print(f"Starting ViT Image Retrieval Web Server...")
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

if __name__ == '__main__':
    main()
