#!/usr/bin/env python3
"""
Run script for ViT Image Retrieval Web Application

This script can be run directly or via python -m vit_image_retrieval.web.run
"""

import os
import sys
from pathlib import Path

def ensure_package_path():
    """Ensure the vit_image_retrieval package can be imported."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # The package root should be two levels up from this file
    # src/vit_image_retrieval/web/run.py -> src/vit_image_retrieval/ -> src/
    src_dir = script_dir.parent.parent
    repo_root = src_dir.parent
    
    # Add src directory to path if not already there
    src_path = str(src_dir.absolute())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Also add repo root for good measure
    repo_path = str(repo_root.absolute())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    
    # Set working directory to repo root so that uploads/indexes folders are created there
    os.chdir(repo_root)

def main():
    """Main entry point for the web application."""
    # Ensure package can be imported
    ensure_package_path()
    
    # Now import the app
    from vit_image_retrieval.web.app import create_app
    
    # Create the Flask app
    app = create_app()
    
    # Configure the app
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    # Get host and port from environment or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    
    print(f"Starting ViT Image Retrieval Web Server...")
    print(f"Repository root: {Path.cwd()}")
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
