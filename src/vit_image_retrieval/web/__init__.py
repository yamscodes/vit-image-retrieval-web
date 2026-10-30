# Web module for ViT Image Retrieval
"""
Web-based interface for ViT Image Retrieval system.
Ported from PyQt5 desktop GUI to Flask web application.
"""

from .app import create_app

__all__ = ['create_app']
