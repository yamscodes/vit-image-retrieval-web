#!/usr/bin/env python3
"""
Setup script for ViT Image Retrieval Web
"""

from setuptools import setup, find_packages
import os

# Read the contents of requirements.txt
with open('requirements-web.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='vit-image-retrieval-web',
    version='1.0.0',
    description='Web-based interface for ViT Image Retrieval system using Flask',
    author='Dr. Sreenivas Bhattiprolu',
    author_email='',
    url='https://github.com/yamscodes/vit-image-retrieval-web',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'vit-web=vit_image_retrieval.web.run:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Internet :: WWW/HTTP :: WSGI Application',
    ],
)
