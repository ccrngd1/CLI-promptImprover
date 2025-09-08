#!/usr/bin/env python3
"""
Setup script for the Bedrock Prompt Optimizer.

This package provides a comprehensive system for iteratively improving prompts
for Amazon Bedrock using multi-agent collaboration and LLM-based orchestration.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required.")

# Read the README file for long description
def read_readme():
    """Read README.md for long description."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A multi-agent system for optimizing prompts for Amazon Bedrock."

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    requirements = []
    
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and development dependencies
                if (line and not line.startswith("#") and 
                    not line.startswith("pytest") and 
                    not line.startswith("black") and
                    not line.startswith("flake8") and
                    not line.startswith("mypy") and
                    not line.startswith("sphinx") and
                    not line.startswith("bandit") and
                    not line.startswith("safety") and
                    not line.startswith("memory-profiler") and
                    not line.startswith("line-profiler")):
                    # Handle conditional requirements
                    if ";" in line:
                        req, condition = line.split(";", 1)
                        requirements.append(f"{req.strip()};{condition.strip()}")
                    else:
                        requirements.append(line)
    
    return requirements

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "isort>=5.12.0",
    "bandit>=1.7.0",
    "safety>=2.3.0",
]

# Documentation requirements
docs_requirements = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
]

# Performance profiling requirements
profiling_requirements = [
    "memory-profiler>=0.60.0",
    "line-profiler>=4.0.0",
]

# Optional ML requirements
ml_requirements = [
    "scikit-learn>=1.2.0",
    "numpy>=1.24.0",
    "nltk>=3.8.0",
]

# Get version from __init__.py
def get_version():
    """Get version from __init__.py."""
    init_path = Path(__file__).parent / "__init__.py"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="bedrock-prompt-optimizer",
    version=get_version(),
    author="Bedrock Prompt Optimizer Team",
    author_email="support@example.com",
    description="Multi-agent system for optimizing prompts for Amazon Bedrock",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/bedrock-prompt-optimizer",
    project_urls={
        "Bug Reports": "https://github.com/example/bedrock-prompt-optimizer/issues",
        "Source": "https://github.com/example/bedrock-prompt-optimizer",
        "Documentation": "https://bedrock-prompt-optimizer.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "samples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "profiling": profiling_requirements,
        "ml": ml_requirements,
        "all": dev_requirements + docs_requirements + profiling_requirements + ml_requirements,
    },
    entry_points={
        "console_scripts": [
            "bedrock-optimizer=cli.main:main",
            "bedrock-prompt-optimizer=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "best_practices": ["data/*.json", "data/*.yaml"],
        "cli": ["*.yaml", "*.json"],
        "": ["*.md", "*.txt", "*.yaml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "bedrock", "prompt-engineering", "ai", "llm", "optimization", 
        "multi-agent", "aws", "machine-learning", "nlp"
    ],
    platforms=["any"],
    license="MIT",
    
    # Additional metadata
    maintainer="Bedrock Prompt Optimizer Team",
    maintainer_email="support@example.com",
    
    # Custom commands
    cmdclass={},
    
    # Test suite
    test_suite="tests",
    tests_require=dev_requirements,
    
    # Options for different installation methods
    options={
        "bdist_wheel": {
            "universal": False,
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)