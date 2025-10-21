"""
Setup script for NHL SOG prediction system.

Install with:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="nhl-sog-predictor",
    version="0.1.0",
    description="NHL Shots on Goal prediction system with calibrated uncertainty",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/nhl-sog-predictor",
    packages=find_packages(exclude=["tests", "notebooks"]),
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "nhl-train=train_model:main",
            "nhl-predict=scripts.predict_today:main",
        ],
    },
)