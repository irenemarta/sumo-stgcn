# Variables
python := "uv run python"
VENV_DIR := ".venv"
REQUIREMENTS_FILE := "requirements.txt"

# Defualt task
default: help

init:
    bash dependencies.sh

clean:
    rm -rf programmi/__pycache__
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

help:
    @just --list 