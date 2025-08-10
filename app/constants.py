"""
Application Constants.

This module defines constants and enums used throughout the application,
including message templates and error message definitions.
"""

from enum import Enum


class MESSAGES(str, Enum):
    """Standard application messages."""
    DEFAULT = lambda msg="": f"{msg if msg else ''}"


class ERROR_MESSAGES(str, Enum):
    """Error message templates for the application."""
    
    def __str__(self) -> str:
        return super().__str__()

    DEFAULT = lambda err="": f"Something went wrong :/\n{err if err else ''}"
    PANDOC_NOT_INSTALLED = "Pandoc is not installed on the server. Please contact your administrator for assistance."
    OPENAI_NOT_FOUND = lambda name="": f"OpenAI API was not found"
    OLLAMA_NOT_FOUND = "WebUI could not connect to Ollama"
    FILE_NOT_FOUND = "The specified file was not found."