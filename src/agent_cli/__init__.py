"""
Agent CLI Package - Clean AI Agent Project Scaffolding.

This package provides focused tools for generating AI agent projects
following Python best practices and clean architecture principles.
"""

__version__ = "0.3.0"
__author__ = "Stamatis Kavidopoulos"

# Import zero-config CLI as the main entry point
from .cli_zero_config import main
from .exceptions import AgentCLIError, ProjectExistsError, TemplateError

__all__ = [
    "main",
    "AgentCLIError", 
    "ProjectExistsError",
    "TemplateError",
]
