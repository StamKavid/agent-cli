"""
Agent CLI Package - Create agent project structures similar to CrewAI.

This package provides tools for generating AI agent projects with
best practices and comprehensive structure.
"""

__version__ = "0.2.0"
__author__ = "Stamatis Kavidopoulos"

from .cli import AgentProjectCLI
from .exceptions import AgentCLIError, ProjectExistsError, TemplateError

__all__ = [
    "AgentProjectCLI",
    "AgentCLIError", 
    "ProjectExistsError",
    "TemplateError",
]
