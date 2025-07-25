"""
Template system for the Agent CLI.

This package provides a modular template system for generating
AI agent project structures.
"""

from .static_templates import StaticTemplateProvider
from .template_categories import (
    FileTemplates,
    NotebookTemplates,
    ToolTemplates,
    TestTemplates,
    SourceTemplates
)

__all__ = [
    "StaticTemplateProvider",
    "FileTemplates",
    "NotebookTemplates", 
    "ToolTemplates",
    "TestTemplates",
    "SourceTemplates"
]
