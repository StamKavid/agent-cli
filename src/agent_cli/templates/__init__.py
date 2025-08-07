"""
Template system for Agent CLI.
"""

from .project_templates import ProjectTemplateManager
from .file_templates import FileTemplateManager

__all__ = ["ProjectTemplateManager", "FileTemplateManager"]
