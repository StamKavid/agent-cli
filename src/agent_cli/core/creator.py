"""
Clean project creator - no over-engineering.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from .templates import TemplateManager
from .validator import NameValidator, PathValidator

logger = logging.getLogger(__name__)


class ProjectCreator:
    """Ultra-simple project creator following Claude Code patterns."""
    
    def __init__(self):
        self.template_manager = TemplateManager()
        self.name_validator = NameValidator()
        self.path_validator = PathValidator()
    
    def create(self, name: str, template: str, output_dir: str) -> bool:
        """Create a new agent project.
        
        Args:
            name: Project name (can contain hyphens for directory)
            template: Template to use (basic, conversational, research, automation)
            output_dir: Directory to create project in
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Always validate paths, but be more permissive with names
            if not self.path_validator.is_valid(output_dir):
                logger.error(f"Invalid output directory: {output_dir}")
                return False
            
            # Setup paths
            output_path = Path(output_dir).resolve()
            project_path = output_path / name
            
            # Check if project already exists
            if project_path.exists():
                logger.error(f"Project already exists: {project_path}")
                return False
            
            # Create project structure
            return self._create_structure(project_path, name, template)
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False
    
    def _create_structure(self, project_path: Path, name: str, template: str) -> bool:
        """Create the actual project structure."""
        try:
            # Use the modern template system for complete project creation
            self.template_manager.create_project_structure(project_path, template, name)
            
            logger.info(f"Successfully created project: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create structure: {e}")
            # Cleanup on failure
            if project_path.exists():
                import shutil
                shutil.rmtree(project_path, ignore_errors=True)
            return False
