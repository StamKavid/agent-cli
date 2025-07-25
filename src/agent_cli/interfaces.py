"""
Abstract base classes and interfaces for the Agent CLI.

This module defines the contracts that implementations must follow,
adhering to the Interface Segregation Principle (ISP) and 
Dependency Inversion Principle (DIP) from SOLID.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List


class TemplateProvider(ABC):
    """Abstract interface for template providers."""
    
    @abstractmethod
    def get_template(self, template_name: str) -> str:
        """Get a template by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template content as string
            
        Raises:
            TemplateError: If template cannot be found or loaded
        """
        pass
    
    @abstractmethod
    def get_available_templates(self) -> List[str]:
        """Get list of available template names.
        
        Returns:
            List of available template names
        """
        pass


class DirectoryCreator(ABC):
    """Abstract interface for directory creation."""
    
    @abstractmethod
    def create_directory_structure(
        self, 
        base_path: Path, 
        directories: List[str],
        project_name: str
    ) -> None:
        """Create directory structure.
        
        Args:
            base_path: Base path where directories should be created
            directories: List of directory patterns to create
            project_name: Name of the project for template substitution
            
        Raises:
            DirectoryCreationError: If directory creation fails
        """
        pass


class FileGenerator(ABC):
    """Abstract interface for file generation."""
    
    @abstractmethod
    def generate_file(
        self,
        file_path: Path,
        template_name: str,
        context: Dict[str, Any]
    ) -> None:
        """Generate a file from a template.
        
        Args:
            file_path: Path where file should be created
            template_name: Name of the template to use
            context: Variables for template substitution
            
        Raises:
            FileCreationError: If file creation fails
            TemplateError: If template processing fails
        """
        pass


class ProjectValidator(ABC):
    """Abstract interface for project validation."""
    
    @abstractmethod
    def validate_project_name(self, project_name: str) -> bool:
        """Validate project name.
        
        Args:
            project_name: Name to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_output_directory(self, output_dir: str) -> bool:
        """Validate output directory.
        
        Args:
            output_dir: Directory path to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass


class ProjectCreator(ABC):
    """Abstract interface for project creation orchestration."""
    
    @abstractmethod
    def create_project(
        self,
        project_name: str,
        output_dir: str
    ) -> bool:
        """Create a new project.
        
        Args:
            project_name: Name of the project to create
            output_dir: Directory where project should be created
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ProjectExistsError: If project already exists
            AgentCLIError: For other creation errors
        """
        pass
