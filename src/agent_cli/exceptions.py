"""
Custom exceptions for the Agent CLI package.
"""

from typing import Optional


class AgentCLIError(Exception):
    """Base exception class for Agent CLI errors."""
    
    def __init__(self, message: str, details: Optional[str] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message
            details: Optional additional details about the error
        """
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ProjectExistsError(AgentCLIError):
    """Raised when trying to create a project that already exists."""
    
    def __init__(self, project_path: str) -> None:
        """Initialize the exception.
        
        Args:
            project_path: Path where the project already exists
        """
        message = f"Project already exists at: {project_path}"
        super().__init__(message)
        self.project_path = project_path


class TemplateError(AgentCLIError):
    """Raised when there's an error with template processing."""
    
    def __init__(self, template_name: str, details: Optional[str] = None) -> None:
        """Initialize the exception.
        
        Args:
            template_name: Name of the template that caused the error
            details: Optional additional details about the error
        """
        message = f"Template error in '{template_name}'"
        super().__init__(message, details)
        self.template_name = template_name


class DirectoryCreationError(AgentCLIError):
    """Raised when directory creation fails."""
    
    def __init__(self, directory_path: str, details: Optional[str] = None) -> None:
        """Initialize the exception.
        
        Args:
            directory_path: Path where directory creation failed
            details: Optional additional details about the error
        """
        message = f"Failed to create directory: {directory_path}"
        super().__init__(message, details)
        self.directory_path = directory_path


class FileCreationError(AgentCLIError):
    """Raised when file creation fails."""
    
    def __init__(self, file_path: str, details: Optional[str] = None) -> None:
        """Initialize the exception.
        
        Args:
            file_path: Path where file creation failed
            details: Optional additional details about the error
        """
        message = f"Failed to create file: {file_path}"
        super().__init__(message, details)
        self.file_path = file_path
