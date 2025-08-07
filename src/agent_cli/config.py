"""
Configuration settings for the Agent CLI.
"""

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ProjectStructure:
    """Configuration for project directory structure."""
    
    directories: List[str]
    required_files: List[str]
    optional_files: List[str]


@dataclass
class CLIConfig:
    """Configuration for the CLI application."""
    
    # ULTRA-SIMPLIFIED directory structure - Essential directories only
    PROJECT_DIRECTORIES = [
        # ONLY 3 essential directories instead of 50+
        "src/{project_name}",   # Source code
        "tests",               # Tests  
        "data",               # Optional data directory
    ]
    
    # MINIMAL packages only - Just what's needed to run
    REQUIRED_PACKAGES = [
        # ONLY core AI packages - users add more as needed
        "langchain>=0.2.0",
        "openai>=1.0.0", 
        "anthropic>=0.8.0",
        "python-dotenv>=1.0.0",
    ]
    
    # Development packages - Minimal set
    DEV_PACKAGES = [
        "pytest>=8.0.0",
        "black>=24.0.0",
    ]
    
    # Python version
    PYTHON_VERSION = "3.10"
    
    # Simplified agent templates - Focus on core functionality
    AGENT_TEMPLATES = {
        "basic": {
            "description": "💬 Basic Agent (Simple chatbot)",
            "features": ["chat", "memory"],
            "tools": ["basic"],
            "llm_provider": "openai",
            "model": "gpt-4o-mini"
        },
        "conversational": {
            "description": "� Conversational Agent (Enhanced chatbot)",
            "features": ["chat", "memory", "context_awareness"],
            "tools": ["web_search", "file_operations"],
            "llm_provider": "openai",
            "model": "gpt-4o"
        },
        "research": {
            "description": "🔍 Research Assistant (Information gathering)",
            "features": ["web_search", "analysis"],
            "tools": ["web_search", "file_operations"],
            "llm_provider": "openai",
            "model": "gpt-4o"
        }
    }
    
    # Default project structure
    @classmethod
    def get_project_structure(cls) -> ProjectStructure:
        """Get the default project structure configuration."""
        return ProjectStructure(
            directories=cls.PROJECT_DIRECTORIES,
            required_files=[
                "pyproject.toml",
                "README.md", 
                ".env",
                ".gitignore",
                "requirements.txt",
            ],
            optional_files=[
                "Dockerfile",
                "Makefile",
                "requirements-dev.txt",
            ]
        )
