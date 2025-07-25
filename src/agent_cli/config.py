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
    
    # Comprehensive directory structure template - Optimal Agent Structure
    PROJECT_DIRECTORIES = [
        # Top-level directories
        "quickstart",
        "notebooks", 
        "examples",
        "config",
        "config/environments",
        "config/agents", 
        "config/llm_providers",
        "cli",
        "cli/commands",
        "cli/interactive",
        "cli/utils",
        "tools",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "infrastructure",
        "infrastructure/docker",
        "infrastructure/kubernetes", 
        "infrastructure/terraform",
        "infrastructure/observability",
        "data",
        "data/cache",
        "data/prompts",
        "data/outputs",
        "data/logs",
        "data/models",
        "docs",
        
        # Source code - Clean Architecture
        "src/{project_name}",
        "src/{project_name}/application",
        "src/{project_name}/application/services",
        "src/{project_name}/application/interfaces",
        "src/{project_name}/application/dto",
        
        "src/{project_name}/core",
        "src/{project_name}/core/agents",
        "src/{project_name}/core/agents/base",
        "src/{project_name}/core/agents/conversational",
        "src/{project_name}/core/agents/research",
        "src/{project_name}/core/agents/automation",
        "src/{project_name}/core/agents/analysis",
        "src/{project_name}/core/agents/creative",
        
        "src/{project_name}/core/memory",
        "src/{project_name}/core/memory/short_term",
        "src/{project_name}/core/memory/long_term",
        "src/{project_name}/core/memory/manager",
        
        "src/{project_name}/core/planning",
        "src/{project_name}/core/planning/planner",
        "src/{project_name}/core/planning/task_decomposer",
        "src/{project_name}/core/planning/reasoning_engine",
        
        "src/{project_name}/core/coordination",
        "src/{project_name}/core/coordination/coordinator",
        "src/{project_name}/core/coordination/agent_registry",
        "src/{project_name}/core/coordination/workflow_orchestrator",
        
        "src/{project_name}/core/communication",
        "src/{project_name}/core/communication/message_bus",
        "src/{project_name}/core/communication/agent_communicator",
        "src/{project_name}/core/communication/protocol_handler",
        
        "src/{project_name}/tools",
        "src/{project_name}/tools/base",
        "src/{project_name}/tools/web_search",
        "src/{project_name}/tools/file_operations",
        "src/{project_name}/tools/database",
        "src/{project_name}/tools/api",
        "src/{project_name}/tools/custom",
        
        "src/{project_name}/mcp",
        "src/{project_name}/mcp/protocol",
        "src/{project_name}/mcp/server",
        "src/{project_name}/mcp/client",
        "src/{project_name}/mcp/tool_interface",
        "src/{project_name}/mcp/resource_interface",
        
        "src/{project_name}/prompts",
        "src/{project_name}/prompts/templates",
        "src/{project_name}/prompts/builders",
        "src/{project_name}/prompts/few_shot",
        "src/{project_name}/prompts/chainery",
        "src/{project_name}/prompts/agent_specific",
        
        "src/{project_name}/infrastructure",
        "src/{project_name}/infrastructure/llm_clients",
        "src/{project_name}/infrastructure/llm_clients/base",
        "src/{project_name}/infrastructure/llm_clients/factory",
        "src/{project_name}/infrastructure/llm_clients/openai",
        "src/{project_name}/infrastructure/llm_clients/anthropic",
        "src/{project_name}/infrastructure/llm_clients/google",
        "src/{project_name}/infrastructure/llm_clients/azure",
        "src/{project_name}/infrastructure/llm_clients/local",
        
        "src/{project_name}/infrastructure/vector_database",
        "src/{project_name}/infrastructure/vector_database/base",
        "src/{project_name}/infrastructure/vector_database/chroma",
        "src/{project_name}/infrastructure/vector_database/pinecone",
        "src/{project_name}/infrastructure/vector_database/weaviate",
        
        "src/{project_name}/infrastructure/monitoring",
        "src/{project_name}/infrastructure/monitoring/logger",
        "src/{project_name}/infrastructure/monitoring/tracer",
        "src/{project_name}/infrastructure/monitoring/metrics",
        "src/{project_name}/infrastructure/monitoring/health_checker",
        
        "src/{project_name}/infrastructure/api",
        "src/{project_name}/infrastructure/api/app",
        "src/{project_name}/infrastructure/api/routes",
        "src/{project_name}/infrastructure/api/middleware",
        
        "src/{project_name}/infrastructure/storage",
        "src/{project_name}/infrastructure/storage/cache",
        "src/{project_name}/infrastructure/storage/database",
        "src/{project_name}/infrastructure/storage/file_system",
        
        "src/{project_name}/utils",
        "src/{project_name}/utils/rate_limiter",
        "src/{project_name}/utils/token_counter",
        "src/{project_name}/utils/cache",
        "src/{project_name}/utils/validators",
        "src/{project_name}/utils/helpers",
        
        # Development and CI/CD
        ".github/workflows",
        ".vscode",
        "scripts",
    ]
    
    # Required Python packages - Enhanced for full AI stack
    REQUIRED_PACKAGES = [
        # Core AI/ML
        "langchain>=0.2.0",
        "langgraph>=0.2.0",
        "langchain-openai>=0.1.0",
        "langchain-anthropic>=0.1.0", 
        "langchain-google-genai>=0.1.0",
        "langchain-community>=0.0.10",
        
        # LLM Providers
        "openai>=1.0.0",
        "anthropic>=0.8.0",
        "google-generativeai>=0.3.0",
        
        # Vector Databases
        "chromadb>=0.4.0",
        "pinecone-client>=3.0.0",
        "weaviate-client>=4.0.0",
        
        # Web Framework
        "fastapi>=0.110.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.7.0",
        
        # Data Processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        
        # Monitoring & Observability
        "opentelemetry-api>=1.24.0",
        "opentelemetry-sdk>=1.24.0",
        "prometheus-client>=0.20.0",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "jinja2>=3.1.3",
        "rich>=13.7.0",
        "click>=8.1.0",
        "typer>=0.9.0",
        
        # Development
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ]
    
    # Development packages
    DEV_PACKAGES = [
        "pytest>=8.3.0",
        "pytest-cov>=5.0.0",
        "pytest-mock>=3.14.0",
        "pytest-asyncio>=0.21.0",
        "black>=24.8.0",
        "isort>=5.13.0", 
        "mypy>=1.11.0",
        "flake8>=7.1.0",
        "pre-commit>=3.8.0",
        "bandit>=1.7.0",
        "safety>=2.3.0",
    ]
    
    # Production packages
    PRODUCTION_PACKAGES = [
        "docker>=7.0.0",
        "kubernetes>=29.0.0",
        "gunicorn>=21.0.0",
        "redis>=5.0.0",
        "celery>=5.3.0",
        "flower>=2.0.0",
    ]
    
    # Python version
    PYTHON_VERSION = "3.13"
    
    # Agent templates
    AGENT_TEMPLATES = {
        "conversational": {
            "description": "💬 Conversational Agent (Chatbot, Customer Service)",
            "features": ["chat", "memory", "personality", "context_awareness"],
            "tools": ["web_search", "file_operations", "database"],
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet-20240229"
        },
        "research": {
            "description": "🔍 Research Assistant (Information Gathering, Analysis)",
            "features": ["web_search", "document_analysis", "summarization", "citation"],
            "tools": ["web_search", "file_operations", "database", "pdf_reader"],
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet-20240229"
        },
        "automation": {
            "description": "⚙️ Task Automation (Workflow Orchestration)",
            "features": ["workflow_engine", "task_decomposition", "error_handling", "retry_logic"],
            "tools": ["file_operations", "api", "database", "scheduler"],
            "llm_provider": "openai",
            "model": "gpt-4o"
        },
        "analysis": {
            "description": "📊 Data Analysis Agent (Reporting, Insights)",
            "features": ["data_processing", "visualization", "statistical_analysis", "reporting"],
            "tools": ["file_operations", "database", "api", "chart_generator"],
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet-20240229"
        },
        "custom": {
            "description": "🎯 Custom Agent (Build from scratch)",
            "features": ["modular", "extensible", "configurable"],
            "tools": ["base_tools"],
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet-20240229"
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
                "Dockerfile",
                "Makefile",
                "requirements.txt",
                "requirements-dev.txt",
                "requirements-prod.txt",
            ],
            optional_files=[
                "ISSUE_TEMPLATE.md",
                "python-version",
                ".github/workflows/ci.yml",
                ".github/workflows/cd.yml",
                ".vscode/settings.json",
                ".vscode/extensions.json",
                "scripts/setup.sh",
                "scripts/deploy.sh",
            ]
        )
