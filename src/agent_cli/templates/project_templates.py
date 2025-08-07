"""
Clean project structure templates for modern AI agent projects.
"""

from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ProjectTemplateManager:
    """Manages clean project structure templates for different agent types."""
    
    def __init__(self):
        self.template_definitions = self._get_template_definitions()
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get list of available project templates with descriptions."""
        return {
            template_id: template_data["description"]
            for template_id, template_data in self.template_definitions.items()
        }
    
    def get_project_structure(self, template_id: str) -> Dict[str, Any]:
        """Get the full project structure for a template."""
        if template_id not in self.template_definitions:
            raise ValueError(f"Template '{template_id}' not found")
        
        return self.template_definitions[template_id]
    
    def create_directory_structure(self, base_path: Path, template_id: str, project_name: str) -> List[Path]:
        """Create the directory structure for a project template.
        
        Args:
            base_path: Base path where project should be created
            template_id: Template to use
            project_name: Name of the project
            
        Returns:
            List of created directories
        """
        template = self.get_project_structure(template_id)
        directories = template["directories"]
        
        created_dirs = []
        
        # Convert project name to Python package name
        package_name = project_name.replace('-', '_').lower()
        
        for dir_path in directories:
            # Replace placeholders
            dir_path = dir_path.replace("{project_name}", project_name)
            dir_path = dir_path.replace("{package_name}", package_name)
            
            full_path = base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(full_path)
            logger.debug(f"Created directory: {full_path}")
        
        return created_dirs
    
    def _get_template_definitions(self) -> Dict[str, Any]:
        """Define the comprehensive structure templates for modern AI agent projects."""
        
        # Comprehensive modern agent structure
        modern_structure = [
            # Root directories
            ".github/workflows",
            "configs",
            "data/knowledge_base",
            "data/evaluation", 
            "k8s",
            "notebooks",
            "src/{package_name}",
            "src/{package_name}/agent",
            "src/{package_name}/memory",
            "src/{package_name}/tools",
            "src/{package_name}/workflows",
            "src/{package_name}/prompts",
            "src/{package_name}/prompts/templates",
            "src/{package_name}/prompts/versions",
            "src/{package_name}/llm",
            "src/{package_name}/api",
            "src/{package_name}/observability",
            "src/{package_name}/utils",
            "tools",
            "tests",
            "docs",
        ]
        
        modern_files = [
            # Root files
            ".gitignore",
            ".env.example", 
            ".env",
            "README.md",
            "Makefile",
            "Dockerfile",
            "docker-compose.yml",
            "pyproject.toml",
            
            # GitHub CI/CD
            ".github/workflows/ci.yml",
            
            # Configuration files
            "configs/agent_config.yaml",
            "configs/llm_config.yaml", 
            "configs/deployment_config.yaml",
            
            # Kubernetes deployment
            "k8s/deployment.yaml",
            "k8s/service.yaml",
            
            # Jupyter notebooks
            "notebooks/01_prompt_engineering.ipynb",
            "notebooks/02_prompt_management.ipynb", 
            "notebooks/03_memory_system.ipynb",
            "notebooks/04_tool_integration.ipynb",
            "notebooks/05_langgraph_workflows.ipynb",
            "notebooks/06_evaluation.ipynb",
            
            # Core package
            "src/{package_name}/__init__.py",
            "src/{package_name}/config.py",
            "src/{package_name}/main.py",
            
            # Agent modules
            "src/{package_name}/agent/__init__.py",
            "src/{package_name}/agent/base_agent.py",
            "src/{package_name}/agent/langgraph_agent.py",
            "src/{package_name}/agent/state_manager.py",
            
            # Memory system
            "src/{package_name}/memory/__init__.py", 
            "src/{package_name}/memory/langmem_integration.py",
            "src/{package_name}/memory/vector_store.py",
            "src/{package_name}/memory/conversation_memory.py",
            
            # Tools
            "src/{package_name}/tools/__init__.py",
            "src/{package_name}/tools/base_tool.py", 
            "src/{package_name}/tools/web_search.py",
            "src/{package_name}/tools/file_operations.py",
            "src/{package_name}/tools/custom_tools.py",
            
            # Workflows
            "src/{package_name}/workflows/__init__.py",
            "src/{package_name}/workflows/workflow_builder.py",
            "src/{package_name}/workflows/common_workflows.py",
            
            # Prompt management
            "src/{package_name}/prompts/__init__.py",
            "src/{package_name}/prompts/prompt_manager.py",
            "src/{package_name}/prompts/prompt_library.py",
            "src/{package_name}/prompts/templates/system_prompts.py",
            "src/{package_name}/prompts/templates/user_prompts.py", 
            "src/{package_name}/prompts/templates/tool_prompts.py",
            "src/{package_name}/prompts/templates/workflow_prompts.py",
            "src/{package_name}/prompts/versions/v1_prompts.py",
            "src/{package_name}/prompts/versions/v2_prompts.py",
            
            # LLM integration
            "src/{package_name}/llm/__init__.py",
            "src/{package_name}/llm/client.py",
            "src/{package_name}/llm/prompt_executor.py",
            
            # API layer
            "src/{package_name}/api/__init__.py",
            "src/{package_name}/api/app.py",
            "src/{package_name}/api/routes.py",
            "src/{package_name}/api/schemas.py",
            
            # Observability
            "src/{package_name}/observability/__init__.py",
            "src/{package_name}/observability/opik_integration.py",
            "src/{package_name}/observability/metrics.py",
            
            # Utilities
            "src/{package_name}/utils/__init__.py",
            "src/{package_name}/utils/logging.py",
            "src/{package_name}/utils/helpers.py",
            
            # Operational tools
            "tools/run_agent.py",
            "tools/evaluate_agent.py", 
            "tools/populate_memory.py",
            "tools/manage_prompts.py",
            "tools/deploy.py",
            
            # Tests
            "tests/test_agent.py",
            "tests/test_memory.py",
            "tests/test_tools.py", 
            "tests/test_workflows.py",
            
            # Documentation
            "docs/getting_started.md",
            "docs/configuration.md",
            "docs/deployment.md",
        ]
        
        return {
            "basic": {
                "description": "🤖 Modern AI Agent - Comprehensive LangGraph/LangChain agent with full production structure",
                "directories": modern_structure,
                "features": ["langgraph", "langchain", "fastapi", "memory", "tools", "workflows", "observability", "kubernetes", "ci-cd"],
                "files": modern_files,
            },
            "conversational": {
                "description": "💬 Conversational Agent - Full modern structure optimized for chat and conversation",  
                "directories": modern_structure,
                "features": ["langgraph", "langchain", "fastapi", "memory", "tools", "workflows", "observability", "kubernetes", "ci-cd"],
                "files": modern_files,
            },
            "research": {
                "description": "🔍 Research Agent - Comprehensive structure with advanced research capabilities",
                "directories": modern_structure, 
                "features": ["langgraph", "langchain", "fastapi", "memory", "tools", "workflows", "research", "observability", "kubernetes", "ci-cd"],
                "files": modern_files,
            },
            "automation": {
                "description": "⚙️ Automation Agent - Full production structure for automation and workflow tasks",
                "directories": modern_structure,
                "features": ["langgraph", "langchain", "fastapi", "memory", "tools", "workflows", "automation", "observability", "kubernetes", "ci-cd"],
                "files": modern_files,
            }
        }


def get_project_template_manager():
    """Get the project template manager instance."""
    return ProjectTemplateManager()
