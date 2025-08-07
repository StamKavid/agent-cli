"""
Modern template manager for AI agent projects.
"""

from pathlib import Path
from typing import Dict, Any
import logging

from ..templates import ProjectTemplateManager, FileTemplateManager

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages project and file templates for modern AI agent projects."""
    
    def __init__(self):
        self.project_templates = ProjectTemplateManager()
        self.file_templates = FileTemplateManager()
    
    def list(self) -> Dict[str, str]:
        """List available templates with descriptions."""
        return self.project_templates.get_available_templates()
    
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context."""
        return self.file_templates.render_template(template_name, context)
    
    def create_project_structure(self, base_path: Path, template_id: str, project_name: str) -> None:
        """Create the complete project structure."""
        # Create directories
        self.project_templates.create_directory_structure(base_path, template_id, project_name)
        
        # Get template structure
        template_structure = self.project_templates.get_project_structure(template_id)
        
        # Convert project name to package name for context
        package_name = project_name.replace('-', '_').lower()
        context = {
            "project_name": project_name,
            "package_name": package_name,
            "template": template_id,
            "description": f"AI Agent Project built with Agent CLI"
        }
        
        # Create files
        self._create_files(base_path, template_structure["files"], context)
    
    def _create_files(self, base_path: Path, file_list: list, context: Dict[str, Any]) -> None:
        """Create all files from the template."""
        for file_path_template in file_list:
            try:
                # Replace placeholders in file path
                file_path_str = file_path_template.replace("{package_name}", context["package_name"])
                file_path_str = file_path_str.replace("{project_name}", context["project_name"])
                
                file_path = base_path / file_path_str
                
                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Get template name from file extension/type
                template_name = self._get_template_name_for_file(file_path)
                
                if template_name:
                    try:
                        content = self.file_templates.render_template(template_name, context)
                        file_path.write_text(content, encoding='utf-8')
                        logger.debug(f"Created file: {file_path}")
                    except ValueError:
                        # Template not found, create empty file or skip
                        if template_name in ["__init__.py", "empty"]:
                            file_path.write_text("", encoding='utf-8')
                        else:
                            # Create placeholder content
                            file_path.write_text(f"# {file_path.name} - TODO: Implement\n", encoding='utf-8')
                        logger.debug(f"Created placeholder file: {file_path}")
                else:
                    # Create empty file for unknown types
                    file_path.write_text("", encoding='utf-8')
                    logger.debug(f"Created empty file: {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to create file {file_path_template}: {e}")
    
    def _get_template_name_for_file(self, file_path: Path) -> str:
        """Determine the template name based on file path."""
        file_name = file_path.name
        file_stem = file_path.stem
        path_str = str(file_path)
        
        # Map specific files to our comprehensive templates
        template_mapping = {
            # Root configuration files
            "pyproject.toml": "pyproject.toml",
            "README.md": "README.md", 
            ".env.example": ".env.example",
            ".env": ".env",
            ".gitignore": ".gitignore",
            "Makefile": "Makefile",
            "Dockerfile": "Dockerfile",
            "docker-compose.yml": "docker-compose.yml",
            
            # GitHub CI/CD
            "ci.yml": ".github/workflows/ci.yml",
            
            # Configuration files
            "agent_config.yaml": "configs/agent_config.yaml",
            "llm_config.yaml": "configs/llm_config.yaml",
            "deployment_config.yaml": "configs/deployment_config.yaml",
            
            # Kubernetes
            "deployment.yaml": "k8s/deployment.yaml",
            "service.yaml": "k8s/service.yaml",
            
            # Core package files
            "__init__.py": "src/{{package_name}}/__init__.py",
            "config.py": "src/{{package_name}}/config.py",
            "main.py": "src/{{package_name}}/main.py",
            
            # Agent files
            "base_agent.py": "src/{{package_name}}/agent/base_agent.py",
            "langgraph_agent.py": "src/{{package_name}}/agent/langgraph_agent.py",
            "state_manager.py": "src/{{package_name}}/agent/state_manager.py",
            
            # Memory files
            "langmem_integration.py": "src/{{package_name}}/memory/langmem_integration.py",
            "vector_store.py": "src/{{package_name}}/memory/vector_store.py",
            "conversation_memory.py": "src/{{package_name}}/memory/conversation_memory.py",
            
            # Tool files
            "base_tool.py": "src/{{package_name}}/tools/base_tool.py",
            "web_search.py": "src/{{package_name}}/tools/web_search.py",
            "file_operations.py": "src/{{package_name}}/tools/file_operations.py",
            "custom_tools.py": "src/{{package_name}}/tools/custom_tools.py",
            
            # Workflow files
            "workflow_builder.py": "src/{{package_name}}/workflows/workflow_builder.py",
            "common_workflows.py": "src/{{package_name}}/workflows/common_workflows.py",
            
            # Prompt files
            "prompt_manager.py": "src/{{package_name}}/prompts/prompt_manager.py",
            "prompt_library.py": "src/{{package_name}}/prompts/prompt_library.py",
            "system_prompts.py": "src/{{package_name}}/prompts/templates/system_prompts.py",
            "user_prompts.py": "src/{{package_name}}/prompts/templates/user_prompts.py",
            "tool_prompts.py": "src/{{package_name}}/prompts/templates/tool_prompts.py",
            "workflow_prompts.py": "src/{{package_name}}/prompts/templates/workflow_prompts.py",
            "v1_prompts.py": "src/{{package_name}}/prompts/versions/v1_prompts.py",
            "v2_prompts.py": "src/{{package_name}}/prompts/versions/v2_prompts.py",
            
            # LLM files
            "client.py": "src/{{package_name}}/llm/client.py",
            "prompt_executor.py": "src/{{package_name}}/llm/prompt_executor.py",
            
            # API files
            "app.py": "src/{{package_name}}/api/app.py",
            "routes.py": "src/{{package_name}}/api/routes.py",
            "schemas.py": "src/{{package_name}}/api/schemas.py",
            
            # Observability files
            "opik_integration.py": "src/{{package_name}}/observability/opik_integration.py",
            "metrics.py": "src/{{package_name}}/observability/metrics.py",
            
            # Utility files
            "logging.py": "src/{{package_name}}/utils/logging.py",
            "helpers.py": "src/{{package_name}}/utils/helpers.py",
            
            # Tool scripts
            "run_agent.py": "tools/run_agent.py",
            "evaluate_agent.py": "tools/evaluate_agent.py",
            "populate_memory.py": "tools/populate_memory.py",
            "manage_prompts.py": "tools/manage_prompts.py",
            "deploy.py": "tools/deploy.py",
            
            # Test files
            "test_agent.py": "tests/test_agent.py",
            "test_memory.py": "tests/test_memory.py",
            "test_tools.py": "tests/test_tools.py",
            "test_workflows.py": "tests/test_workflows.py",
            
            # Documentation
            "getting_started.md": "docs/getting_started.md",
            "configuration.md": "docs/configuration.md",
            "deployment.md": "docs/deployment.md",
        }
        
        # Check for exact filename match
        if file_name in template_mapping:
            return template_mapping[file_name]
        
        # Check for directory-specific patterns
        if file_name == "__init__.py":
            if "tests/" in path_str:
                return "tests/__init__.py"
            elif "/agent/" in path_str:
                return "src/{{package_name}}/agent/__init__.py"
            elif "/memory/" in path_str:
                return "src/{{package_name}}/memory/__init__.py"
            elif "/tools/" in path_str:
                return "src/{{package_name}}/tools/__init__.py"
            elif "/workflows/" in path_str:
                return "src/{{package_name}}/workflows/__init__.py"
            elif "/prompts/" in path_str:
                return "src/{{package_name}}/prompts/__init__.py"
            elif "/llm/" in path_str:
                return "src/{{package_name}}/llm/__init__.py"
            elif "/api/" in path_str:
                return "src/{{package_name}}/api/__init__.py"
            elif "/observability/" in path_str:
                return "src/{{package_name}}/observability/__init__.py"
            elif "/utils/" in path_str:
                return "src/{{package_name}}/utils/__init__.py"
            else:
                return "src/{{package_name}}/__init__.py"
        
        # Check for notebook files
        if file_name.endswith(".ipynb"):
            return "notebooks/template.ipynb"
        
        # Return None for files we don't have templates for
        # They will be created as placeholder files
        return None
