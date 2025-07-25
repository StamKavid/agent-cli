"""
Template provider implementations.

This module provides concrete implementations of the TemplateProvider interface,
following the Single Responsibility Principle (SRP) and using modular template categories.
"""

import logging
from pathlib import Path
from typing import Dict, List

from ..interfaces import TemplateProvider
from ..exceptions import TemplateError
from .template_categories import (
    FileTemplates,
    NotebookTemplates,
    ToolTemplates,
    TestTemplates,
    SourceTemplates,
    QuickstartTemplates
)

logger = logging.getLogger(__name__)


class StaticTemplateProvider(TemplateProvider):
    """Template provider that stores templates as static strings using modular categories."""
    
    def __init__(self) -> None:
        """Initialize the template provider."""
        self._templates: Dict[str, str] = {}
        self._load_templates()
    
    def get_template(self, template_name: str, context: dict = None) -> str:
        """Get a template by name, optionally rendering with context."""
        if template_name not in self._templates:
            raise TemplateError(
                template_name, 
                f"Template '{template_name}' not found"
            )
        
        template_str = self._templates[template_name]
        
        # Simple template rendering with context
        if context:
            try:
                # For certain templates, we need to escape curly braces that aren't meant for formatting
                if template_name in ['makefile', 'dockerfile', 'gitignore']:
                    # Escape curly braces by doubling them
                    template_str = template_str.replace('{', '{{').replace('}', '}}')
                
                # Use safe formatting that handles curly braces in templates
                return template_str.format_map(context)
            except KeyError as e:
                raise TemplateError(
                    template_name,
                    f"Missing required context variable: {e}"
                )
            except ValueError as e:
                # Handle cases where templates have curly braces that aren't meant for formatting
                if "Single '{'" in str(e) or "Single '}'" in str(e):
                    # Return template as-is if it has unmatched curly braces
                    return template_str
                else:
                    raise TemplateError(
                        template_name,
                        f"Template formatting error: {e}"
                    )
        
        return template_str
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self._templates.keys())
    
    def _load_templates(self) -> None:
        """Load all templates into memory using modular categories."""
        logger.debug("Loading templates from modular categories...")
        
        # Load file templates
        self._load_file_templates()
        
        # Load notebook templates
        self._load_notebook_templates()
        
        # Load tool templates
        self._load_tool_templates()
        
        # Load test templates
        self._load_test_templates()
        
        # Load source templates
        self._load_source_templates()
        
        # Load quickstart templates
        self._load_quickstart_templates()
        
        # Load configuration templates
        self._load_config_templates()
        
        # Load requirements templates
        self._load_requirements_templates()
        
        # Load CLI templates
        self._load_cli_templates()
        
        # Load additional source templates
        self._load_source_templates()
        
        logger.debug(f"Loaded {len(self._templates)} templates")
    
    def _load_file_templates(self) -> None:
        """Load basic file templates."""
        self._templates.update({
            "gitignore": FileTemplates.get_gitignore(),
            "env": FileTemplates.get_env(),
            "dockerfile": FileTemplates.get_dockerfile(),
            "makefile": FileTemplates.get_makefile(),
        })
    
    def _load_notebook_templates(self) -> None:
        """Load notebook templates."""
        notebook_templates = {
            "01_prompt_engineering_playground.ipynb": NotebookTemplates.get_prompt_engineering_playground(),
            "02_short_term_memory.ipynb": NotebookTemplates.get_short_term_memory(),
            "03_long_term_memory.ipynb": NotebookTemplates.get_long_term_memory(),
            "04_tool_calling_playground.ipynb": NotebookTemplates.get_tool_calling_playground(),
        }
        
        for name, content in notebook_templates.items():
            self._templates[f"notebook_{name}"] = content
    
    def _load_tool_templates(self) -> None:
        """Load tool script templates."""
        tool_templates = {
            "run_agent.py": ToolTemplates.get_run_agent(),
            "populate_long_term_memory.py": ToolTemplates.get_populate_long_term_memory(),
            "delete_long_term_memory.py": ToolTemplates.get_delete_long_term_memory(),
            "evaluate_agent.py": ToolTemplates.get_evaluate_agent(),
        }
        
        for name, content in tool_templates.items():
            self._templates[f"tool_{name}"] = content
    
    def _load_test_templates(self) -> None:
        """Load test file templates."""
        test_templates = {
            "test_agent.py": TestTemplates.get_test_agent(),
            "test_memory.py": TestTemplates.get_test_memory(),
            "__init__.py": TestTemplates.get_init(),
        }
        
        for name, content in test_templates.items():
            self._templates[f"test_{name}"] = content
    
    def _load_quickstart_templates(self) -> None:
        """Load quickstart templates."""
        self._templates.update({
            "quickstart.py": QuickstartTemplates.get_quickstart_py(),
            "requirements-quickstart.txt": QuickstartTemplates.get_requirements_quickstart(),
            "quickstart_README.md": QuickstartTemplates.get_quickstart_readme(),
        })
    
    def _load_config_templates(self) -> None:
        """Load configuration templates."""
        self._templates.update({
            "config_base": QuickstartTemplates.get_config_base(),
            "config_development": QuickstartTemplates.get_config_development(),
            "config_staging": QuickstartTemplates.get_config_staging(),
            "config_production": QuickstartTemplates.get_config_production(),
            "config_agent_conversational": QuickstartTemplates.get_config_agent_conversational(),
            "config_agent_research": QuickstartTemplates.get_config_agent_research(),
            "config_agent_automation": QuickstartTemplates.get_config_agent_automation(),
            "config_agent_analysis": QuickstartTemplates.get_config_agent_analysis(),
            "config_llm_openai": QuickstartTemplates.get_config_llm_openai(),
            "config_llm_anthropic": QuickstartTemplates.get_config_llm_anthropic(),
            "config_llm_google": QuickstartTemplates.get_config_llm_google(),
            "config_llm_azure": QuickstartTemplates.get_config_llm_azure(),
            "config_llm_local": QuickstartTemplates.get_config_llm_local(),
            "config_prompt_templates": QuickstartTemplates.get_config_prompt_templates(),
            "config_logging": QuickstartTemplates.get_config_logging(),
        })
    
    def _load_requirements_templates(self) -> None:
        """Load requirements templates."""
        self._templates.update({
            "requirements": QuickstartTemplates.get_requirements(),
            "requirements_dev": QuickstartTemplates.get_requirements_dev(),
            "requirements_prod": QuickstartTemplates.get_requirements_prod(),
        })
    
    def _load_cli_templates(self) -> None:
        """Load CLI templates."""
        self._templates.update({
            "cli_main": QuickstartTemplates.get_cli_main(),
            "cli_command_run": QuickstartTemplates.get_cli_command_run(),
            "cli_command_chat": QuickstartTemplates.get_cli_command_chat(),
        })
    
    def _load_source_templates(self) -> None:
        """Load source code templates."""
        self._templates.update({
            "__init__.py": SourceTemplates.get_init(),
            "base_agent": QuickstartTemplates.get_base_agent(),
            "conversational_agent": QuickstartTemplates.get_conversational_agent(),
            "research_agent": QuickstartTemplates.get_research_agent(),
            "automation_agent": QuickstartTemplates.get_automation_agent(),
            "analysis_agent": QuickstartTemplates.get_analysis_agent(),
            "agent_service": QuickstartTemplates.get_agent_service(),
        })
    
    def _load_templates(self) -> None:
        """Load all templates into memory using modular categories."""
        logger.debug("Loading templates from modular categories...")
        
        # Load file templates
        self._load_file_templates()
        
        # Load notebook templates
        self._load_notebook_templates()
        
        # Load tool templates
        self._load_tool_templates()
        
        # Load test templates
        self._load_test_templates()
        
        # Load source templates
        self._load_source_templates()
        
        # Load quickstart templates
        self._load_quickstart_templates()
        
        # Load configuration templates
        self._load_config_templates()
        
        # Load requirements templates
        self._load_requirements_templates()
        
        # Load CLI templates
        self._load_cli_templates()
        
        # Load additional source templates
        self._load_source_templates()
        
        logger.debug(f"Loaded {len(self._templates)} templates")
    
    def _get_dynamic_templates(self, project_name: str) -> Dict[str, str]:
        """Get templates that require project name substitution."""
        return {
            "pyproject_toml": FileTemplates.get_pyproject_toml(project_name),
            "readme": FileTemplates.get_readme(project_name),
        }
    
    def get_template_with_context(self, template_name: str, context: Dict[str, str]) -> str:
        """Get a template with context, handling dynamic templates."""
        # Handle dynamic templates that need project name
        if template_name in ["pyproject_toml", "readme"]:
            project_name = context.get("project_name", "unknown")
            dynamic_templates = self._get_dynamic_templates(project_name)
            return dynamic_templates[template_name]
        
        # Handle regular templates
        return self.get_template(template_name, context)
