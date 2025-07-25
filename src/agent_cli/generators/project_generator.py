"""
Concrete implementations of project creation components.

This module provides implementations that follow SOLID principles
and proper separation of concerns with improved template handling.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List

from ..interfaces import (
    DirectoryCreator,
    FileGenerator, 
    ProjectValidator,
    ProjectCreator,
    TemplateProvider
)
from ..exceptions import (
    DirectoryCreationError,
    FileCreationError,
    ProjectExistsError,
    TemplateError,
    AgentCLIError
)
from ..config import CLIConfig

logger = logging.getLogger(__name__)


class DefaultProjectValidator(ProjectValidator):
    """Default implementation of project validation with enhanced validation."""
    
    def validate_project_name(self, project_name: str) -> bool:
        """Validate project name according to Python package naming conventions."""
        # Check for valid Python package name
        pattern = r'^[a-z][a-z0-9_]*[a-z0-9]$|^[a-z]$'
        
        if not re.match(pattern, project_name):
            logger.error(f"Invalid project name: {project_name}")
            logger.error("Project name must be a valid Python package name")
            logger.error("Use lowercase letters, numbers, and underscores only")
            return False
        
        # Check length
        if len(project_name) > 50:
            logger.error("Project name too long (max 50 characters)")
            return False
        
        # Check for reserved words
        reserved_words = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
            'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass',
            'raise', 'return', 'try', 'while', 'with', 'yield'
        }
        if project_name in reserved_words:
            logger.error(f"Project name '{project_name}' is a Python reserved word")
            return False
            
        return True
    
    def validate_output_directory(self, output_dir: str) -> bool:
        """Validate output directory exists and is writable."""
        try:
            output_path = Path(output_dir)
            
            # Check if directory exists
            if not output_path.exists():
                logger.error(f"Output directory does not exist: {output_dir}")
                return False
            
            # Check if it's a directory
            if not output_path.is_dir():
                logger.error(f"Output path is not a directory: {output_dir}")
                return False
                
            # Check if writable (try to create a temporary file)
            test_file = output_path / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError):
                logger.error(f"No write permission in directory: {output_dir}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating output directory: {e}")
            return False


class DefaultDirectoryCreator(DirectoryCreator):
    """Default implementation of directory creation with enhanced error handling."""
    
    def create_directory_structure(
        self,
        base_path: Path,
        directories: List[str],
        project_name: str
    ) -> None:
        """Create directory structure with proper error handling and validation."""
        logger.info(f"Creating directory structure in {base_path}")
        
        created_dirs = []
        
        try:
            for directory_pattern in directories:
                try:
                    # Substitute project name in directory pattern
                    directory_path = directory_pattern.format(project_name=project_name)
                    full_path = base_path / directory_path
                    
                    # Create directory
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(full_path)
                    logger.debug(f"Created directory: {full_path}")
                    
                    # Create __init__.py for Python packages
                    if self._is_python_package_directory(directory_path):
                        init_file = full_path / "__init__.py"
                        if not init_file.exists():
                            init_file.write_text("")
                            logger.debug(f"Created __init__.py in {full_path}")
                            
                except OSError as e:
                    raise DirectoryCreationError(
                        directory_path,
                        f"Failed to create directory: {e}"
                    )
            
            logger.info("Directory structure created successfully")
            
        except Exception as e:
            # Clean up created directories on failure
            logger.error(f"Error creating directory structure: {e}")
            for created_dir in reversed(created_dirs):
                try:
                    if created_dir.exists():
                        created_dir.rmdir()
                except OSError:
                    pass  # Ignore cleanup errors
            raise
    
    def _is_python_package_directory(self, directory_path: str) -> bool:
        """Check if directory should be a Python package."""
        python_package_indicators = [
            "src/", 
            "/application/",
            "/core/",
            "/infrastructure/"
        ]
        return any(indicator in directory_path for indicator in python_package_indicators)


class DefaultFileGenerator(FileGenerator):
    """Default implementation of file generation with enhanced template handling."""
    
    def __init__(self, template_provider: TemplateProvider) -> None:
        """Initialize with a template provider.
        
        Args:
            template_provider: Provider for template content
        """
        self.template_provider = template_provider
    
    def generate_file(
        self,
        file_path: Path,
        template_name: str,
        context: Dict[str, Any]
    ) -> None:
        """Generate a file from template with enhanced error handling and validation."""
        try:
            # Validate template name
            if not template_name:
                raise TemplateError("", "Template name cannot be empty")
            
            # Get template content with context
            if hasattr(self.template_provider, 'get_template_with_context'):
                template_content = self.template_provider.get_template_with_context(template_name, context)
            else:
                template_content = self.template_provider.get_template(template_name, context)
            
            # Validate template content
            if not template_content:
                raise TemplateError(template_name, "Template content is empty")
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file with proper encoding
            file_path.write_text(template_content, encoding='utf-8')
            logger.debug(f"Generated file: {file_path}")
            
        except TemplateError:
            raise
        except OSError as e:
            raise FileCreationError(
                str(file_path),
                f"Failed to write file: {e}"
            )
        except Exception as e:
            raise FileCreationError(
                str(file_path),
                f"Unexpected error generating file: {e}"
            )


class AgentProjectCreator(ProjectCreator):
    """Main project creator that orchestrates the creation process with enhanced robustness."""
    
    def __init__(
        self,
        validator: ProjectValidator,
        directory_creator: DirectoryCreator,
        file_generator: FileGenerator,
        template_provider: TemplateProvider
    ) -> None:
        """Initialize with dependencies.
        
        Args:
            validator: Project validator
            directory_creator: Directory creator
            file_generator: File generator  
            template_provider: Template provider
        """
        self.validator = validator
        self.directory_creator = directory_creator
        self.file_generator = file_generator
        self.template_provider = template_provider
        self.config = CLIConfig()
    
    def create_project(self, project_name: str, output_dir: str) -> bool:
        """Create a new agent project with comprehensive error handling and rollback."""
        project_path = None
        
        try:
            # Validate inputs
            if not self.validator.validate_project_name(project_name):
                return False
                
            if not self.validator.validate_output_directory(output_dir):
                return False
            
            # Check if project already exists
            project_path = Path(output_dir) / project_name
            if project_path.exists():
                raise ProjectExistsError(str(project_path))
            
            logger.info(f"Creating agent project: {project_name}")
            
            # Create main project directory
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            self.directory_creator.create_directory_structure(
                project_path,
                self.config.PROJECT_DIRECTORIES,
                project_name
            )
            
            # Generate files from templates
            self._generate_project_files(project_path, project_name)
            
            # Success message
            self._print_success_message(project_name, project_path)
            
            return True
            
        except (ProjectExistsError, DirectoryCreationError, FileCreationError) as e:
            logger.error(str(e))
            self._cleanup_on_failure(project_path)
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating project: {e}")
            self._cleanup_on_failure(project_path)
            return False
    
    def _cleanup_on_failure(self, project_path: Path) -> None:
        """Clean up project directory on failure."""
        if project_path and project_path.exists():
            try:
                shutil.rmtree(project_path)
                logger.debug(f"Cleaned up failed project: {project_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up project directory: {e}")
    
    def _generate_project_files(self, project_path: Path, project_name: str) -> None:
        """Generate all project files from templates with enhanced error handling."""
        context = {"project_name": project_name}
        
        # Root level files
        file_mappings = {
            ".gitignore": "gitignore",
            "pyproject.toml": "pyproject_toml", 
            "README.md": "readme",
            ".env": "env",
            "Dockerfile": "dockerfile",
            "Makefile": "makefile",
            "requirements.txt": "requirements",
            "requirements-dev.txt": "requirements_dev",
            "requirements-prod.txt": "requirements_prod",
        }
        
        # Generate root level files
        for file_name, template_name in file_mappings.items():
            file_path = project_path / file_name
            self.file_generator.generate_file(file_path, template_name, context)
        
        # Generate comprehensive project structure
        self._generate_source_files(project_path, project_name, context)
        self._generate_config_files(project_path, project_name, context)
        self._generate_cli_files(project_path, project_name, context)
        self._generate_notebooks(project_path, context)
        self._generate_examples(project_path, context)
        self._generate_tools(project_path, context)
        self._generate_tests(project_path, context)
        self._generate_infrastructure(project_path, context)
        self._generate_data_directories(project_path, context)
        self._generate_docs(project_path, context)
        self._generate_github_workflows(project_path, context)
        self._generate_vscode_config(project_path, context)
        self._generate_scripts(project_path, context)
        self._generate_quickstart_files(project_path, context)
    
    def _generate_source_files(self, project_path: Path, project_name: str, context: Dict[str, Any]) -> None:
        """Generate comprehensive source code structure."""
        src_path = project_path / "src" / project_name
        
        # Application layer
        self._generate_application_files(src_path, context)
        
        # Core layer
        self._generate_core_files(src_path, context)
        
        # Tools layer
        self._generate_tools_files(src_path, context)
        
        # MCP layer
        self._generate_mcp_files(src_path, context)
        
        # Prompts layer
        self._generate_prompts_files(src_path, context)
        
        # Infrastructure layer
        self._generate_infrastructure_files(src_path, context)
        
        # Utils layer
        self._generate_utils_files(src_path, context)
    
    def _generate_application_files(self, src_path: Path, context: Dict[str, Any]) -> None:
        """Generate application layer files."""
        app_path = src_path / "application"
        
        # Services
        services_path = app_path / "services"
        self.file_generator.generate_file(
            services_path / "__init__.py", "__init__.py", context
        )
        self.file_generator.generate_file(
            services_path / "agent_service.py", "agent_service", context
        )
        self.file_generator.generate_file(
            services_path / "memory_service.py", "memory_service", context
        )
        self.file_generator.generate_file(
            services_path / "tool_service.py", "tool_service", context
        )
        
        # Interfaces
        interfaces_path = app_path / "interfaces"
        self.file_generator.generate_file(
            interfaces_path / "__init__.py", "__init__.py", context
        )
        self.file_generator.generate_file(
            interfaces_path / "agent_interface.py", "agent_interface", context
        )
        
        # DTOs
        dto_path = app_path / "dto"
        self.file_generator.generate_file(
            dto_path / "__init__.py", "__init__.py", context
        )
        self.file_generator.generate_file(
            dto_path / "agent_dto.py", "agent_dto", context
        )
    
    def _generate_core_files(self, src_path: Path, context: Dict[str, Any]) -> None:
        """Generate core layer files."""
        core_path = src_path / "core"
        
        # Agents
        agents_path = core_path / "agents"
        self.file_generator.generate_file(
            agents_path / "__init__.py", "__init__.py", context
        )
        
        # Base agent
        base_path = agents_path / "base"
        self.file_generator.generate_file(
            base_path / "__init__.py", "__init__.py", context
        )
        self.file_generator.generate_file(
            base_path / "base_agent.py", "base_agent", context
        )
        
        # Specific agent types
        agent_types = ["conversational", "research", "automation", "analysis", "creative"]
        for agent_type in agent_types:
            agent_path = agents_path / agent_type
            self.file_generator.generate_file(
                agent_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                agent_path / f"{agent_type}_agent.py", f"{agent_type}_agent", context
            )
        
        # Memory
        memory_path = core_path / "memory"
        self.file_generator.generate_file(
            memory_path / "__init__.py", "__init__.py", context
        )
        
        memory_types = ["short_term", "long_term", "manager"]
        for memory_type in memory_types:
            mem_path = memory_path / memory_type
            self.file_generator.generate_file(
                mem_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                mem_path / f"{memory_type}_memory.py", f"{memory_type}_memory", context
            )
        
        # Planning
        planning_path = core_path / "planning"
        self.file_generator.generate_file(
            planning_path / "__init__.py", "__init__.py", context
        )
        
        planning_components = ["planner", "task_decomposer", "reasoning_engine"]
        for component in planning_components:
            comp_path = planning_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
        
        # Coordination
        coordination_path = core_path / "coordination"
        self.file_generator.generate_file(
            coordination_path / "__init__.py", "__init__.py", context
        )
        
        coord_components = ["coordinator", "agent_registry", "workflow_orchestrator"]
        for component in coord_components:
            comp_path = coordination_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
        
        # Communication
        communication_path = core_path / "communication"
        self.file_generator.generate_file(
            communication_path / "__init__.py", "__init__.py", context
        )
        
        comm_components = ["message_bus", "agent_communicator", "protocol_handler"]
        for component in comm_components:
            comp_path = communication_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
    
    def _generate_tools_files(self, src_path: Path, context: Dict[str, Any]) -> None:
        """Generate tools layer files."""
        tools_path = src_path / "tools"
        self.file_generator.generate_file(
            tools_path / "__init__.py", "__init__.py", context
        )
        
        # Base tools
        base_path = tools_path / "base"
        self.file_generator.generate_file(
            base_path / "__init__.py", "__init__.py", context
        )
        self.file_generator.generate_file(
            base_path / "base_tool.py", "base_tool", context
        )
        
        # Tool categories
        tool_categories = ["web_search", "file_operations", "database", "api", "custom"]
        for category in tool_categories:
            cat_path = tools_path / category
            self.file_generator.generate_file(
                cat_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                cat_path / f"{category}_tools.py", f"{category}_tools", context
            )
    
    def _generate_mcp_files(self, src_path: Path, context: Dict[str, Any]) -> None:
        """Generate MCP layer files."""
        mcp_path = src_path / "mcp"
        self.file_generator.generate_file(
            mcp_path / "__init__.py", "__init__.py", context
        )
        
        mcp_components = ["protocol", "server", "client", "tool_interface", "resource_interface"]
        for component in mcp_components:
            comp_path = mcp_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
    
    def _generate_prompts_files(self, src_path: Path, context: Dict[str, Any]) -> None:
        """Generate prompts layer files."""
        prompts_path = src_path / "prompts"
        self.file_generator.generate_file(
            prompts_path / "__init__.py", "__init__.py", context
        )
        
        prompt_components = ["templates", "builders", "few_shot", "chainery", "agent_specific"]
        for component in prompt_components:
            comp_path = prompts_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
    
    def _generate_infrastructure_files(self, src_path: Path, context: Dict[str, Any]) -> None:
        """Generate infrastructure layer files."""
        infra_path = src_path / "infrastructure"
        self.file_generator.generate_file(
            infra_path / "__init__.py", "__init__.py", context
        )
        
        # LLM Clients
        llm_path = infra_path / "llm_clients"
        self.file_generator.generate_file(
            llm_path / "__init__.py", "__init__.py", context
        )
        
        llm_components = ["base", "factory", "openai", "anthropic", "google", "azure", "local"]
        for component in llm_components:
            comp_path = llm_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}_client.py", f"{component}_client", context
            )
        
        # Vector Database
        vector_path = infra_path / "vector_database"
        self.file_generator.generate_file(
            vector_path / "__init__.py", "__init__.py", context
        )
        
        vector_components = ["base", "chroma", "pinecone", "weaviate"]
        for component in vector_components:
            comp_path = vector_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}_client.py", f"{component}_client", context
            )
        
        # Monitoring
        monitoring_path = infra_path / "monitoring"
        self.file_generator.generate_file(
            monitoring_path / "__init__.py", "__init__.py", context
        )
        
        monitoring_components = ["logger", "tracer", "metrics", "health_checker"]
        for component in monitoring_components:
            comp_path = monitoring_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
        
        # API
        api_path = infra_path / "api"
        self.file_generator.generate_file(
            api_path / "__init__.py", "__init__.py", context
        )
        
        api_components = ["app", "routes", "middleware"]
        for component in api_components:
            comp_path = api_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
        
        # Storage
        storage_path = infra_path / "storage"
        self.file_generator.generate_file(
            storage_path / "__init__.py", "__init__.py", context
        )
        
        storage_components = ["cache", "database", "file_system"]
        for component in storage_components:
            comp_path = storage_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
    
    def _generate_utils_files(self, src_path: Path, context: Dict[str, Any]) -> None:
        """Generate utils layer files."""
        utils_path = src_path / "utils"
        self.file_generator.generate_file(
            utils_path / "__init__.py", "__init__.py", context
        )
        
        utils_components = ["rate_limiter", "token_counter", "cache", "validators", "helpers"]
        for component in utils_components:
            comp_path = utils_path / component
            self.file_generator.generate_file(
                comp_path / "__init__.py", "__init__.py", context
            )
            self.file_generator.generate_file(
                comp_path / f"{component}.py", component, context
            )
    
    def _generate_config_files(self, project_path: Path, project_name: str, context: Dict[str, Any]) -> None:
        """Generate configuration files."""
        config_path = project_path / "config"
        
        # Base configuration
        self.file_generator.generate_file(
            config_path / "base.yaml", "config_base", context
        )
        
        # Environment configurations
        env_path = config_path / "environments"
        environments = ["development", "staging", "production"]
        for env in environments:
            self.file_generator.generate_file(
                env_path / f"{env}.yaml", f"config_{env}", context
            )
        
        # Agent configurations
        agents_path = config_path / "agents"
        agent_types = ["conversational", "research", "automation", "analysis"]
        for agent_type in agent_types:
            self.file_generator.generate_file(
                agents_path / f"{agent_type}.yaml", f"config_agent_{agent_type}", context
            )
        
        # LLM provider configurations
        llm_path = config_path / "llm_providers"
        providers = ["openai", "anthropic", "google", "azure", "local"]
        for provider in providers:
            self.file_generator.generate_file(
                llm_path / f"{provider}.yaml", f"config_llm_{provider}", context
            )
        
        # Additional config files
        self.file_generator.generate_file(
            config_path / "prompt_templates.yaml", "config_prompt_templates", context
        )
        self.file_generator.generate_file(
            config_path / "logging_config.yaml", "config_logging", context
        )
    
    def _generate_cli_files(self, project_path: Path, project_name: str, context: Dict[str, Any]) -> None:
        """Generate CLI interface files."""
        cli_path = project_path / "cli"
        
        # Main CLI
        self.file_generator.generate_file(
            cli_path / "__main__.py", "cli_main", context
        )
        
        # Commands
        commands_path = cli_path / "commands"
        commands = ["init", "run", "deploy", "monitor", "debug", "config"]
        for command in commands:
            self.file_generator.generate_file(
                commands_path / f"{command}.py", f"cli_command_{command}", context
            )
        
        # Interactive
        interactive_path = cli_path / "interactive"
        interactive_components = ["chat", "shell", "playground"]
        for component in interactive_components:
            self.file_generator.generate_file(
                interactive_path / f"{component}.py", f"cli_interactive_{component}", context
            )
        
        # Utils
        utils_path = cli_path / "utils"
        cli_utils = ["output", "progress", "validation", "error_recovery"]
        for util in cli_utils:
            self.file_generator.generate_file(
                utils_path / f"{util}.py", f"cli_util_{util}", context
            )
    
    def _generate_examples(self, project_path: Path, context: Dict[str, Any]) -> None:
        """Generate example files."""
        examples_path = project_path / "examples"
        
        examples = [
            "conversational_agent.py",
            "research_assistant.py", 
            "task_automation.py",
            "data_analysis_agent.py",
            "multi_agent_scenario.py",
            "mcp_integration_demo.py"
        ]
        
        for example in examples:
            self.file_generator.generate_file(
                examples_path / example, f"example_{example.replace('.py', '')}", context
            )
    
    def _generate_infrastructure(self, project_path: Path, context: Dict[str, Any]) -> None:
        """Generate infrastructure files."""
        infra_path = project_path / "infrastructure"
        
        # Docker
        docker_path = infra_path / "docker"
        self.file_generator.generate_file(
            docker_path / "Dockerfile", "dockerfile", context
        )
        self.file_generator.generate_file(
            docker_path / "docker-compose.yml", "docker_compose", context
        )
        
        # Kubernetes
        k8s_path = infra_path / "kubernetes"
        k8s_files = ["deployment.yaml", "service.yaml", "configmap.yaml", "secret.yaml"]
        for file in k8s_files:
            self.file_generator.generate_file(
                k8s_path / file, f"k8s_{file.replace('.yaml', '')}", context
            )
        
        # Terraform
        terraform_path = infra_path / "terraform"
        tf_files = ["main.tf", "variables.tf", "outputs.tf", "providers.tf"]
        for file in tf_files:
            self.file_generator.generate_file(
                terraform_path / file, f"terraform_{file.replace('.tf', '')}", context
            )
        
        # Observability
        obs_path = infra_path / "observability"
        obs_files = ["prometheus.yml", "grafana.yml", "alerting.yml"]
        for file in obs_files:
            self.file_generator.generate_file(
                obs_path / file, f"observability_{file.replace('.yml', '')}", context
            )
    
    def _generate_data_directories(self, project_path: Path, context: Dict[str, Any]) -> None:
        """Generate data directory structure."""
        data_path = project_path / "data"
        
        # Create .gitkeep files for empty directories
        data_dirs = ["cache", "prompts", "outputs", "logs", "models"]
        for dir_name in data_dirs:
            dir_path = data_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / ".gitkeep").touch()
    
    def _generate_docs(self, project_path: Path, context: Dict[str, Any]) -> None:
        """Generate documentation files."""
        docs_path = project_path / "docs"
        
        doc_files = [
            "getting_started.md",
            "agent_types.md", 
            "deployment.md",
            "api_reference.md",
            "troubleshooting.md"
        ]
        
        for doc in doc_files:
            self.file_generator.generate_file(
                docs_path / doc, f"doc_{doc.replace('.md', '')}", context
            )
    
    def _generate_github_workflows(self, project_path: Path, context: Dict[str, Any]) -> None:
        """Generate GitHub workflow files."""
        workflows_path = project_path / ".github" / "workflows"
        
        workflow_files = ["ci.yml", "cd.yml", "release.yml"]
        for workflow in workflow_files:
            self.file_generator.generate_file(
                workflows_path / workflow, f"github_{workflow.replace('.yml', '')}", context
            )
    
    def _generate_vscode_config(self, project_path: Path, context: Dict[str, Any]) -> None:
        """Generate VS Code configuration."""
        vscode_path = project_path / ".vscode"
        
        vscode_files = ["settings.json", "extensions.json", "launch.json"]
        for file in vscode_files:
            self.file_generator.generate_file(
                vscode_path / file, f"vscode_{file.replace('.json', '')}", context
            )
    
    def _generate_scripts(self, project_path: Path, context: Dict[str, Any]) -> None:
        """Generate utility scripts."""
        scripts_path = project_path / "scripts"
        
        script_files = ["setup.sh", "deploy.sh", "test.sh", "lint.sh"]
        for script in script_files:
            self.file_generator.generate_file(
                scripts_path / script, f"script_{script.replace('.sh', '')}", context
            )
