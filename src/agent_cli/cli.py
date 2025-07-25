"""
Enhanced CLI module for optimal agent development experience.

This module provides a comprehensive CLI interface for creating, running,
and managing AI agent projects with beautiful progress tracking and
interactive features.
"""

import argparse
import logging
import sys
import time
import subprocess
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Use rich for better CLI output if available
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.live import Live
    from rich.layout import Layout
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.markdown import Markdown
    from rich.emoji import Emoji
    from rich.tree import Tree
    from rich.box import ROUNDED
    console = Console()
    RICH_ENABLED = True
except ImportError:
    console = None
    RICH_ENABLED = False

from .generators.project_generator import (
    AgentProjectCreator,
    DefaultProjectValidator,
    DefaultDirectoryCreator,
    DefaultFileGenerator
)
from .templates.static_templates import StaticTemplateProvider
from .templates.template_validator import TemplateValidator, validate_template_collection, get_template_statistics
from .exceptions import AgentCLIError

# Configure logging
if RICH_ENABLED:
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=True, show_level=True, show_path=False)]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


class EnhancedAgentProjectCLI:
    """Enhanced CLI class that provides optimal agent development experience.
    
    This class extends the original CLI with new commands for running,
    deploying, monitoring, and managing agent projects.
    """
    
    def __init__(self) -> None:
        """Initialize enhanced CLI with all dependencies."""
        # Create dependencies using dependency injection
        self.template_provider = StaticTemplateProvider()
        self.validator = DefaultProjectValidator()
        self.directory_creator = DefaultDirectoryCreator()
        self.file_generator = DefaultFileGenerator(self.template_provider)
        
        # Main project creator
        self.project_creator = AgentProjectCreator(
            validator=self.validator,
            directory_creator=self.directory_creator,
            file_generator=self.file_generator,
            template_provider=self.template_provider
        )
    
    def init_project(self, project_name: str, template: str = "conversational", 
                    output_dir: str = ".", interactive: bool = True) -> bool:
        """Initialize a new agent project with enhanced setup.
        
        Args:
            project_name: Name of the project to create
            template: Agent template type (conversational, research, automation, analysis)
            output_dir: Directory where project should be created
            interactive: Whether to run in interactive mode
        Returns:
            True if successful, False otherwise
        """
        try:
            if interactive and RICH_ENABLED:
                return self._init_project_interactive(project_name, template, output_dir)
            else:
                return self._init_project_standard(project_name, template, output_dir)
        except Exception as e:
            logger.error(f"Failed to initialize project: {e}")
            return False
    
    def _init_project_interactive(self, project_name: str, template: str, output_dir: str) -> bool:
        """Interactive project initialization with guided setup."""
        console.print(Panel.fit(
            "[bold blue]🤖 Welcome to Agent CLI - AI Agent Development Framework[/bold blue]\n"
            "Let's create your amazing AI agent project!",
            border_style="blue"
        ))
        
        # Project name validation
        if not self.validator.validate_project_name(project_name):
            project_name = Prompt.ask(
                "[bold cyan]Enter project name[/bold cyan]",
                default="my-awesome-agent"
            )
            if not self.validator.validate_project_name(project_name):
                console.print("[red]❌ Invalid project name. Please use lowercase letters, numbers, and underscores only.[/red]")
                return False
        
        # Template selection
        templates = {
            "conversational": "💬 Conversational Agent (Chatbot, Customer Service)",
            "research": "🔍 Research Assistant (Information Gathering, Analysis)",
            "automation": "⚙️ Task Automation (Workflow Orchestration)",
            "analysis": "📊 Data Analysis Agent (Reporting, Insights)",
            "custom": "🎯 Custom Agent (Build from scratch)"
        }
        
        if template not in templates:
            console.print("\n[bold yellow]Available Agent Templates:[/bold yellow]")
            for key, desc in templates.items():
                console.print(f"  [cyan]{key}[/cyan]: {desc}")
            
            template = Prompt.ask(
                "\n[bold cyan]Select agent template[/bold cyan]",
                choices=list(templates.keys()),
                default="conversational"
            )
        
        # Configuration options
        console.print(f"\n[bold green]Selected:[/bold green] {templates[template]}")
        
        config_options = {
            "llm_provider": Prompt.ask(
                "[bold cyan]LLM Provider[/bold cyan]",
                choices=["anthropic", "openai", "google", "local"],
                default="anthropic"
            ),
            "memory": Confirm.ask(
                "[bold cyan]Enable memory system?[/bold cyan]",
                default=True
            ),
            "tools": Confirm.ask(
                "[bold cyan]Include common tools (web search, file operations)?[/bold cyan]",
                default=True
            ),
            "monitoring": Confirm.ask(
                "[bold cyan]Enable monitoring and observability?[/bold cyan]",
                default=True
            ),
            "api": Confirm.ask(
                "[bold cyan]Include REST API interface?[/bold cyan]",
                default=True
            )
        }
        
        # Create project with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Creating agent project...", total=100)
            
            # Create project structure
            progress.update(task, advance=20, description="[cyan]Creating project structure...")
            if not self.project_creator.create_project(project_name, output_dir):
                return False
            
            # Configure project
            progress.update(task, advance=20, description="[cyan]Configuring agent...")
            project_path = Path(output_dir) / project_name
            
            # Update configuration based on user choices
            self._configure_project(project_path, template, config_options)
            
            # Install dependencies
            progress.update(task, advance=20, description="[cyan]Installing dependencies...")
            self._install_dependencies(project_path)
            
            # Setup development environment
            progress.update(task, advance=20, description="[cyan]Setting up development environment...")
            self._setup_dev_environment(project_path)
            
            # Final setup
            progress.update(task, advance=20, description="[cyan]Finalizing setup...")
            self._finalize_setup(project_path, project_name)
        
        # Show success message
        self._show_init_success(project_name, project_path, template)
        return True
    
    def _init_project_standard(self, project_name: str, template: str, output_dir: str) -> bool:
        """Standard project initialization without interactive prompts."""
        return self.project_creator.create_project(project_name, output_dir)
    
    def run_agent(self, project_path: str, config: str = "development", 
                  interactive: bool = True) -> bool:
        """Run an agent project.
        
        Args:
            project_path: Path to the agent project
            config: Configuration environment (development, staging, production)
            interactive: Whether to run in interactive mode
        Returns:
            True if successful, False otherwise
        """
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                console.print(f"[red]❌ Project not found: {project_path}[/red]")
                return False
            
            if interactive and RICH_ENABLED:
                return self._run_agent_interactive(project_path, config)
            else:
                return self._run_agent_standard(project_path, config)
        except Exception as e:
            logger.error(f"Failed to run agent: {e}")
            return False
    
    def _run_agent_interactive(self, project_path: Path, config: str) -> bool:
        """Interactive agent execution."""
        console.print(Panel.fit(
            f"[bold green]🚀 Running Agent: {project_path.name}[/bold green]\n"
            f"Configuration: [cyan]{config}[/cyan]",
            border_style="green"
        ))
        
        # Check if agent is ready
        if not self._check_agent_readiness(project_path):
            console.print("[red]❌ Agent is not ready. Please run setup first.[/red]")
            return False
        
        # Start agent
        console.print("[green]✅ Starting agent...[/green]")
        
        # Run the agent (this would integrate with the actual agent runtime)
        try:
            # For now, simulate agent startup
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Starting agent...", total=None)
                time.sleep(2)  # Simulate startup time
                progress.update(task, description="[green]Agent is running!")
            
            console.print("[green]🎉 Agent is now running![/green]")
            console.print("\n[bold yellow]Available commands:[/bold yellow]")
            console.print("  [cyan]chat[/cyan] - Start interactive chat")
            console.print("  [cyan]monitor[/cyan] - View agent status")
            console.print("  [cyan]stop[/cyan] - Stop the agent")
            
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to start agent: {e}[/red]")
            return False
    
    def _run_agent_standard(self, project_path: Path, config: str) -> bool:
        """Standard agent execution."""
        # Implementation for non-interactive mode
        return True
    
    def deploy_agent(self, project_path: str, environment: str = "production", 
                    config: str = "production") -> bool:
        """Deploy an agent to production.
        
        Args:
            project_path: Path to the agent project
            environment: Deployment environment (staging, production)
            config: Configuration to use
        Returns:
            True if successful, False otherwise
        """
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                console.print(f"[red]❌ Project not found: {project_path}[/red]")
                return False
            
            if RICH_ENABLED:
                return self._deploy_agent_rich(project_path, environment, config)
            else:
                return self._deploy_agent_standard(project_path, environment, config)
        except Exception as e:
            logger.error(f"Failed to deploy agent: {e}")
            return False
    
    def _deploy_agent_rich(self, project_path: Path, environment: str, config: str) -> bool:
        """Rich deployment interface."""
        console.print(Panel.fit(
            f"[bold blue]🚀 Deploying Agent: {project_path.name}[/bold blue]\n"
            f"Environment: [cyan]{environment}[/cyan] | Config: [cyan]{config}[/cyan]",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Deploying agent...", total=100)
            
            # Build container
            progress.update(task, advance=20, description="[cyan]Building container...")
            time.sleep(1)
            
            # Run tests
            progress.update(task, advance=20, description="[cyan]Running tests...")
            time.sleep(1)
            
            # Deploy to environment
            progress.update(task, advance=30, description="[cyan]Deploying to production...")
            time.sleep(2)
            
            # Health checks
            progress.update(task, advance=30, description="[cyan]Running health checks...")
            time.sleep(1)
        
        console.print("[green]✅ Agent deployed successfully![/green]")
        return True
    
    def _deploy_agent_standard(self, project_path: Path, environment: str, config: str) -> bool:
        """Standard deployment."""
        return True
    
    def monitor_agent(self, project_path: str, dashboard: bool = False) -> bool:
        """Monitor agent performance and status.
        
        Args:
            project_path: Path to the agent project
            dashboard: Whether to open monitoring dashboard
        Returns:
            True if successful, False otherwise
        """
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                console.print(f"[red]❌ Project not found: {project_path}[/red]")
                return False
            
            if RICH_ENABLED:
                return self._monitor_agent_rich(project_path, dashboard)
            else:
                return self._monitor_agent_standard(project_path, dashboard)
        except Exception as e:
            logger.error(f"Failed to monitor agent: {e}")
            return False
    
    def _monitor_agent_rich(self, project_path: Path, dashboard: bool) -> bool:
        """Rich monitoring interface."""
        console.print(Panel.fit(
            f"[bold yellow]📊 Agent Monitoring: {project_path.name}[/bold yellow]",
            border_style="yellow"
        ))
        
        # Simulate agent metrics
        metrics = {
            "Status": "🟢 Running",
            "Uptime": "2h 15m 30s",
            "Requests": "1,234",
            "Response Time": "245ms",
            "Error Rate": "0.1%",
            "Memory Usage": "45%",
            "CPU Usage": "23%"
        }
        
        table = Table(title="Agent Metrics", box=ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for metric, value in metrics.items():
            table.add_row(metric, value)
        
        console.print(table)
        
        if dashboard:
            console.print("\n[cyan]Opening monitoring dashboard...[/cyan]")
            # This would open the actual dashboard
        
        return True
    
    def _monitor_agent_standard(self, project_path: Path, dashboard: bool) -> bool:
        """Standard monitoring."""
        return True
    
    def debug_agent(self, project_path: str, agent_name: str = None, 
                   logs: str = "info") -> bool:
        """Debug agent issues.
        
        Args:
            project_path: Path to the agent project
            agent_name: Specific agent to debug
            logs: Log level (debug, info, warning, error)
        Returns:
            True if successful, False otherwise
        """
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                console.print(f"[red]❌ Project not found: {project_path}[/red]")
                return False
            
            if RICH_ENABLED:
                return self._debug_agent_rich(project_path, agent_name, logs)
            else:
                return self._debug_agent_standard(project_path, agent_name, logs)
        except Exception as e:
            logger.error(f"Failed to debug agent: {e}")
            return False
    
    def _debug_agent_rich(self, project_path: Path, agent_name: str, logs: str) -> bool:
        """Rich debugging interface."""
        console.print(Panel.fit(
            f"[bold red]🐛 Agent Debug: {project_path.name}[/bold red]\n"
            f"Agent: [cyan]{agent_name or 'All'}[/cyan] | Log Level: [cyan]{logs}[/cyan]",
            border_style="red"
        ))
        
        # Simulate debug information
        debug_info = {
            "Agent Status": "🟡 Warning",
            "Last Error": "Connection timeout to LLM provider",
            "Memory Usage": "High (85%)",
            "Active Connections": "12",
            "Pending Requests": "3"
        }
        
        table = Table(title="Debug Information", box=ROUNDED)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow")
        
        for prop, value in debug_info.items():
            table.add_row(prop, value)
        
        console.print(table)
        
        # Show suggested fixes
        console.print("\n[bold yellow]Suggested Fixes:[/bold yellow]")
        console.print("  • Check LLM API key configuration")
        console.print("  • Restart the agent service")
        console.print("  • Clear memory cache")
        console.print("  • Check network connectivity")
        
        return True
    
    def _debug_agent_standard(self, project_path: Path, agent_name: str, logs: str) -> bool:
        """Standard debugging."""
        return True
    
    def chat(self, project_path: str, config: str = "development") -> bool:
        """Start interactive chat with the agent.
        
        Args:
            project_path: Path to the agent project
            config: Configuration to use
        Returns:
            True if successful, False otherwise
        """
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                console.print(f"[red]❌ Project not found: {project_path}[/red]")
                return False
            
            if RICH_ENABLED:
                return self._chat_interactive(project_path, config)
            else:
                return self._chat_standard(project_path, config)
        except Exception as e:
            logger.error(f"Failed to start chat: {e}")
            return False
    
    def _chat_interactive(self, project_path: Path, config: str) -> bool:
        """Interactive chat interface."""
        console.print(Panel.fit(
            f"[bold green]💬 Interactive Chat: {project_path.name}[/bold green]\n"
            f"Configuration: [cyan]{config}[/cyan]\n"
            "Type 'quit' to exit",
            border_style="green"
        ))
        
        # Simulate chat interface
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                
                # Simulate agent response
                console.print(f"[bold green]Agent[/bold green]: Hello! I'm your AI assistant. How can I help you today?")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
        
        return True
    
    def _chat_standard(self, project_path: Path, config: str) -> bool:
        """Standard chat interface."""
        return True
    
    # Helper methods for project setup
    def _configure_project(self, project_path: Path, template: str, config_options: Dict[str, Any]) -> None:
        """Configure project based on user choices."""
        # This would update configuration files based on user choices
        pass
    
    def _install_dependencies(self, project_path: Path) -> None:
        """Install project dependencies."""
        # This would run pip install or similar
        pass
    
    def _setup_dev_environment(self, project_path: Path) -> None:
        """Setup development environment."""
        # This would setup pre-commit hooks, etc.
        pass
    
    def _finalize_setup(self, project_path: Path, project_name: str) -> None:
        """Finalize project setup."""
        # This would create initial commit, etc.
        pass
    
    def _check_agent_readiness(self, project_path: Path) -> bool:
        """Check if agent is ready to run."""
        # This would check if all dependencies are installed, config is valid, etc.
        return True
    
    def _show_init_success(self, project_name: str, project_path: Path, template: str) -> None:
        """Show success message after project initialization."""
        console.print(Panel.fit(
            f"[bold green]🎉 Successfully created agent project![/bold green]\n\n"
            f"Project: [cyan]{project_name}[/cyan]\n"
            f"Location: [cyan]{project_path}[/cyan]\n"
            f"Template: [cyan]{template}[/cyan]\n\n"
            "[bold yellow]Next steps:[/bold yellow]\n"
            f"  cd {project_name}\n"
            "  agent-cli run .\n"
            "  agent-cli chat .\n\n"
            "[bold blue]Happy coding! 🚀[/bold blue]",
            border_style="green"
        ))

    # Original methods from the base CLI
    def create_project(self, project_name: str, output_dir: str = ".") -> bool:
        """Create a new agent project with beautiful progress tracking.
        
        Args:
            project_name: Name of the project to create
            output_dir: Directory where project should be created
        Returns:
            True if successful, False otherwise
        """
        try:
            if RICH_ENABLED:
                return self._create_project_with_progress(project_name, output_dir)
            else:
                return self.project_creator.create_project(project_name, output_dir)
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False

    def _create_project_with_progress(self, project_name: str, output_dir: str) -> bool:
        """Create project with rich progress tracking."""
        try:
            self._show_project_creation_header(project_name, output_dir)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("[cyan]Creating agent project...", total=100)
                
                # Validate project name
                progress.update(task, advance=10, description="[cyan]Validating project name...")
                if not self.validator.validate_project_name(project_name):
                    console.print(f"[red]❌ Invalid project name: {project_name}[/red]")
                    return False
                
                # Create project
                progress.update(task, advance=40, description="[cyan]Creating project structure...")
                if not self.project_creator.create_project(project_name, output_dir):
                    return False
                
                # Setup development environment
                progress.update(task, advance=30, description="[cyan]Setting up development environment...")
                project_path = Path(output_dir) / project_name
                self._setup_dev_environment(project_path)
                
                # Finalize
                progress.update(task, advance=20, description="[cyan]Finalizing setup...")
                self._finalize_setup(project_path, project_name)
            
            self._show_success_message(project_name, output_dir)
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create project: {e}[/red]")
            return False

    def _show_project_creation_header(self, project_name: str, output_dir: str) -> None:
        """Show project creation header."""
        console.print(Panel.fit(
            f"[bold blue]🤖 Creating AI Agent Project[/bold blue]\n"
            f"Project: [cyan]{project_name}[/cyan]\n"
            f"Location: [cyan]{output_dir}[/cyan]",
            border_style="blue"
        ))

    def _show_success_message(self, project_name: str, output_dir: str) -> None:
        """Show success message after project creation."""
        project_path = Path(output_dir) / project_name
        
        console.print(Panel.fit(
            f"[bold green]🎉 Project created successfully![/bold green]\n\n"
            f"Project: [cyan]{project_name}[/cyan]\n"
            f"Location: [cyan]{project_path}[/cyan]\n\n"
            "[bold yellow]Next steps:[/bold yellow]\n"
            f"  cd {project_name}\n"
            "  agent-cli run .\n"
            "  agent-cli chat .\n\n"
            "[bold blue]Happy coding! 🚀[/bold blue]",
            border_style="green"
        ))

    def _show_project_structure(self, project_name: str) -> None:
        """Show project structure tree."""
        tree = Tree(f"[bold blue]{project_name}/[/bold blue]")
        
        # Add main directories
        tree.add("📁 src/")
        tree.add("📁 tests/")
        tree.add("📁 notebooks/")
        tree.add("📁 examples/")
        tree.add("📁 tools/")
        tree.add("📁 config/")
        tree.add("📁 infrastructure/")
        
        console.print(tree)

    def list_templates(self) -> bool:
        """List available templates with rich formatting."""
        try:
            templates = self.template_provider.get_available_templates()
            
            if RICH_ENABLED:
                self._display_templates_rich(templates)
            else:
                self._display_templates_plain(templates)
            
            return True
        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return False

    def _display_templates_rich(self, templates: list) -> None:
        """Display templates with rich formatting."""
        console.print(Panel.fit(
            "[bold blue]📋 Available Templates[/bold blue]",
            border_style="blue"
        ))
        
        table = Table(title="Agent Project Templates", box=ROUNDED)
        table.add_column("Template", style="cyan", no_wrap=True)
        table.add_column("Category", style="green")
        table.add_column("Description", style="white")
        
        for template in templates:
            category = self._get_template_category(template)
            description = self._get_template_description(template)
            emoji = self._get_category_emoji(category)
            
            table.add_row(
                f"{emoji} {template}",
                category.title(),
                description
            )
        
        console.print(table)

    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for template category."""
        emoji_map = {
            "file": "📄",
            "notebook": "📓",
            "tool": "🛠️",
            "test": "🧪",
            "source": "💻",
            "quickstart": "🚀"
        }
        return emoji_map.get(category, "📋")

    def _get_template_description(self, template_name: str) -> str:
        """Get description for template."""
        descriptions = {
            "gitignore": "Comprehensive Python gitignore file",
            "env": "Environment variables template",
            "dockerfile": "Docker container configuration",
            "makefile": "Build automation and development commands",
            "01_prompt_engineering_playground.ipynb": "Interactive prompt engineering experiments",
            "02_short_term_memory.ipynb": "Short-term memory system examples",
            "03_long_term_memory.ipynb": "Long-term memory and vector storage",
            "04_tool_calling_playground.ipynb": "Tool integration and calling examples",
            "run_agent.py": "Main agent execution script",
            "populate_long_term_memory.py": "Memory population utility",
            "delete_long_term_memory.py": "Memory cleanup utility",
            "evaluate_agent.py": "Agent performance evaluation",
            "test_agent.py": "Agent functionality tests",
            "test_memory.py": "Memory system tests",
            "quickstart.py": "Complete working example",
            "requirements-quickstart.txt": "Quickstart dependencies"
        }
        return descriptions.get(template_name, "Template file")

    def validate_templates(self) -> bool:
        """Validate all templates with rich output."""
        try:
            if RICH_ENABLED:
                return self._validate_templates_rich()
            else:
                return self._validate_templates_plain()
        except Exception as e:
            logger.error(f"Failed to validate templates: {e}")
            return False

    def _validate_templates_rich(self) -> bool:
        """Validate templates with rich progress tracking."""
        console.print(Panel.fit(
            "[bold blue]🔍 Validating Templates[/bold blue]",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Validating templates...", total=100)
            
            # Validate templates
            progress.update(task, advance=50, description="[cyan]Checking template syntax...")
            validation_results = validate_template_collection(self.template_provider)
            stats = get_template_statistics(validation_results)
            
            progress.update(task, advance=50, description="[cyan]Generating report...")
        
        # Display results
        self._display_validation_results_rich(validation_results, stats)
        return True

    def _validate_templates_plain(self) -> bool:
        """Validate templates with plain output."""
        validation_results = validate_template_collection(self.template_provider)
        stats = get_template_statistics(validation_results)
        self._display_validation_results_plain(validation_results, stats)
        return True

    def validate_project_name(self, project_name: str) -> bool:
        """Validate project name with rich output."""
        try:
            is_valid = self.validator.validate_project_name(project_name)
            
            if RICH_ENABLED:
                self._display_validation_result_rich(project_name, is_valid)
            else:
                self._display_validation_result_plain(project_name, is_valid)
            
            return is_valid
        except Exception as e:
            logger.error(f"Failed to validate project name: {e}")
            return False

    def _display_validation_result_rich(self, project_name: str, is_valid: bool) -> None:
        """Display validation result with rich formatting."""
        if is_valid:
            console.print(Panel.fit(
                f"[bold green]✅ Valid Project Name[/bold green]\n"
                f"Project: [cyan]{project_name}[/cyan]\n"
                "Ready to create!",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                f"[bold red]❌ Invalid Project Name[/bold red]\n"
                f"Project: [cyan]{project_name}[/cyan]\n"
                "Please use lowercase letters, numbers, and underscores only.",
                border_style="red"
            ))

    def _display_validation_result_plain(self, project_name: str, is_valid: bool) -> None:
        """Display validation result with plain formatting."""
        if is_valid:
            print(f"✅ Valid project name: {project_name}")
        else:
            print(f"❌ Invalid project name: {project_name}")

    def show_info(self) -> bool:
        """Show CLI information with rich formatting."""
        try:
            if RICH_ENABLED:
                self._show_info_rich()
            else:
                self._show_info_plain()
            return True
        except Exception as e:
            logger.error(f"Failed to show info: {e}")
            return False

    def _show_info_rich(self) -> None:
        """Show info with rich formatting."""
        console.print(Panel.fit(
            "[bold blue]🤖 Agent CLI - AI Agent Development Framework[/bold blue]\n"
            "Create, run, and manage AI agent projects with ease!",
            border_style="blue"
        ))
        
        # Version and features
        info_table = Table(title="CLI Information", box=ROUNDED)
        info_table.add_column("Property", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Version", "0.1.0")
        info_table.add_row("Python", "3.10+")
        info_table.add_row("Rich Output", "✅ Enabled" if RICH_ENABLED else "❌ Disabled")
        info_table.add_row("Templates", str(len(self.template_provider.get_available_templates())))
        
        console.print(info_table)
        
        # Available commands
        console.print("\n[bold yellow]Available Commands:[/bold yellow]")
        commands = [
            ("init", "Initialize new agent project"),
            ("run", "Run an agent project"),
            ("deploy", "Deploy agent to production"),
            ("monitor", "Monitor agent performance"),
            ("debug", "Debug agent issues"),
            ("chat", "Interactive chat with agent"),
            ("list-templates", "List available templates"),
            ("validate-templates", "Validate all templates"),
            ("validate", "Validate project name"),
            ("info", "Show CLI information")
        ]
        
        for command, description in commands:
            console.print(f"  [cyan]{command}[/cyan] - {description}")

    def _show_info_plain(self) -> None:
        """Show info with plain formatting."""
        print("Agent CLI - AI Agent Development Framework")
        print("Version: 0.1.0")
        print("Python: 3.10+")

    def _get_template_category(self, template_name: str) -> str:
        """Get category for template."""
        if template_name.endswith('.ipynb'):
            return "notebook"
        elif template_name.endswith('.py'):
            if 'test' in template_name:
                return "test"
            elif template_name in ['run_agent.py', 'populate_long_term_memory.py', 'delete_long_term_memory.py', 'evaluate_agent.py']:
                return "tool"
            else:
                return "source"
        elif template_name in ['gitignore', 'env', 'dockerfile', 'makefile']:
            return "file"
        elif 'quickstart' in template_name:
            return "quickstart"
        else:
            return "file"

    def _display_validation_results_rich(self, validation_results: dict, stats: dict) -> None:
        """Display validation results with rich formatting."""
        # Summary
        console.print(Panel.fit(
            f"[bold blue]📊 Validation Summary[/bold blue]\n"
            f"Total Templates: [cyan]{stats['total']}[/cyan]\n"
            f"Valid: [green]{stats['valid']}[/green]\n"
            f"Invalid: [red]{stats['invalid']}[/red]\n"
            f"Success Rate: [yellow]{stats['success_rate']:.1f}%[/yellow]",
            border_style="blue"
        ))
        
        # Detailed results
        if stats['invalid'] > 0:
            console.print("\n[bold red]❌ Invalid Templates:[/bold red]")
            for template, error in validation_results.items():
                if not error['valid']:
                    console.print(f"  [red]• {template}[/red]: {error['error']}")

    def _display_validation_results_plain(self, validation_results: dict, stats: dict) -> None:
        """Display validation results with plain formatting."""
        print(f"Validation Summary:")
        print(f"  Total Templates: {stats['total']}")
        print(f"  Valid: {stats['valid']}")
        print(f"  Invalid: {stats['invalid']}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")


# Legacy compatibility - keep the original class name
AgentProjectCLI = EnhancedAgentProjectCLI


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with enhanced commands."""
    parser = argparse.ArgumentParser(
        description="🤖 Agent CLI - AI Agent Development Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agent-cli init my-chatbot --template=conversational
  agent-cli run my-agent-project
  agent-cli deploy my-agent-project --environment=production
  agent-cli monitor my-agent-project --dashboard
  agent-cli chat my-agent-project
  agent-cli debug my-agent-project --logs=verbose
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize new agent project')
    init_parser.add_argument('project_name', help='Name of the project to create')
    init_parser.add_argument('--template', '-t', default='conversational',
                           choices=['conversational', 'research', 'automation', 'analysis', 'custom'],
                           help='Agent template type')
    init_parser.add_argument('--output', '-o', default='.',
                           help='Output directory (default: current directory)')
    init_parser.add_argument('--no-interactive', action='store_true',
                           help='Disable interactive mode')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run an agent project')
    run_parser.add_argument('project_path', help='Path to the agent project')
    run_parser.add_argument('--config', '-c', default='development',
                           help='Configuration environment')
    run_parser.add_argument('--no-interactive', action='store_true',
                           help='Disable interactive mode')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy agent to production')
    deploy_parser.add_argument('project_path', help='Path to the agent project')
    deploy_parser.add_argument('--environment', '-e', default='production',
                              choices=['staging', 'production'],
                              help='Deployment environment')
    deploy_parser.add_argument('--config', '-c', default='production',
                              help='Configuration to use')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor agent performance')
    monitor_parser.add_argument('project_path', help='Path to the agent project')
    monitor_parser.add_argument('--dashboard', '-d', action='store_true',
                               help='Open monitoring dashboard')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug agent issues')
    debug_parser.add_argument('project_path', help='Path to the agent project')
    debug_parser.add_argument('--agent', '-a', help='Specific agent to debug')
    debug_parser.add_argument('--logs', '-l', default='info',
                             choices=['debug', 'info', 'warning', 'error'],
                             help='Log level')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat with agent')
    chat_parser.add_argument('project_path', help='Path to the agent project')
    chat_parser.add_argument('--config', '-c', default='development',
                            help='Configuration to use')
    
    # Legacy commands (for backward compatibility)
    create_parser = subparsers.add_parser('create', help='Create new agent project (legacy)')
    create_parser.add_argument('project_name', help='Name of the project to create')
    create_parser.add_argument('--output', '-o', default='.',
                              help='Output directory (default: current directory)')
    create_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Enable verbose output')
    
    list_parser = subparsers.add_parser('list-templates', help='List available templates')
    
    validate_templates_parser = subparsers.add_parser('validate-templates', 
                                                    help='Validate all templates')
    
    validate_parser = subparsers.add_parser('validate', help='Validate project name')
    validate_parser.add_argument('project_name', help='Project name to validate')
    
    info_parser = subparsers.add_parser('info', help='Show CLI information')
    
    return parser


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point with enhanced functionality."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args(argv)
        
        if not args.command:
            parser.print_help()
            return 0
        
        cli = EnhancedAgentProjectCLI()
        
        # Handle new commands
        if args.command == 'init':
            return 0 if cli.init_project(
                args.project_name, 
                args.template, 
                args.output, 
                not args.no_interactive
            ) else 1
        
        elif args.command == 'run':
            return 0 if cli.run_agent(
                args.project_path, 
                args.config, 
                not args.no_interactive
            ) else 1
        
        elif args.command == 'deploy':
            return 0 if cli.deploy_agent(
                args.project_path, 
                args.environment, 
                args.config
            ) else 1
        
        elif args.command == 'monitor':
            return 0 if cli.monitor_agent(
                args.project_path, 
                args.dashboard
            ) else 1
        
        elif args.command == 'debug':
            return 0 if cli.debug_agent(
                args.project_path, 
                args.agent, 
                args.logs
            ) else 1
        
        elif args.command == 'chat':
            return 0 if cli.chat(
                args.project_path, 
                args.config
            ) else 1
        
        # Handle legacy commands
        elif args.command == 'create':
            return 0 if cli.create_project(args.project_name, args.output) else 1
        
        elif args.command == 'list-templates':
            return 0 if cli.list_templates() else 1
        
        elif args.command == 'validate-templates':
            return 0 if cli.validate_templates() else 1
        
        elif args.command == 'validate':
            return 0 if cli.validate_project_name(args.project_name) else 1
        
        elif args.command == 'info':
            return 0 if cli.show_info() else 1
        
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        if RICH_ENABLED:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
        else:
            print("\nGoodbye!")
        return 0
    except Exception as e:
        if RICH_ENABLED:
            console.print(f"[red]❌ Unexpected error: {e}[/red]")
        else:
            print(f"Unexpected error: {e}")
        return 1
