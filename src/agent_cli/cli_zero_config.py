"""
Zero-configuration CLI inspired by Claude Code.
One command, instant results, seamless UX.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# Rich imports with graceful fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    console = Console()
    RICH_ENABLED = True
except ImportError:
    console = None
    RICH_ENABLED = False

from .core.creator import ProjectCreator
from .core.templates import TemplateManager


class ZeroConfigCLI:
    """Zero-configuration CLI - just works out of the box."""
    
    def __init__(self):
        self.creator = ProjectCreator()
        self.template_manager = TemplateManager()
    
    def run(self, project_name: Optional[str] = None) -> bool:
        """Main entry point - zero config, just works."""
        try:
            # Welcome & smart project naming
            if not project_name:
                project_name = self._smart_project_name()
            
            # Smart defaults - no questions unless needed
            config = self._get_smart_defaults(project_name)
            
            # Instant creation with beautiful feedback
            return self._instant_create(project_name, config)
            
        except KeyboardInterrupt:
            self._print("👋 Cancelled", style="yellow")
            return False
        except Exception as e:
            self._print(f"❌ {e}", style="red")
            return False
    
    def _smart_project_name(self) -> str:
        """Smart project naming with current directory context."""
        # Try to use current directory name
        current_dir = Path.cwd().name
        
        # Check if current directory looks like a project
        if self._is_empty_or_suitable_directory():
            suggestion = self._sanitize_name(current_dir)
            
            if RICH_ENABLED:
                welcome = Panel.fit(
                    "[bold blue]🤖 Agent CLI[/bold blue]\n"
                    "Scaffold AI agent projects instantly - zero configuration needed!",
                    border_style="blue"
                )
                console.print(welcome)
                
                # Smart suggestion
                if suggestion and suggestion != "downloads":  # Don't suggest "downloads"
                    self._print(f"\n💡 Detected project context: [bold cyan]{suggestion}[/bold cyan]")
                    name = Prompt.ask("Agent project name", default=suggestion)
                else:
                    name = Prompt.ask("Agent project name", default="my-agent-project")
            else:
                print("🤖 Agent CLI - Scaffold AI agent projects instantly!")
                name = input(f"Agent project name [{suggestion or 'my-agent-project'}]: ").strip()
                if not name:
                    name = suggestion or "my-agent-project"
        else:
            # Directory not suitable, ask for name
            if RICH_ENABLED:
                welcome = Panel.fit(
                    "[bold blue]🤖 Agent CLI[/bold blue]\n"
                    "Scaffold AI agent projects instantly - zero configuration needed!",
                    border_style="blue"
                )
                console.print(welcome)
                name = Prompt.ask("Agent project name", default="my-agent-project")
            else:
                print("🤖 Agent CLI - Scaffold AI agent projects instantly!")
                name = input("Agent project name [my-agent-project]: ").strip()
                if not name:
                    name = "my-agent-project"
        
        return self._sanitize_name(name)
    
    def _get_smart_defaults(self, project_name: str) -> dict:
        """Smart defaults based on project name and context."""
        # Analyze project name for hints
        name_lower = project_name.lower()
        
        # Smart template detection
        if any(word in name_lower for word in ["chat", "bot", "conversation", "talk"]):
            template = "conversational"
        elif any(word in name_lower for word in ["research", "search", "analyze", "study"]):
            template = "research"
        elif any(word in name_lower for word in ["auto", "task", "workflow", "process"]):
            template = "automation"
        else:
            template = "basic"  # Safe default
        
        return {
            "template": template,
            "output": ".",  # Current directory
            "with_api": True,  # Always include API
            "with_memory": True,  # Always include memory
            "llm_provider": "openai"  # Most common
        }
    
    def _instant_create(self, name: str, config: dict) -> bool:
        """Create project instantly with beautiful progress."""
        if RICH_ENABLED:
            # Show what we're creating
            info = Panel.fit(
                f"[bold green]Creating:[/bold green] {name}\n"
                f"[bold]Template:[/bold] {config['template']}\n"
                f"[bold]Features:[/bold] API, Memory, Tools\n"
                f"[bold]Provider:[/bold] {config['llm_provider']}",
                title="🚀 Smart Defaults",
                border_style="green"
            )
            console.print(info)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                # Multi-stage progress
                task = progress.add_task("Setting up project...", total=4)
                
                # Stage 1: Structure
                progress.update(task, description="📁 Creating structure...")
                success = self.creator.create(name, config["template"], config["output"])
                if not success:
                    return False
                progress.advance(task)
                
                # Stage 2: Dependencies
                progress.update(task, description="📦 Setting up dependencies...")
                self._setup_dependencies(name)
                progress.advance(task)
                
                # Stage 3: Environment
                progress.update(task, description="⚙️ Configuring environment...")
                self._setup_environment(name)
                progress.advance(task)
                
                # Stage 4: Ready
                progress.update(task, description="✨ Finalizing...")
                progress.advance(task)
                
                # Success
                self._show_instant_success(name, Path(config["output"]) / name)
                return True
        else:
            print(f"Creating {name} with smart defaults...")
            success = self.creator.create(name, config["template"], config["output"])
            if success:
                print(f"✅ Created: {name}")
                print("Next: cd {name} && pip install -e .")
            return success
    
    def _setup_dependencies(self, name: str) -> None:
        """Setup basic dependencies file."""
        project_path = Path(name)
        if project_path.exists():
            # Create a basic requirements.txt if it doesn't exist
            req_file = project_path / "requirements.txt"
            if not req_file.exists():
                req_file.write_text(
                    "openai>=1.0.0\n"
                    "python-dotenv>=1.0.0\n"
                    "rich>=13.0.0\n"
                    "fastapi>=0.100.0\n"
                    "uvicorn>=0.20.0\n"
                )
    
    def _setup_environment(self, name: str) -> None:
        """Setup environment file."""
        project_path = Path(name)
        if project_path.exists():
            env_file = project_path / ".env.example"
            if not env_file.exists():
                env_file.write_text(
                    "OPENAI_API_KEY=your_openai_api_key_here\n"
                    "LOG_LEVEL=INFO\n"
                    "# Add your API keys here\n"
                )
    
    def _show_instant_success(self, name: str, path: Path) -> None:
        """Show success with immediate next steps."""
        if RICH_ENABLED:
            # Get started panel
            next_steps = Text()
            next_steps.append("cd ", style="dim")
            next_steps.append(name, style="bold cyan")
            next_steps.append("\npip install -e .", style="dim")
            next_steps.append("\ncp .env.example .env", style="dim")
            next_steps.append("\n# Add your API keys to .env", style="green")
            next_steps.append("\npython main.py", style="bold yellow")
            
            success = Panel.fit(
                f"[bold green]🎉 Agent project scaffolded![/bold green]\n\n"
                f"Your agent project structure is ready with:\n"
                f"✅ Smart template selection\n"
                f"✅ API interface\n"
                f"✅ Memory system\n"
                f"✅ Environment setup\n\n"
                f"[bold cyan]Get started:[/bold cyan]\n{next_steps}",
                title=f"🤖 {name}",
                border_style="green"
            )
            console.print(success)
        else:
            print(f"\n🎉 {name} project scaffolded!")
            print("\nGet started:")
            print(f"cd {name}")
            print("pip install -e .")
            print("cp .env.example .env")
            print("# Add your API keys to .env")
            print("python main.py")
    
    def _is_empty_or_suitable_directory(self) -> bool:
        """Check if current directory is suitable for project creation."""
        current_files = list(Path.cwd().iterdir())
        
        # Empty directory
        if not current_files:
            return True
        
        # Only hidden files/dirs
        if all(f.name.startswith('.') for f in current_files):
            return True
        
        # Only common non-conflicting files
        safe_files = {'.gitignore', 'README.md', 'LICENSE', '.git'}
        if all(f.name in safe_files for f in current_files):
            return True
        
        return False
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name to be valid for both directory and Python package."""
        import re
        # Keep original for directory name, create Python-safe version for package
        original_name = name.strip()
        
        # For directory: allow hyphens, just clean up
        dir_name = re.sub(r'[^a-zA-Z0-9_-]', '-', original_name.lower())
        dir_name = re.sub(r'-+', '-', dir_name)
        dir_name = dir_name.strip('-')
        
        if not dir_name or dir_name.isdigit():
            dir_name = "my-agent-project"
        
        return dir_name
    
    def _print(self, message: str, style: str = "white") -> None:
        """Print with optional styling."""
        if RICH_ENABLED:
            console.print(message, style=style)
        else:
            print(message)


def main():
    """Main entry point for zero-config CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🤖 Agent CLI - Scaffold AI agent projects instantly",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "name", 
        nargs="?", 
        help="Agent project name (optional - will be prompted if not provided)"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="Agent CLI 0.3.0"
    )
    
    args = parser.parse_args()
    
    cli = ZeroConfigCLI()
    success = cli.run(args.name)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
