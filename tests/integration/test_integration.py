"""
Integration tests for the Agent CLI.

These tests verify end-to-end functionality and integration between components.
"""

import subprocess
import tempfile
import sys
from pathlib import Path
import pytest


class TestCLIIntegration:
    """Integration tests for the CLI."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_cli", "--help"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Agent Project CLI" in result.stdout
        assert "create" in result.stdout
    
    def test_cli_create_command_help(self):
        """Test CLI create command help."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_cli", "create", "--help"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Create a new agent project" in result.stdout
        assert "project_name" in result.stdout
    
    def test_cli_create_project_integration(self):
        """Test full CLI project creation integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "integration_test_project"
            
            result = subprocess.run([
                sys.executable, "-m", "agent_cli", 
                "create", project_name,
                "--output", temp_dir,
                "--verbose"
            ], 
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
            )
            
            assert result.returncode == 0
            assert "Successfully created" in result.stdout
            
            # Verify project structure
            project_path = Path(temp_dir) / project_name
            assert project_path.exists()
            
            # Check key files
            key_files = [
                "pyproject.toml",
                "README.md",
                ".env",
                "Dockerfile",
                "Makefile",
                f"src/{project_name}/config.py",
                f"src/{project_name}/main.py"
            ]
            
            for file_path in key_files:
                full_path = project_path / file_path
                assert full_path.exists(), f"Missing: {file_path}"
    
    def test_cli_invalid_project_name(self):
        """Test CLI with invalid project name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run([
                sys.executable, "-m", "agent_cli",
                "create", "Invalid-Name",
                "--output", temp_dir
            ],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
            )
            
            assert result.returncode == 1
    
    def test_cli_project_already_exists(self):
        """Test CLI when project already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "existing_project"
            
            # Create project first time
            result1 = subprocess.run([
                sys.executable, "-m", "agent_cli",
                "create", project_name,
                "--output", temp_dir
            ],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
            )
            
            assert result1.returncode == 0
            
            # Try to create again
            result2 = subprocess.run([
                sys.executable, "-m", "agent_cli",
                "create", project_name,
                "--output", temp_dir
            ],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
            )
            
            assert result2.returncode == 1


class TestGeneratedProjectStructure:
    """Test the structure of generated projects."""
    
    @pytest.fixture
    def generated_project(self):
        """Create a test project for structure validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "structure_test"
            
            subprocess.run([
                sys.executable, "-m", "agent_cli",
                "create", project_name,
                "--output", temp_dir
            ],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True
            )
            
            yield Path(temp_dir) / project_name
    
    def test_project_directories(self, generated_project):
        """Test that all required directories are created."""
        required_dirs = [
            ".github/workflows",
            "static",
            "data", 
            "k8s",
            "notebooks",
            f"src/{generated_project.name}/application/conversation_service",
            f"src/{generated_project.name}/core/agent",
            f"src/{generated_project.name}/infrastructure/vector_database",
            "tools",
            "tests"
        ]
        
        for dir_path in required_dirs:
            full_path = generated_project / dir_path
            assert full_path.exists(), f"Directory missing: {dir_path}"
            assert full_path.is_dir(), f"Not a directory: {dir_path}"
    
    def test_project_files(self, generated_project):
        """Test that all required files are created."""
        required_files = [
            "pyproject.toml",
            "README.md",
            ".env",
            "Dockerfile", 
            "Makefile",
            ".gitignore",
            "python-version",
            "ISSUE_TEMPLATE.md"
        ]
        
        for file_path in required_files:
            full_path = generated_project / file_path
            assert full_path.exists(), f"File missing: {file_path}"
            assert full_path.is_file(), f"Not a file: {file_path}"
    
    def test_notebook_files(self, generated_project):
        """Test that notebook files are created."""
        notebook_files = [
            "01_prompt_engineering_playground.ipynb",
            "02_short_term_memory.ipynb", 
            "03_long_term_memory.ipynb",
            "04_tool_calling_playground.ipynb"
        ]
        
        for notebook in notebook_files:
            notebook_path = generated_project / "notebooks" / notebook
            assert notebook_path.exists(), f"Notebook missing: {notebook}"
            
            # Verify it's valid JSON (basic check)
            content = notebook_path.read_text()
            assert '"cells"' in content
            assert '"metadata"' in content
    
    def test_tool_scripts(self, generated_project):
        """Test that tool scripts are created."""
        tool_scripts = [
            "run_agent.py",
            "populate_long_term_memory.py",
            "delete_long_term_memory.py",
            "evaluate_agent.py"
        ]
        
        for script in tool_scripts:
            script_path = generated_project / "tools" / script
            assert script_path.exists(), f"Tool script missing: {script}"
            
            # Verify it's executable Python
            content = script_path.read_text()
            assert content.startswith("#!/usr/bin/env python3")
            assert "def main(" in content
    
    def test_source_structure(self, generated_project):
        """Test source code structure."""
        project_name = generated_project.name
        src_path = generated_project / "src" / project_name
        
        # Check main source files
        assert (src_path / "__init__.py").exists()
        assert (src_path / "config.py").exists()
        assert (src_path / "main.py").exists()
        
        # Check package structure
        packages = [
            "application",
            "core", 
            "infrastructure"
        ]
        
        for package in packages:
            package_path = src_path / package
            assert package_path.exists()
            assert (package_path / "__init__.py").exists()
    
    def test_pyproject_toml_content(self, generated_project):
        """Test pyproject.toml content."""
        pyproject_path = generated_project / "pyproject.toml"
        content = pyproject_path.read_text()
        
        project_name = generated_project.name
        
        # Check key sections
        assert f'name = "{project_name}"' in content
        assert "[build-system]" in content
        assert "[project]" in content
        assert "[tool.black]" in content
        assert "[tool.isort]" in content
        assert "[tool.mypy]" in content
        assert "[tool.pytest.ini_options]" in content
    
    def test_readme_content(self, generated_project):
        """Test README.md content."""
        readme_path = generated_project / "README.md"
        content = readme_path.read_text()
        
        project_name = generated_project.name
        
        # Check project name is included
        assert project_name in content
        assert "# " + project_name in content
        
        # Check key sections
        assert "## 🚀 Features" in content
        assert "## 🛠️ Installation" in content
        assert "## 🚀 Quick Start" in content


if __name__ == "__main__":
    pytest.main([__file__])
