"""
Unit tests for the AgentProjectCLI and related components.

These tests follow testing best practices and provide comprehensive coverage.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from agent_cli.cli import AgentProjectCLI
from agent_cli.exceptions import (
    ProjectExistsError,
    TemplateError,
    DirectoryCreationError,
    FileCreationError
)
from agent_cli.generators.project_generator import (
    DefaultProjectValidator,
    DefaultDirectoryCreator,
    DefaultFileGenerator,
    AgentProjectCreator
)
from agent_cli.templates.static_templates import StaticTemplateProvider
from agent_cli.templates.template_validator import TemplateValidator, validate_template_collection


class TestDefaultProjectValidator:
    """Test cases for DefaultProjectValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DefaultProjectValidator()
    
    def test_validate_project_name_valid(self):
        """Test validation of valid project names."""
        valid_names = [
            "my_agent", 
            "agent123",
            "simple_ai_agent",
            "a",
            "test_project_name"
        ]
        
        for name in valid_names:
            assert self.validator.validate_project_name(name), f"'{name}' should be valid"
    
    def test_validate_project_name_invalid(self):
        """Test validation of invalid project names."""
        invalid_names = [
            "My-Agent",      # hyphens not allowed
            "123agent",      # can't start with number
            "my agent",      # spaces not allowed
            "my-agent-project",  # hyphens not allowed
            "Agent",         # uppercase not allowed
            "",              # empty string
            "a" * 51,        # too long
            "_agent",        # can't start with underscore
            "agent_",        # can't end with underscore for single char
            "class",         # Python reserved word
            "import",        # Python reserved word
            "def",           # Python reserved word
        ]
        
        for name in invalid_names:
            assert not self.validator.validate_project_name(name), f"'{name}' should be invalid"
    
    def test_validate_output_directory_valid(self):
        """Test validation of valid output directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert self.validator.validate_output_directory(temp_dir)
    
    def test_validate_output_directory_nonexistent(self):
        """Test validation of non-existent directory."""
        nonexistent_dir = "/path/that/does/not/exist"
        assert not self.validator.validate_output_directory(nonexistent_dir)
    
    def test_validate_output_directory_file(self):
        """Test validation when path is a file, not directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            assert not self.validator.validate_output_directory(temp_file.name)


class TestDefaultDirectoryCreator:
    """Test cases for DefaultDirectoryCreator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.directory_creator = DefaultDirectoryCreator()
    
    def test_create_directory_structure_success(self):
        """Test successful directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            directories = [
                "src/{project_name}/core",
                "src/{project_name}/application", 
                "tests",
                "tools"
            ]
            project_name = "test_project"
            
            self.directory_creator.create_directory_structure(
                base_path, directories, project_name
            )
            
            # Check that directories were created
            assert (base_path / "src/test_project/core").exists()
            assert (base_path / "src/test_project/application").exists()
            assert (base_path / "tests").exists()
            assert (base_path / "tools").exists()
            
            # Check that __init__.py files were created in Python packages
            assert (base_path / "src/test_project/__init__.py").exists()
            assert (base_path / "src/test_project/core/__init__.py").exists()
            assert (base_path / "src/test_project/application/__init__.py").exists()
    
    def test_create_directory_structure_failure(self):
        """Test directory creation failure."""
        # Try to create directory in read-only location (simulated)
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = OSError("Permission denied")
            
            base_path = Path("/tmp")
            directories = ["test_dir"]
            project_name = "test_project"
            
            with pytest.raises(DirectoryCreationError):
                self.directory_creator.create_directory_structure(
                    base_path, directories, project_name
                )


class TestStaticTemplateProvider:
    """Test cases for StaticTemplateProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.template_provider = StaticTemplateProvider()
    
    def test_get_template_success(self):
        """Test successful template retrieval."""
        template_content = self.template_provider.get_template("gitignore")
        assert isinstance(template_content, str)
        assert len(template_content) > 0
        assert "# Byte-compiled" in template_content
    
    def test_get_template_not_found(self):
        """Test template not found error."""
        with pytest.raises(TemplateError) as exc_info:
            self.template_provider.get_template("nonexistent_template")
        
        assert "nonexistent_template" in str(exc_info.value)
    
    def test_get_available_templates(self):
        """Test getting list of available templates."""
        templates = self.template_provider.get_available_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "gitignore" in templates
        assert "env" in templates
    
    def test_get_template_with_context(self):
        """Test template retrieval with context."""
        context = {"project_name": "test_project"}
        template_content = self.template_provider.get_template("gitignore", context)
        assert isinstance(template_content, str)
        assert len(template_content) > 0


class TestDefaultFileGenerator:
    """Test cases for DefaultFileGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.template_provider = StaticTemplateProvider()
        self.file_generator = DefaultFileGenerator(self.template_provider)
    
    def test_generate_file_success(self):
        """Test successful file generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_file.txt"
            context = {"project_name": "test_project"}
            
            # Create a simple template for testing
            with patch.object(self.template_provider, 'get_template') as mock_get:
                mock_get.return_value = "Project: {project_name}"
                
                self.file_generator.generate_file(
                    file_path, "test_template", context
                )
                
                assert file_path.exists()
                content = file_path.read_text()
                assert content == "Project: test_project"
    
    def test_generate_file_missing_context(self):
        """Test file generation with missing context variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_file.txt"
            context = {}  # Missing required variables
            
            with patch.object(self.template_provider, 'get_template') as mock_get:
                mock_get.return_value = "Project: {project_name}"
                
                with pytest.raises(TemplateError):
                    self.file_generator.generate_file(
                        file_path, "test_template", context
                    )
    
    def test_generate_file_write_error(self):
        """Test file generation write error."""
        # Try to write to a read-only location
        file_path = Path("/invalid/path/test_file.txt")
        context = {"project_name": "test_project"}
        
        with patch.object(self.template_provider, 'get_template') as mock_get:
            mock_get.return_value = "Project: {project_name}"
            
            with pytest.raises(FileCreationError):
                self.file_generator.generate_file(
                    file_path, "test_template", context
                )


class TestAgentProjectCreator:
    """Test cases for AgentProjectCreator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.template_provider = StaticTemplateProvider()
        self.validator = DefaultProjectValidator()
        self.directory_creator = DefaultDirectoryCreator()
        self.file_generator = DefaultFileGenerator(self.template_provider)
        
        self.project_creator = AgentProjectCreator(
            validator=self.validator,
            directory_creator=self.directory_creator,
            file_generator=self.file_generator,
            template_provider=self.template_provider
        )
    
    def test_create_project_success(self):
        """Test successful project creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "test_agent"
            
            success = self.project_creator.create_project(project_name, temp_dir)
            
            assert success
            project_path = Path(temp_dir) / project_name
            assert project_path.exists()
            
            # Check key files exist
            assert (project_path / "pyproject.toml").exists()
            assert (project_path / ".env").exists()
            assert (project_path / "Dockerfile").exists()
    
    def test_create_project_invalid_name(self):
        """Test project creation with invalid name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_name = "Invalid-Name"
            
            success = self.project_creator.create_project(invalid_name, temp_dir)
            
            assert not success
    
    def test_create_project_already_exists(self):
        """Test project creation when project already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "test_agent"
            project_path = Path(temp_dir) / project_name
            project_path.mkdir()  # Create the directory first
            
            success = self.project_creator.create_project(project_name, temp_dir)
            
            assert not success
    
    def test_create_project_invalid_output_dir(self):
        """Test project creation with invalid output directory."""
        invalid_output_dir = "/path/that/does/not/exist"
        project_name = "test_agent"
        
        success = self.project_creator.create_project(project_name, invalid_output_dir)
        
        assert not success


class TestAgentProjectCLI:
    """Test cases for AgentProjectCLI."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cli = AgentProjectCLI()
    
    def test_create_project_integration(self):
        """Integration test for CLI project creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_name = "test_integration"
            
            success = self.cli.create_project(project_name, temp_dir)
            
            assert success
            project_path = Path(temp_dir) / project_name
            assert project_path.exists()
            
            # Verify key project structure
            expected_files = [
                "pyproject.toml",
                ".env",
                "Dockerfile",
                "Makefile",
                f"src/{project_name}/config.py",
                f"src/{project_name}/main.py",
                "notebooks/01_prompt_engineering_playground.ipynb",
                "tools/run_agent.py",
                "tests/test_agent.py"
            ]
            
            for file_path in expected_files:
                full_path = project_path / file_path
                assert full_path.exists(), f"Missing file: {file_path}"
    
    def test_create_project_error_handling(self):
        """Test CLI error handling."""
        # Test with invalid project name
        success = self.cli.create_project("Invalid-Name", ".")
        assert not success
        
        # Test with invalid output directory
        success = self.cli.create_project("valid_name", "/invalid/path")
        assert not success
    
    def test_list_templates(self):
        """Test list templates functionality."""
        success = self.cli.list_templates()
        assert success
    
    def test_validate_project_name_valid(self):
        """Test project name validation with valid name."""
        success = self.cli.validate_project_name("valid_project_name")
        assert success
    
    def test_validate_project_name_invalid(self):
        """Test project name validation with invalid name."""
        success = self.cli.validate_project_name("Invalid-Name")
        assert not success
    
    def test_show_info(self):
        """Test show info functionality."""
        success = self.cli.show_info()
        assert success
    
    def test_validate_templates(self):
        """Test template validation functionality."""
        success = self.cli.validate_templates()
        assert success


class TestTemplateValidator:
    """Test cases for TemplateValidator."""
    
    def test_validate_template_name_valid(self):
        """Test validation of valid template names."""
        valid_names = ["gitignore", "pyproject_toml", "test_template_123"]
        
        for name in valid_names:
            is_valid, errors = TemplateValidator.validate_template_name(name)
            assert is_valid, f"'{name}' should be valid: {errors}"
    
    def test_validate_template_name_invalid(self):
        """Test validation of invalid template names."""
        invalid_names = [
            "",  # empty
            "template with spaces",  # spaces
            "template@name",  # invalid characters
            "a" * 101,  # too long
        ]
        
        for name in invalid_names:
            is_valid, errors = TemplateValidator.validate_template_name(name)
            assert not is_valid, f"'{name}' should be invalid"
    
    def test_validate_template_content_valid(self):
        """Test validation of valid template content."""
        valid_content = "This is a valid template with {project_name} placeholder"
        is_valid, errors = TemplateValidator.validate_template_content("test_template", valid_content)
        assert is_valid, f"Template should be valid: {errors}"
    
    def test_validate_template_content_empty(self):
        """Test validation of empty template content."""
        is_valid, errors = TemplateValidator.validate_template_content("test_template", "")
        assert not is_valid
        assert "empty" in errors[0]
    
    def test_validate_template_content_malformed_placeholder(self):
        """Test validation of template with malformed placeholders."""
        malformed_content = "Template with {malformed placeholder"
        is_valid, errors = TemplateValidator.validate_template_content("test_template", malformed_content)
        assert not is_valid
        assert "Malformed placeholders" in errors[0]
    
    def test_validate_template_context_valid(self):
        """Test validation of valid template context."""
        context = {"project_name": "test_project"}
        is_valid, errors = TemplateValidator.validate_template_context("pyproject_toml", context)
        assert is_valid, f"Context should be valid: {errors}"
    
    def test_validate_template_context_missing(self):
        """Test validation of template context with missing variables."""
        context = {}  # Missing project_name
        is_valid, errors = TemplateValidator.validate_template_context("pyproject_toml", context)
        assert not is_valid
        assert "Missing required context variable" in errors[0]


class TestTemplateValidationIntegration:
    """Integration tests for template validation."""
    
    def test_validate_template_collection(self):
        """Test validation of a collection of templates."""
        templates = {
            "valid_template": "This is a valid template with {project_name}",
            "empty_template": "",
            "malformed_template": "Template with {malformed placeholder",
        }
        
        results = validate_template_collection(templates)
        
        # Check that validation results are returned
        assert "valid_template" in results
        assert "empty_template" in results
        assert "malformed_template" in results
        
        # Check that valid template has no errors
        assert len(results["valid_template"]) == 0
        
        # Check that invalid templates have errors
        assert len(results["empty_template"]) > 0
        assert len(results["malformed_template"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
