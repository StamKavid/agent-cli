"""
Template validation utilities.

This module provides validation functions for ensuring templates
are properly formatted and contain required content.
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..exceptions import TemplateError


class TemplateValidator:
    """Validator for template content and structure."""
    
    @staticmethod
    def validate_template_content(template_name: str, content: str) -> Tuple[bool, List[str]]:
        """Validate template content.
        
        Args:
            template_name: Name of the template
            content: Template content to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if content is empty
        if not content or not content.strip():
            errors.append("Template content is empty")
            return False, errors
        
        # Check for required placeholders based on template type
        required_placeholders = TemplateValidator._get_required_placeholders(template_name)
        for placeholder in required_placeholders:
            if placeholder not in content:
                errors.append(f"Missing required placeholder: {placeholder}")
        
        # Check for malformed placeholders
        malformed_placeholders = TemplateValidator._find_malformed_placeholders(content)
        if malformed_placeholders:
            errors.append(f"Malformed placeholders found: {', '.join(malformed_placeholders)}")
        
        # Template-specific validations
        template_errors = TemplateValidator._validate_template_specific(template_name, content)
        errors.extend(template_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_template_name(template_name: str) -> Tuple[bool, List[str]]:
        """Validate template name format.
        
        Args:
            template_name: Name of the template to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if name is empty
        if not template_name or not template_name.strip():
            errors.append("Template name cannot be empty")
            return False, errors
        
        # Check for invalid characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', template_name):
            errors.append("Template name contains invalid characters (use only letters, numbers, underscores, and hyphens)")
        
        # Check length
        if len(template_name) > 100:
            errors.append("Template name too long (max 100 characters)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_template_context(template_name: str, context: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate that context contains required variables for template.
        
        Args:
            template_name: Name of the template
            context: Context variables to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Get required context variables for this template
        required_vars = TemplateValidator._get_required_context_variables(template_name)
        
        for var in required_vars:
            if var not in context:
                errors.append(f"Missing required context variable: {var}")
            elif not context[var]:
                errors.append(f"Context variable '{var}' is empty")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _get_required_placeholders(template_name: str) -> List[str]:
        """Get required placeholders for a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            List of required placeholder names
        """
        # Basic templates that always need project_name
        basic_templates = {
            "pyproject_toml", "readme", "src_config_py", "src_main_py"
        }
        
        if template_name in basic_templates:
            return ["{project_name}"]
        
        return []
    
    @staticmethod
    def _get_required_context_variables(template_name: str) -> List[str]:
        """Get required context variables for a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            List of required context variable names
        """
        # Templates that require project_name
        project_name_templates = {
            "pyproject_toml", "readme", "src_config_py", "src_main_py"
        }
        
        if template_name in project_name_templates:
            return ["project_name"]
        
        return []
    
    @staticmethod
    def _find_malformed_placeholders(content: str) -> List[str]:
        """Find malformed placeholders in content.
        
        Args:
            content: Content to check
            
        Returns:
            List of malformed placeholder strings
        """
        # Find all potential placeholders
        placeholder_pattern = r'\{[^}]*\}'
        placeholders = re.findall(placeholder_pattern, content)
        
        malformed = []
        for placeholder in placeholders:
            # Check for common malformed patterns
            if placeholder.count('{') != 1 or placeholder.count('}') != 1:
                malformed.append(placeholder)
            elif placeholder == '{}':
                malformed.append(placeholder)
            elif '{' in placeholder[1:] or '}' in placeholder[:-1]:
                malformed.append(placeholder)
        
        return malformed
    
    @staticmethod
    def _validate_template_specific(template_name: str, content: str) -> List[str]:
        """Perform template-specific validations.
        
        Args:
            template_name: Name of the template
            content: Template content
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate Python files
        if template_name.endswith('.py') or template_name in ['src_config_py', 'src_main_py']:
            python_errors = TemplateValidator._validate_python_template(content)
            errors.extend(python_errors)
        
        # Validate JSON files (notebooks)
        elif template_name.endswith('.ipynb') or template_name.startswith('notebook_'):
            json_errors = TemplateValidator._validate_json_template(content)
            errors.extend(json_errors)
        
        # Validate TOML files
        elif template_name == 'pyproject_toml':
            toml_errors = TemplateValidator._validate_toml_template(content)
            errors.extend(toml_errors)
        
        return errors
    
    @staticmethod
    def _validate_python_template(content: str) -> List[str]:
        """Validate Python template content.
        
        Args:
            content: Python template content
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for basic Python syntax indicators
        if '"""' in content and not content.count('"""') % 2 == 0:
            errors.append("Unmatched triple quotes in Python template")
        
        if "import" in content and "from" in content:
            # Check if imports are at the top
            lines = content.split('\n')
            import_lines = [i for i, line in enumerate(lines) if line.strip().startswith(('import ', 'from '))]
            if import_lines and min(import_lines) > 10:  # Imports should be near the top
                errors.append("Import statements should be at the top of the file")
        
        return errors
    
    @staticmethod
    def _validate_json_template(content: str) -> List[str]:
        """Validate JSON template content.
        
        Args:
            content: JSON template content
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Basic JSON structure check
        if not content.strip().startswith('{'):
            errors.append("JSON template should start with '{'")
        
        if not content.strip().endswith('}'):
            errors.append("JSON template should end with '}'")
        
        # Check for balanced braces
        brace_count = content.count('{') - content.count('}')
        if brace_count != 0:
            errors.append(f"Unmatched braces in JSON template (difference: {brace_count})")
        
        return errors
    
    @staticmethod
    def _validate_toml_template(content: str) -> List[str]:
        """Validate TOML template content.
        
        Args:
            content: TOML template content
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for required TOML sections
        required_sections = ['[build-system]', '[project]']
        for section in required_sections:
            if section not in content:
                errors.append(f"Missing required TOML section: {section}")
        
        # Check for project name placeholder
        if '{project_name}' not in content:
            errors.append("TOML template missing {project_name} placeholder")
        
        return errors


def validate_template_collection(templates: Dict[str, str]) -> Dict[str, List[str]]:
    """Validate a collection of templates.
    
    Args:
        templates: Dictionary of template names to content
        
    Returns:
        Dictionary mapping template names to lists of validation errors
    """
    validation_results = {}
    
    for template_name, content in templates.items():
        # Validate template name
        name_valid, name_errors = TemplateValidator.validate_template_name(template_name)
        
        # Validate template content
        content_valid, content_errors = TemplateValidator.validate_template_content(template_name, content)
        
        # Combine all errors
        all_errors = name_errors + content_errors
        validation_results[template_name] = all_errors
    
    return validation_results


def get_template_statistics(templates: Dict[str, str]) -> Dict[str, int]:
    """Get statistics about template collection.
    
    Args:
        templates: Dictionary of template names to content
        
    Returns:
        Dictionary with template statistics
    """
    stats = {
        'total_templates': len(templates),
        'total_size': sum(len(content) for content in templates.values()),
        'python_templates': len([name for name in templates.keys() if name.endswith('.py') or name in ['src_config_py', 'src_main_py']]),
        'notebook_templates': len([name for name in templates.keys() if name.endswith('.ipynb') or name.startswith('notebook_')]),
        'file_templates': len([name for name in templates.keys() if not any(name.startswith(prefix) for prefix in ['src_', 'notebook_', 'tool_', 'test_'])]),
    }
    
    return stats 