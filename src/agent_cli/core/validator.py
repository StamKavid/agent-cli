"""
Clean validators - simple and focused.
"""

import re
import keyword
from pathlib import Path
from typing import Optional


class NameValidator:
    """Validate and suggest fixes for Python project names."""
    
    def is_valid(self, name: str) -> bool:
        """Check if name is a valid project name (more permissive)."""
        if not name:
            return False
        
        # Allow hyphens and underscores - we'll sanitize for Python later
        if re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
            return True
        
        return False
    
    def suggest_fix(self, name: str) -> Optional[str]:
        """Suggest a fixed version of the name."""
        if not name:
            return None
        
        # Start with the original name
        fixed = name.strip()
        
        # Convert to lowercase
        fixed = fixed.lower()
        
        # Replace invalid characters with underscores
        fixed = re.sub(r'[^a-z0-9_]', '_', fixed)
        
        # Remove multiple consecutive underscores
        fixed = re.sub(r'_+', '_', fixed)
        
        # Remove leading/trailing underscores
        fixed = fixed.strip('_')
        
        # Ensure it doesn't start with a number
        if fixed and fixed[0].isdigit():
            fixed = f"project_{fixed}"
        
        # Handle Python keywords
        if keyword.iskeyword(fixed):
            fixed = f"{fixed}_project"
        
        # Must not be empty
        if not fixed:
            return "my_agent"
        
        # Validate the fix
        if self.is_valid(fixed):
            return fixed
        
        # If still invalid, return a safe default
        return "my_agent"


class PathValidator:
    """Validate file and directory paths."""
    
    def is_valid(self, path_str: str) -> bool:
        """Check if path is valid and writable."""
        try:
            path = Path(path_str).resolve()
            
            # Path must exist or be creatable
            if path.exists():
                return path.is_dir() and self._is_writable(path)
            else:
                # Check if parent exists and is writable
                parent = path.parent
                return parent.exists() and parent.is_dir() and self._is_writable(parent)
        
        except (OSError, ValueError):
            return False
    
    def _is_writable(self, path: Path) -> bool:
        """Check if path is writable."""
        try:
            # Try creating a temporary file
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False
