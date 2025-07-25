#!/usr/bin/env python3
"""
Agent Project CLI

This module provides a simple interface to the full CLI functionality.
For production use, prefer importing from src.agent_cli package.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from agent_cli.cli import main
except ImportError as e:
    print(f"Error importing CLI module: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
