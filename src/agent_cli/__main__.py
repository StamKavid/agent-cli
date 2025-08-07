"""
Main entry point for the agent_cli package.

This allows the package to be run as a module: python -m agent_cli
Uses the zero-config CLI implementation for seamless UX.
"""

from .cli_zero_config import main

if __name__ == "__main__":
    main()
