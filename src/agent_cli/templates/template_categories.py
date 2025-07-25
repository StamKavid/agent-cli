"""
Template categories for organizing project templates.

This module provides organized template collections for different
types of project files and structures.
"""

from typing import Dict, List


class FileTemplates:
    """Templates for basic project files."""
    
    @staticmethod
    def get_gitignore() -> str:
        """Get .gitignore template."""
        return """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/

# macOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini

# Project specific
data/
logs/
*.log
.env.local
.env.production
"""

    @staticmethod
    def get_pyproject_toml(project_name: str) -> str:
        """Get pyproject.toml template."""
        return f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "AI Agent Project"
authors = [
    {{name = "Your Name", email = "your.email@example.com"}}
]
readme = "README.md"
license = {{text = "MIT"}}
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "openai>=1.0.0",
    "anthropic>=0.8.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
    "chromadb>=0.4.0",
    "pinecone-client>=3.0.0",
    "redis>=5.0.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "requests>=2.31.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "jupyter>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "flake8>=6.0.0",
    "pre-commit>=3.4.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/{project_name}"
Repository = "https://github.com/yourusername/{project_name}"
Issues = "https://github.com/yourusername/{project_name}/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312', 'py313']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src/{project_name} --cov-report=html --cov-report=term-missing"
"""

    @staticmethod
    def get_readme(project_name: str) -> str:
        """Get README.md template."""
        return f"""# {project_name}

AI Agent project built with modern Python practices.

## Features

- Clean architecture design
- Production-ready setup
- Comprehensive testing
- Docker and Kubernetes support
- Jupyter notebooks for experimentation

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

2. Configure environment:
   ```bash
   cp .env .env.local
   # Edit .env.local with your API keys
   ```

3. Run the agent:
   ```bash
   python tools/run_agent.py
   ```

## Development

- Run tests: `pytest`
- Format code: `black src/`
- Type check: `mypy src/`
- Lint: `flake8 src/`

## License

MIT License
"""

    @staticmethod
    def get_env() -> str:
        """Get .env template."""
        return """# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector Database
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
"""

    @staticmethod
    def get_dockerfile() -> str:
        """Get Dockerfile template."""
        return """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .[dev]

# Copy source code
COPY src/ ./src/
COPY tools/ ./tools/
COPY notebooks/ ./notebooks/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "tools/run_agent.py"]
"""

    @staticmethod
    def get_makefile() -> str:
        """Get Makefile template."""
        return """# AI Agent Project Makefile

.PHONY: help install test lint clean run

help:
	@echo "Available commands:"
	@echo "  install - Install dependencies"
	@echo "  test    - Run tests"
	@echo "  lint    - Run linting"
	@echo "  clean   - Clean up"
	@echo "  run     - Run the agent"

install:
	pip install -e .[dev]

test:
	pytest

lint:
	black --check src/
	isort --check-only src/
	mypy src/
	flake8 src/

format:
	black src/
	isort src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/

run:
	python tools/run_agent.py
"""


class NotebookTemplates:
    """Templates for Jupyter notebooks."""
    
    @staticmethod
    def get_prompt_engineering_playground() -> str:
        """Get prompt engineering playground notebook."""
        return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prompt Engineering Playground\\n",
    "\\n",
    "Experiment with different prompts and see how they affect agent behavior."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\\n",
    "from dotenv import load_dotenv\\n",
    "\\n",
    "load_dotenv()\\n",
    "\\n",
    "# Your prompt engineering experiments here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""

    @staticmethod
    def get_short_term_memory() -> str:
        """Get short term memory notebook."""
        return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Short Term Memory Experiments\\n",
    "\\n",
    "Explore how the agent maintains context within a conversation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Short term memory experiments here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""

    @staticmethod
    def get_long_term_memory() -> str:
        """Get long term memory notebook."""
        return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Long Term Memory Experiments\\n",
    "\\n",
    "Explore how the agent stores and retrieves information over time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Long term memory experiments here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""

    @staticmethod
    def get_tool_calling_playground() -> str:
        """Get tool calling playground notebook."""
        return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tool Calling Playground\\n",
    "\\n",
    "Experiment with different tools and see how the agent uses them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tool calling experiments here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""


class ToolTemplates:
    """Templates for tool scripts."""
    
    @staticmethod
    def get_run_agent() -> str:
        """Get run_agent.py template."""
        return '''"""Run the AI agent."""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the agent."""
    logger.info("Starting AI agent...")
    
    # Your agent initialization and running logic here
    logger.info("Agent started successfully!")

if __name__ == "__main__":
    asyncio.run(main())
'''

    @staticmethod
    def get_populate_long_term_memory() -> str:
        """Get populate_long_term_memory.py template."""
        return '''"""Populate long term memory with initial data."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def populate_memory():
    """Populate the agent's long term memory."""
    logger.info("Populating long term memory...")
    
    # Your memory population logic here
    logger.info("Long term memory populated successfully!")

if __name__ == "__main__":
    populate_memory()
'''

    @staticmethod
    def get_delete_long_term_memory() -> str:
        """Get delete_long_term_memory.py template."""
        return '''"""Delete long term memory data."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def delete_memory():
    """Delete the agent's long term memory."""
    logger.info("Deleting long term memory...")
    
    # Your memory deletion logic here
    logger.info("Long term memory deleted successfully!")

if __name__ == "__main__":
    delete_memory()
'''

    @staticmethod
    def get_evaluate_agent() -> str:
        """Get evaluate_agent.py template."""
        return '''"""Evaluate agent performance."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def evaluate_agent():
    """Evaluate the agent's performance."""
    logger.info("Evaluating agent performance...")
    
    # Your evaluation logic here
    logger.info("Agent evaluation completed!")

if __name__ == "__main__":
    evaluate_agent()
'''


class TestTemplates:
    """Templates for test files."""
    
    @staticmethod
    def get_test_agent() -> str:
        """Get test_agent.py template."""
        return '''"""Tests for the agent module."""

import pytest
from unittest.mock import Mock, patch

def test_agent_initialization():
    """Test agent initialization."""
    # Your test logic here
    assert True

def test_agent_response():
    """Test agent response generation."""
    # Your test logic here
    assert True

if __name__ == "__main__":
    pytest.main([__file__])
'''

    @staticmethod
    def get_test_memory() -> str:
        """Get test_memory.py template."""
        return '''"""Tests for the memory module."""

import pytest
from unittest.mock import Mock, patch

def test_memory_storage():
    """Test memory storage functionality."""
    # Your test logic here
    assert True

def test_memory_retrieval():
    """Test memory retrieval functionality."""
    # Your test logic here
    assert True

if __name__ == "__main__":
    pytest.main([__file__])
'''

    @staticmethod
    def get_init() -> str:
        """Get __init__.py template for tests."""
        return '''"""Test package initialization."""'''


class SourceTemplates:
    """Templates for source code files."""
    
    @staticmethod
    def get_init() -> str:
        """Get __init__.py template."""
        return '''"""Package initialization."""'''

    @staticmethod
    def get_config_py() -> str:
        """Get config.py template."""
        return '''"""Configuration management for the agent."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Vector Database
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Application
    log_level: str = "INFO"
    environment: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
'''

    @staticmethod
    def get_main_py() -> str:
        """Get main.py template."""
        return '''"""Main application entry point."""

import asyncio
import logging
from pathlib import Path

from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main application function."""
    logger.info("Starting AI Agent application...")
    
    # Your main application logic here
    
    logger.info("Application started successfully!")

if __name__ == "__main__":
    asyncio.run(main())
'''

    @staticmethod
    def get_llm_client_base() -> str:
        """Get base LLM client interface template."""
        return '''"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized LLM response."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def generate_structured(
        self, 
        prompt: str, 
        response_schema: BaseModel,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """Generate structured response."""
        pass
'''

    @staticmethod
    def get_openai_client() -> str:
        """Get OpenAI client implementation template."""
        return '''"""OpenAI LLM client implementation."""

import os
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel

from .llm_client_base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """OpenAI client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.default_model = "gpt-4o"
    
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    async def generate_chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    async def generate_structured(
        self, 
        prompt: str, 
        response_schema: BaseModel,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """Generate structured response using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                **kwargs
            )
            
            import json
            content = response.choices[0].message.content
            data = json.loads(content)
            return response_schema(**data)
        except Exception as e:
            raise Exception(f"OpenAI structured generation error: {e}")
'''

    @staticmethod
    def get_anthropic_client() -> str:
        """Get Anthropic client implementation template."""
        return '''"""Anthropic LLM client implementation."""

import os
from typing import Dict, Any, List, Optional
import anthropic
from pydantic import BaseModel

from .llm_client_base import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    """Anthropic client implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic client."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.default_model = "claude-3-5-sonnet-20241022"
    
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using Anthropic."""
        try:
            response = await self.client.messages.create(
                model=model or self.default_model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                } if response.usage else None,
                metadata={"stop_reason": response.stop_reason}
            )
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    async def generate_chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Anthropic."""
        try:
            response = await self.client.messages.create(
                model=model or self.default_model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=messages,
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                } if response.usage else None,
                metadata={"stop_reason": response.stop_reason}
            )
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    async def generate_structured(
        self, 
        prompt: str, 
        response_schema: BaseModel,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """Generate structured response using Anthropic."""
        try:
            response = await self.client.messages.create(
                model=model or self.default_model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            # Parse the response and validate against schema
            content = response.content[0].text
            # You might need to implement JSON parsing here
            # For now, we'll return a basic implementation
            return response_schema(content=content)
        except Exception as e:
            raise Exception(f"Anthropic structured generation error: {e}")
'''

    @staticmethod
    def get_llm_factory() -> str:
        """Get LLM client factory template."""
        return '''"""LLM client factory for creating different provider instances."""

from typing import Optional, Dict, Type
from .llm_client_base import LLMClient
from .openai.openai_client import OpenAIClient
from .anthropic.anthropic_client import AnthropicClient


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    _providers: Dict[str, Type[LLMClient]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }
    
    @classmethod
    def create_client(
        cls, 
        provider: str, 
        **kwargs
    ) -> LLMClient:
        """Create an LLM client instance."""
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        client_class = cls._providers[provider]
        return client_class(**kwargs)
    
    @classmethod
    def register_provider(
        cls, 
        name: str, 
        client_class: Type[LLMClient]
    ) -> None:
        """Register a new LLM provider."""
        cls._providers[name] = client_class
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers."""
        return list(cls._providers.keys())


# Convenience function
def create_llm_client(provider: str, **kwargs) -> LLMClient:
    """Create an LLM client with the specified provider."""
    return LLMClientFactory.create_client(provider, **kwargs)
'''

class QuickstartTemplates:
    """Templates for quickstart examples."""
    
    @staticmethod
    def get_quickstart_py() -> str:
        """Get quickstart.py template with LangGraph example."""
        return '''"""
LangGraph Quickstart Example

This example demonstrates how to set up and use LangGraph's prebuilt, reusable components
to construct agentic systems quickly and reliably.

Prerequisites:
- An Anthropic API key (set in .env file)
- Install dependencies: pip install -U langgraph "langchain[anthropic]"
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def create_basic_agent():
    """Create a basic LangGraph agent with a simple tool."""
    try:
        from langgraph.prebuilt import create_react_agent
        
        agent = create_react_agent(
            model="anthropic:claude-3-7-sonnet-latest",
            tools=[get_weather],
            prompt="You are a helpful assistant"
        )
        
        print("✅ Basic agent created successfully!")
        return agent
    
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install -U langgraph 'langchain[anthropic]'")
        return None

def create_configured_agent():
    """Create an agent with custom LLM configuration."""
    try:
        from langchain.chat_models import init_chat_model
        from langgraph.prebuilt import create_react_agent
        
        model = init_chat_model(
            "anthropic:claude-3-7-sonnet-latest",
            temperature=0
        )
        
        agent = create_react_agent(
            model=model,
            tools=[get_weather],
        )
        
        print("✅ Configured agent created successfully!")
        return agent
    
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return None

def create_agent_with_memory():
    """Create an agent with memory for multi-turn conversations."""
    try:
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import InMemorySaver
        
        checkpointer = InMemorySaver()
        
        agent = create_react_agent(
            model="anthropic:claude-3-7-sonnet-latest",
            tools=[get_weather],
            checkpointer=checkpointer
        )
        
        print("✅ Agent with memory created successfully!")
        return agent
    
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return None

def create_agent_with_structured_output():
    """Create an agent with structured output using Pydantic models."""
    try:
        from pydantic import BaseModel
        from langgraph.prebuilt import create_react_agent
        
        class WeatherResponse(BaseModel):
            conditions: str
            city: str
            temperature: str
        
        agent = create_react_agent(
            model="anthropic:claude-3-7-sonnet-latest",
            tools=[get_weather],
            response_format=WeatherResponse
        )
        
        print("✅ Agent with structured output created successfully!")
        return agent
    
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return None

def run_basic_example():
    """Run a basic agent example."""
    print("🚀 Running LangGraph Quickstart Example")
    print("=" * 50)
    
    # Create basic agent
    agent = create_basic_agent()
    if not agent:
        return
    
    # Run the agent
    print("\\n📝 Testing basic agent...")
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        )
        print("✅ Agent response received!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error running agent: {e}")

def run_memory_example():
    """Run an agent with memory example."""
    print("\\n🧠 Testing agent with memory...")
    
    agent = create_agent_with_memory()
    if not agent:
        return
    
    try:
        # First conversation
        config = {"configurable": {"thread_id": "1"}}
        sf_response = agent.invoke(
            {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
            config
        )
        print("✅ First response received!")
        
        # Second conversation (should remember context)
        ny_response = agent.invoke(
            {"messages": [{"role": "user", "content": "what about new york?"}]},
            config
        )
        print("✅ Second response received!")
        print("Agent maintained conversation context!")
        
    except Exception as e:
        print(f"❌ Error running memory example: {e}")

def run_structured_output_example():
    """Run an agent with structured output example."""
    print("\\n📊 Testing agent with structured output...")
    
    agent = create_agent_with_structured_output()
    if not agent:
        return
    
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        )
        print("✅ Structured response received!")
        if "structured_response" in response:
            print(f"Structured data: {response['structured_response']}")
        
    except Exception as e:
        print(f"❌ Error running structured output example: {e}")

def main():
    """Main function to run all examples."""
    print("🎯 LangGraph Quickstart Examples")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not found in environment")
        print("Please set your API key in the .env file")
        print("Example: ANTHROPIC_API_KEY=your_api_key_here")
        return
    
    # Run examples
    run_basic_example()
    run_memory_example()
    run_structured_output_example()
    
    print("\\n🎉 All examples completed!")
    print("\\nNext steps:")
    print("1. Explore the notebooks/ directory for more experiments")
    print("2. Check out the tools/ directory for utility scripts")
    print("3. Read the LangGraph documentation: https://langchain-ai.github.io/langgraph/")

if __name__ == "__main__":
    main()
'''

    @staticmethod
    def get_requirements_quickstart() -> str:
        """Get requirements-quickstart.txt template."""
        return """# LangGraph Quickstart Dependencies
langgraph>=0.2.0
langchain[anthropic]>=0.2.0
anthropic>=0.25.0
python-dotenv>=1.0.0
pydantic>=2.0.0
"""

    @staticmethod
    def get_quickstart_readme() -> str:
        """Get quickstart README template."""
        return """# Quick Start Guide

This is a complete working example of your AI agent project.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements-quickstart.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the quickstart:
   ```bash
   python quickstart.py
   ```

## What's Included

- Complete agent implementation
- Memory system
- Tool integration
- Example conversations
- Performance monitoring

## Next Steps

1. Explore the notebooks in `notebooks/`
2. Check out examples in `examples/`
3. Customize your agent in `src/`
4. Deploy with `agent-cli deploy .`

Happy coding! 🚀
"""

    # Configuration Templates
    @staticmethod
    def get_config_base() -> str:
        """Get base configuration template."""
        return """# Base Configuration
# This file contains the base configuration for all environments

agent:
  default_type: conversational
  memory:
    short_term: true
    long_term: true
    vector_store: chroma
    max_tokens: 10000
    similarity_threshold: 0.8

llm:
  default_provider: anthropic
  model: claude-3-sonnet-20240229
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  retry_attempts: 3

tools:
  enabled: true
  web_search: true
  file_operations: true
  database: true
  api: true

monitoring:
  enabled: true
  metrics: true
  tracing: true
  logging: true
  log_level: INFO
  metrics_interval: 60

api:
  enabled: true
  host: 0.0.0.0
  port: 8000
  cors_origins: ["*"]
  rate_limit: 100

storage:
  cache:
    type: redis
    host: localhost
    port: 6379
    ttl: 3600
  database:
    type: sqlite
    path: data/agent.db
"""

    @staticmethod
    def get_config_development() -> str:
        """Get development configuration template."""
        return """# Development Configuration
extends: base.yaml

llm:
  model: claude-3-haiku-20240307  # Faster, cheaper for dev
  temperature: 0.9  # More creative for experimentation
  max_tokens: 2048

monitoring:
  log_level: DEBUG
  metrics_interval: 30

api:
  port: 8001
  cors_origins: ["http://localhost:3000", "http://localhost:8080"]

storage:
  cache:
    type: memory  # Use in-memory cache for development
    ttl: 1800
"""

    @staticmethod
    def get_config_staging() -> str:
        """Get staging configuration template."""
        return """# Staging Configuration
extends: base.yaml

llm:
  model: claude-3-sonnet-20240229
  temperature: 0.7
  max_tokens: 4096

monitoring:
  log_level: INFO
  metrics_interval: 60

api:
  port: 8000
  cors_origins: ["https://staging.yourapp.com"]

storage:
  cache:
    type: redis
    host: staging-redis
    port: 6379
    ttl: 3600
"""

    @staticmethod
    def get_config_production() -> str:
        """Get production configuration template."""
        return """# Production Configuration
extends: base.yaml

llm:
  model: claude-3-sonnet-20240229
  temperature: 0.5  # More conservative for production
  max_tokens: 4096
  timeout: 60

monitoring:
  log_level: WARNING
  metrics_interval: 30

api:
  port: 8000
  cors_origins: ["https://yourapp.com"]
  rate_limit: 1000

storage:
  cache:
    type: redis
    host: production-redis
    port: 6379
    ttl: 7200
  database:
    type: postgresql
    host: production-db
    port: 5432
    database: agent_production
"""

    @staticmethod
    def get_config_agent_conversational() -> str:
        """Get conversational agent configuration."""
        return """# Conversational Agent Configuration
agent_type: conversational

personality:
  name: "AI Assistant"
  tone: "friendly and helpful"
  expertise: ["general knowledge", "conversation", "problem solving"]
  limitations: ["I cannot access real-time information", "I cannot perform actions"]

capabilities:
  chat: true
  memory: true
  context_awareness: true
  personality: true
  emotion_recognition: true

tools:
  - web_search
  - file_operations
  - database
  - calculator

prompts:
  system_prompt: "You are a helpful AI assistant. Be friendly, informative, and engaging in conversation."
  greeting: "Hello! I'm your AI assistant. How can I help you today?"
  farewell: "Goodbye! It was nice chatting with you."
"""

    @staticmethod
    def get_config_agent_research() -> str:
        """Get research agent configuration."""
        return """# Research Agent Configuration
agent_type: research

personality:
  name: "Research Assistant"
  tone: "analytical and thorough"
  expertise: ["research", "analysis", "information gathering", "citation"]
  limitations: ["I cannot access real-time information", "I cannot verify current events"]

capabilities:
  web_search: true
  document_analysis: true
  summarization: true
  citation: true
  fact_checking: true

tools:
  - web_search
  - file_operations
  - database
  - pdf_reader
  - citation_generator

prompts:
  system_prompt: "You are a research assistant. Be thorough, analytical, and always cite your sources."
  research_prompt: "Let me research this topic for you. I'll gather information from reliable sources."
  analysis_prompt: "Based on my research, here's what I found..."
"""

    @staticmethod
    def get_config_agent_automation() -> str:
        """Get automation agent configuration."""
        return """# Automation Agent Configuration
agent_type: automation

personality:
  name: "Task Automation Assistant"
  tone: "efficient and systematic"
  expertise: ["workflow automation", "task decomposition", "process optimization"]
  limitations: ["I cannot access external systems without proper authentication"]

capabilities:
  workflow_engine: true
  task_decomposition: true
  error_handling: true
  retry_logic: true
  progress_tracking: true

tools:
  - file_operations
  - api
  - database
  - scheduler
  - email_sender

prompts:
  system_prompt: "You are a task automation assistant. Be systematic, efficient, and thorough in executing workflows."
  task_prompt: "I'll help you automate this task. Let me break it down into steps."
  progress_prompt: "Task progress: {current_step}/{total_steps} - {description}"
"""

    @staticmethod
    def get_config_agent_analysis() -> str:
        """Get analysis agent configuration."""
        return """# Analysis Agent Configuration
agent_type: analysis

personality:
  name: "Data Analysis Assistant"
  tone: "analytical and precise"
  expertise: ["data analysis", "statistics", "visualization", "reporting"]
  limitations: ["I cannot access real-time data", "I cannot perform real-time calculations"]

capabilities:
  data_processing: true
  statistical_analysis: true
  visualization: true
  reporting: true
  trend_analysis: true

tools:
  - file_operations
  - database
  - api
  - chart_generator
  - statistical_library

prompts:
  system_prompt: "You are a data analysis assistant. Be precise, analytical, and provide clear insights."
  analysis_prompt: "Let me analyze this data for you. I'll provide insights and visualizations."
  report_prompt: "Here's my analysis report with key findings and recommendations."
"""

    @staticmethod
    def get_config_llm_openai() -> str:
        """Get OpenAI LLM configuration."""
        return """# OpenAI Configuration
provider: openai

models:
  default: gpt-4o
  fallback: gpt-4o-mini
  experimental: gpt-4o-preview

settings:
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  retry_attempts: 3

api:
  base_url: https://api.openai.com/v1
  version: 2024-01-01

features:
  function_calling: true
  streaming: true
  vision: true
  audio: true
"""

    @staticmethod
    def get_config_llm_anthropic() -> str:
        """Get Anthropic LLM configuration."""
        return """# Anthropic Configuration
provider: anthropic

models:
  default: claude-3-sonnet-20240229
  fallback: claude-3-haiku-20240307
  experimental: claude-3-5-sonnet-20241022

settings:
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  retry_attempts: 3

api:
  base_url: https://api.anthropic.com
  version: 2023-06-01

features:
  function_calling: true
  streaming: true
  vision: true
  tools: true
"""

    @staticmethod
    def get_config_llm_google() -> str:
        """Get Google LLM configuration."""
        return """# Google Configuration
provider: google

models:
  default: gemini-1.5-pro
  fallback: gemini-1.5-flash
  experimental: gemini-1.5-pro-latest

settings:
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  retry_attempts: 3

api:
  base_url: https://generativelanguage.googleapis.com
  version: v1

features:
  function_calling: true
  streaming: true
  vision: true
  multimodal: true
"""

    @staticmethod
    def get_config_llm_azure() -> str:
        """Get Azure LLM configuration."""
        return """# Azure Configuration
provider: azure

models:
  default: gpt-4
  fallback: gpt-35-turbo
  experimental: gpt-4-preview

settings:
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  retry_attempts: 3

api:
  base_url: https://{resource_name}.openai.azure.com
  api_version: 2024-02-15-preview

features:
  function_calling: true
  streaming: true
  vision: true
  deployment_specific: true
"""

    @staticmethod
    def get_config_llm_local() -> str:
        """Get local LLM configuration."""
        return """# Local LLM Configuration
provider: local

models:
  default: llama-3.1-8b-instruct
  fallback: mistral-7b-instruct
  experimental: codellama-34b-instruct

settings:
  temperature: 0.7
  max_tokens: 4096
  timeout: 60
  retry_attempts: 1

api:
  base_url: http://localhost:11434
  format: ollama

features:
  function_calling: false
  streaming: true
  vision: false
  offline: true
"""

    @staticmethod
    def get_config_prompt_templates() -> str:
        """Get prompt templates configuration."""
        return """# Prompt Templates Configuration

templates:
  system:
    conversational: "You are a helpful AI assistant. Be friendly, informative, and engaging in conversation."
    research: "You are a research assistant. Be thorough, analytical, and always cite your sources."
    automation: "You are a task automation assistant. Be systematic, efficient, and thorough in executing workflows."
    analysis: "You are a data analysis assistant. Be precise, analytical, and provide clear insights."
    creative: "You are a creative assistant. Be imaginative, innovative, and help bring ideas to life."

  user:
    greeting: "Hello! How can I help you today?"
    clarification: "Could you please clarify what you mean by '{term}'?"
    confirmation: "I understand you want me to {action}. Is that correct?"
    follow_up: "Is there anything else you'd like me to help you with?"

  assistant:
    greeting: "Hello! I'm your AI assistant. How can I help you today?"
    thinking: "Let me think about this..."
    processing: "I'm processing your request..."
    error: "I apologize, but I encountered an error. Let me try a different approach."
    success: "Great! I've completed the task successfully."

variables:
  user_name: "{user_name}"
  current_time: "{current_time}"
  context: "{context}"
  memory: "{memory}"
"""

    @staticmethod
    def get_config_logging() -> str:
        """Get logging configuration."""
        return """# Logging Configuration

version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
  
  detailed:
    format: "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: data/logs/agent.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: data/logs/error.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  agent_cli:
    level: INFO
    handlers: [console, file, error_file]
    propagate: false

  src:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
"""

    # Requirements Templates
    @staticmethod
    def get_requirements() -> str:
        """Get main requirements template."""
        return """# Core AI/ML
langchain>=0.2.0
langgraph>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-google-genai>=0.1.0
langchain-community>=0.0.10

# LLM Providers
openai>=1.0.0
anthropic>=0.8.0
google-generativeai>=0.3.0

# Vector Databases
chromadb>=0.4.0
pinecone-client>=3.0.0
weaviate-client>=4.0.0

# Web Framework
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.7.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Monitoring & Observability
opentelemetry-api>=1.24.0
opentelemetry-sdk>=1.24.0
prometheus-client>=0.20.0

# Utilities
python-dotenv>=1.0.0
requests>=2.31.0
pyyaml>=6.0
jinja2>=3.1.3
rich>=13.7.0
click>=8.1.0
typer>=0.9.0

# Development
jupyter>=1.0.0
ipykernel>=6.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    @staticmethod
    def get_requirements_dev() -> str:
        """Get development requirements template."""
        return """# Development Dependencies
-r requirements.txt

# Testing
pytest>=8.3.0
pytest-cov>=5.0.0
pytest-mock>=3.14.0
pytest-asyncio>=0.21.0

# Code Quality
black>=24.8.0
isort>=5.13.0
mypy>=1.11.0
flake8>=7.1.0
pre-commit>=3.8.0
bandit>=1.7.0
safety>=2.3.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=2.0.0
myst-parser>=2.0.0

# Development Tools
ipython>=8.0.0
jupyterlab>=4.0.0
"""

    @staticmethod
    def get_requirements_prod() -> str:
        """Get production requirements template."""
        return """# Production Dependencies
-r requirements.txt

# Production Server
gunicorn>=21.0.0
uvloop>=0.19.0

# Production Database
psycopg2-binary>=2.9.0
redis>=5.0.0

# Task Queue
celery>=5.3.0
flower>=2.0.0

# Monitoring
grafana-api>=1.0.0
prometheus-api-client>=0.5.0

# Security
cryptography>=41.0.0
"""

    # CLI Templates
    @staticmethod
    def get_cli_main() -> str:
        """Get CLI main template."""
        return '''"""CLI main entry point for the agent project."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def main():
    """Main CLI entry point."""
    console.print(Panel.fit(
        "[bold blue]🤖 Agent CLI[/bold blue]\\n"
        "Welcome to your AI agent project!",
        border_style="blue"
    ))
    
    console.print("Available commands:")
    console.print("  [cyan]run[/cyan] - Start the agent")
    console.print("  [cyan]chat[/cyan] - Interactive chat")
    console.print("  [cyan]deploy[/cyan] - Deploy to production")
    console.print("  [cyan]monitor[/cyan] - View metrics")

if __name__ == "__main__":
    main()
'''

    @staticmethod
    def get_cli_command_run() -> str:
        """Get CLI run command template."""
        return '''"""Run command for the agent CLI."""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

async def run_agent(config_path: str = "config/development.yaml"):
    """Run the agent with the specified configuration."""
    console.print("[bold green]🚀 Starting agent...[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initializing agent...", total=None)
        
        # Load configuration
        progress.update(task, description="[cyan]Loading configuration...")
        await asyncio.sleep(1)
        
        # Initialize components
        progress.update(task, description="[cyan]Initializing components...")
        await asyncio.sleep(1)
        
        # Start agent
        progress.update(task, description="[cyan]Starting agent...")
        await asyncio.sleep(1)
        
        progress.update(task, description="[green]Agent is running!")
    
    console.print("[green]✅ Agent started successfully![/green]")
    console.print("Press Ctrl+C to stop")

def main():
    """Main entry point for run command."""
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        console.print("\\n[yellow]👋 Goodbye![/yellow]")

if __name__ == "__main__":
    main()
'''

    @staticmethod
    def get_cli_command_chat() -> str:
        """Get CLI chat command template."""
        return '''"""Chat command for the agent CLI."""

import asyncio
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

async def chat_with_agent():
    """Start interactive chat with the agent."""
    console.print(Panel.fit(
        "[bold green]💬 Interactive Chat[/bold green]\\n"
        "Type 'quit' to exit",
        border_style="green"
    ))
    
    while True:
        try:
            user_input = Prompt.ask("\\n[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            
            # Process user input (this would integrate with actual agent)
            console.print(f"[bold green]Agent[/bold green]: Hello! I'm your AI assistant. How can I help you today?")
            
        except KeyboardInterrupt:
            console.print("\\n[yellow]👋 Goodbye![/yellow]")
            break

def main():
    """Main entry point for chat command."""
    try:
        asyncio.run(chat_with_agent())
    except KeyboardInterrupt:
        console.print("\\n[yellow]👋 Goodbye![/yellow]")

if __name__ == "__main__":
    main()
'''

    # Source Code Templates
    @staticmethod
    def get_base_agent() -> str:
        """Get base agent template."""
        return '''"""Base agent implementation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

class AgentMessage(BaseModel):
    """Message model for agent communication."""
    content: str
    sender: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    """Response model for agent interactions."""
    content: str
    success: bool
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the agent with configuration."""
        self.config = config
        self.name = config.get("name", "AI Agent")
        self.personality = config.get("personality", {})
        self.capabilities = config.get("capabilities", [])
        self.tools = config.get("tools", [])
        self.memory = None
        self.llm_client = None
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process an incoming message and return a response."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent components."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        if self.memory:
            await self.memory.close()
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return self.capabilities
    
    def get_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.tools
'''

    @staticmethod
    def get_conversational_agent() -> str:
        """Get conversational agent template."""
        return '''"""Conversational agent implementation."""

from typing import Dict, Any
from .base.base_agent import BaseAgent, AgentMessage, AgentResponse

class ConversationalAgent(BaseAgent):
    """Conversational agent for chat and dialogue."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize conversational agent."""
        super().__init__(config)
        self.conversation_history = []
        self.personality_traits = config.get("personality", {})
    
    async def initialize(self) -> bool:
        """Initialize conversational agent components."""
        # Initialize LLM client
        # Initialize memory system
        # Load personality configuration
        return True
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process conversational message."""
        try:
            # Add message to conversation history
            self.conversation_history.append(message)
            
            # Generate response using LLM
            response_content = await self._generate_response(message)
            
            # Create response
            response = AgentResponse(
                content=response_content,
                success=True,
                metadata={"conversation_length": len(self.conversation_history)}
            )
            
            return response
            
        except Exception as e:
            return AgentResponse(
                content="I apologize, but I encountered an error processing your message.",
                success=False,
                error=str(e)
            )
    
    async def _generate_response(self, message: AgentMessage) -> str:
        """Generate response using LLM."""
        # This would integrate with actual LLM client
        return f"Hello! I'm {self.name}. How can I help you today?"
    
    def get_conversation_history(self) -> List[AgentMessage]:
        """Get conversation history."""
        return self.conversation_history.copy()
'''

    @staticmethod
    def get_research_agent() -> str:
        """Get research agent template."""
        return '''"""Research agent implementation."""

from typing import Dict, Any, List
from .base.base_agent import BaseAgent, AgentMessage, AgentResponse

class ResearchAgent(BaseAgent):
    """Research agent for information gathering and analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize research agent."""
        super().__init__(config)
        self.research_tools = config.get("research_tools", [])
        self.analysis_capabilities = config.get("analysis_capabilities", [])
    
    async def initialize(self) -> bool:
        """Initialize research agent components."""
        # Initialize web search tools
        # Initialize document analysis tools
        # Initialize citation system
        return True
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process research request."""
        try:
            # Analyze the research request
            research_query = await self._analyze_query(message.content)
            
            # Gather information
            sources = await self._gather_information(research_query)
            
            # Analyze and synthesize
            analysis = await self._analyze_sources(sources)
            
            # Generate response with citations
            response_content = await self._generate_research_response(analysis, sources)
            
            return AgentResponse(
                content=response_content,
                success=True,
                metadata={"sources": len(sources), "analysis_type": analysis.get("type")}
            )
            
        except Exception as e:
            return AgentResponse(
                content="I apologize, but I encountered an error during research.",
                success=False,
                error=str(e)
            )
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze research query."""
        # Extract key terms, scope, and requirements
        return {"terms": query.split(), "scope": "general"}
    
    async def _gather_information(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather information from various sources."""
        # Use web search, databases, etc.
        return []
    
    async def _analyze_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gathered sources."""
        # Perform analysis and synthesis
        return {"type": "general", "summary": "Analysis complete"}
    
    async def _generate_research_response(self, analysis: Dict[str, Any], sources: List[Dict[str, Any]]) -> str:
        """Generate research response with citations."""
        return "Based on my research, here's what I found..."
'''

    @staticmethod
    def get_automation_agent() -> str:
        """Get automation agent template."""
        return '''"""Automation agent implementation."""

from typing import Dict, Any, List
from .base.base_agent import BaseAgent, AgentMessage, AgentResponse

class AutomationAgent(BaseAgent):
    """Automation agent for task execution and workflow orchestration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize automation agent."""
        super().__init__(config)
        self.workflow_engine = None
        self.task_queue = []
        self.execution_history = []
    
    async def initialize(self) -> bool:
        """Initialize automation agent components."""
        # Initialize workflow engine
        # Initialize task queue
        # Load automation tools
        return True
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process automation request."""
        try:
            # Parse automation request
            task = await self._parse_automation_request(message.content)
            
            # Decompose into subtasks
            subtasks = await self._decompose_task(task)
            
            # Execute subtasks
            results = await self._execute_subtasks(subtasks)
            
            # Generate response
            response_content = await self._generate_automation_response(results)
            
            return AgentResponse(
                content=response_content,
                success=True,
                metadata={"tasks_completed": len(results), "execution_time": 0}
            )
            
        except Exception as e:
            return AgentResponse(
                content="I apologize, but I encountered an error during automation.",
                success=False,
                error=str(e)
            )
    
    async def _parse_automation_request(self, request: str) -> Dict[str, Any]:
        """Parse automation request into structured task."""
        return {"type": "general", "description": request}
    
    async def _decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task into subtasks."""
        return [task]  # Simple implementation
    
    async def _execute_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute subtasks."""
        results = []
        for subtask in subtasks:
            result = await self._execute_single_task(subtask)
            results.append(result)
        return results
    
    async def _execute_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task."""
        # This would integrate with actual automation tools
        return {"status": "completed", "result": "Task executed successfully"}
    
    async def _generate_automation_response(self, results: List[Dict[str, Any]]) -> str:
        """Generate automation response."""
        return f"Automation completed successfully. {len(results)} tasks executed."
'''

    @staticmethod
    def get_analysis_agent() -> str:
        """Get analysis agent template."""
        return '''"""Analysis agent implementation."""

from typing import Dict, Any, List
from .base.base_agent import BaseAgent, AgentMessage, AgentResponse

class AnalysisAgent(BaseAgent):
    """Analysis agent for data processing and insights generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analysis agent."""
        super().__init__(config)
        self.analysis_tools = config.get("analysis_tools", [])
        self.visualization_capabilities = config.get("visualization_capabilities", [])
    
    async def initialize(self) -> bool:
        """Initialize analysis agent components."""
        # Initialize data processing tools
        # Initialize statistical analysis libraries
        # Initialize visualization tools
        return True
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process analysis request."""
        try:
            # Parse analysis request
            analysis_request = await self._parse_analysis_request(message.content)
            
            # Load and process data
            data = await self._load_data(analysis_request)
            
            # Perform analysis
            analysis_results = await self._perform_analysis(data, analysis_request)
            
            # Generate visualizations
            visualizations = await self._generate_visualizations(analysis_results)
            
            # Generate response
            response_content = await self._generate_analysis_response(analysis_results, visualizations)
            
            return AgentResponse(
                content=response_content,
                success=True,
                metadata={"data_points": len(data), "analysis_type": analysis_request.get("type")}
            )
            
        except Exception as e:
            return AgentResponse(
                content="I apologize, but I encountered an error during analysis.",
                success=False,
                error=str(e)
            )
    
    async def _parse_analysis_request(self, request: str) -> Dict[str, Any]:
        """Parse analysis request."""
        return {"type": "general", "description": request}
    
    async def _load_data(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load data for analysis."""
        # Load from file, database, API, etc.
        return []
    
    async def _perform_analysis(self, data: List[Dict[str, Any]], request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis."""
        # Perform various analyses
        return {"summary": "Analysis complete", "insights": []}
    
    async def _generate_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Generate visualizations."""
        # Create charts, graphs, etc.
        return []
    
    async def _generate_analysis_response(self, results: Dict[str, Any], visualizations: List[str]) -> str:
        """Generate analysis response."""
        return "Here's my analysis of the data..."
'''

    @staticmethod
    def get_agent_service() -> str:
        """Get agent service template."""
        return '''"""Agent service implementation."""

from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class AgentServiceRequest(BaseModel):
    """Request model for agent service."""
    message: str
    agent_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class AgentServiceResponse(BaseModel):
    """Response model for agent service."""
    response: str
    success: bool
    agent_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class AgentService:
    """Service for managing agent interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agent service."""
        self.config = config
        self.agents = {}
        self.sessions = {}
    
    async def initialize(self) -> bool:
        """Initialize the service."""
        # Load agent configurations
        # Initialize agent instances
        # Setup monitoring
        return True
    
    async def process_request(self, request: AgentServiceRequest) -> AgentServiceResponse:
        """Process agent service request."""
        try:
            # Get or create agent
            agent = await self._get_agent(request.agent_type)
            
            # Process message
            response = await agent.process_message(request.message)
            
            # Create service response
            service_response = AgentServiceResponse(
                response=response.content,
                success=response.success,
                agent_type=request.agent_type,
                timestamp=datetime.now(),
                metadata=response.metadata,
                error=response.error
            )
            
            return service_response
            
        except Exception as e:
            return AgentServiceResponse(
                response="Service error occurred",
                success=False,
                agent_type=request.agent_type,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _get_agent(self, agent_type: str):
        """Get or create agent instance."""
        if agent_type not in self.agents:
            # Create new agent instance
            pass
        return self.agents.get(agent_type)
    
    async def cleanup(self) -> None:
        """Clean up service resources."""
        for agent in self.agents.values():
            await agent.cleanup()
''' 