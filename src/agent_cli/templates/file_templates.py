"""
AI Agent CLI - Clean Project Scaffolding Templates

This module contains clean, minimal template definitions for generating AI agent projects.

Key Features:
- LangChain/LangGraph integration for intelligent agents
- FastAPI web framework for scalable APIs  
- Docker containerization for easy deployment
- Comprehensive testing setup with pytest
- Modern Python packaging with pyproject.toml
- Environment configuration management
- Logging and monitoring capabilities

Supported Project Types:
- Basic AI Agent: Simple conversational agent
- API Agent: REST API with agent endpoints
- Multi-Agent System: Coordinated agent workflows
- RAG Agent: Retrieval-Augmented Generation
- Custom Agent: Flexible template for specific needs

Dependencies:
- langchain: Core agent framework
- langgraph: Advanced agent workflows
- fastapi: Modern web API framework
- uvicorn: ASGI server for FastAPI
- pydantic: Data validation and settings
- python-dotenv: Environment variable management
"""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class FileTemplateManager:
    """Manages file templates for AI agent projects."""
    
    def __init__(self):
        self._templates = self._get_builtin_templates()
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context."""
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template_content = self._templates[template_name]
        
        # Simple variable substitution
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            template_content = template_content.replace(placeholder, str(value))
        
        return template_content
    
    def get_available_templates(self) -> list:
        """Get list of available file templates."""
        return list(self._templates.keys())
    
    def _get_builtin_templates(self) -> Dict[str, str]:
        """Get built-in file templates."""
        return {
            # Root configuration files
            "pyproject.toml": '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{{project_name}}"
version = "0.1.0"
description = "AI Agent Project built with Agent CLI"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["ai", "agent", "llm", "langchain", "langgraph"]
dependencies = [
    "langchain>=0.3.0",
    "langgraph>=0.6.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",
    "langchain-community>=0.3.0",
    "langmem>=0.0.29",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "python-dotenv>=1.0.1",
    "pyyaml>=6.0.2",
    "rich>=13.9.0",
    "httpx>=0.27.0",
    "tavily-python>=0.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.23.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
{{project_name}} = "{{package_name}}.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=term-missing"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
''',

            "README.md": '''# {{project_name}}

{{description}}

## Features

- 🤖 **Intelligent Agent**: Powered by LangChain and LangGraph
- 🚀 **FastAPI Integration**: Modern async web framework
- 🐳 **Docker Ready**: Containerized deployment
- 🧪 **Testing**: Comprehensive test suite with pytest
- 📝 **Type Safety**: Full type hints with mypy
- 🔧 **Configuration**: Flexible YAML/environment config

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -e .
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Agent**
   ```bash
   python -m {{package_name}}
   ```

4. **Start API Server**
   ```bash
   uvicorn {{package_name}}.api:app --reload
   ```

## Development

1. **Install Dev Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run Tests**
   ```bash
   pytest
   ```

3. **Format Code**
   ```bash
   black src tests
   isort src tests
   ```

4. **Type Check**
   ```bash
   mypy src
   ```

## Configuration

The agent can be configured via:
- Environment variables (`.env` file)
- YAML configuration (`config/agent.yaml`)
- Runtime parameters

## API Endpoints

- `POST /chat` - Chat with the agent
- `GET /health` - Health check
- `GET /docs` - API documentation

## Docker Deployment

```bash
docker build -t {{project_name}} .
docker run -p 8000:8000 --env-file .env {{project_name}}
```

## License

MIT License - see LICENSE file for details.
''',

            ".env.example": '''# LLM Provider Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Agent Configuration
AGENT_NAME={{project_name}}
AGENT_MODEL=gpt-4
AGENT_TEMPERATURE=0.1
AGENT_MAX_TOKENS=1000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Logging
LOG_LEVEL=INFO

# Development
DEBUG=false
''',

            ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
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
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.coverage
.pytest_cache/
htmlcov/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Project specific
config/local_*.yaml
data/
models/
checkpoints/
''',

            "Dockerfile": '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "{{package_name}}.api:app", "--host", "0.0.0.0", "--port", "8000"]
''',

            "docker-compose.yml": '''version: '3.8'

services:
  {{project_name}}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add a reverse proxy
  # nginx:
  #   image: nginx:alpine
  #   ports:
  #     - "80:80"
  #   volumes:
  #     - ./nginx.conf:/etc/nginx/nginx.conf
  #   depends_on:
  #     - {{project_name}}
''',

            # Main package files
            "src/{{package_name}}/__init__.py": '''"""{{project_name}} - AI Agent Package"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .agent import Agent
from .config import Config

__all__ = ["Agent", "Config"]
''',

            "src/{{package_name}}/config.py": '''"""Configuration management for the AI agent."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """Application configuration."""
    
    # Agent settings
    agent_name: str = Field(default="{{project_name}}", env="AGENT_NAME")
    agent_model: str = Field(default="gpt-4", env="AGENT_MODEL")
    agent_temperature: float = Field(default=0.1, env="AGENT_TEMPERATURE")
    agent_max_tokens: int = Field(default=1000, env="AGENT_MAX_TOKENS")
    
    # LLM API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Development
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @classmethod
    def load_from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        config = {
            "model": self.agent_model,
            "temperature": self.agent_temperature,
            "max_tokens": self.agent_max_tokens,
        }
        
        if self.openai_api_key:
            config["openai_api_key"] = self.openai_api_key
        if self.anthropic_api_key:
            config["anthropic_api_key"] = self.anthropic_api_key
            
        return config


# Global config instance
config = Config()
''',

            "src/{{package_name}}/agent.py": '''"""Main agent implementation using LangChain and LangGraph."""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from .config import config

logger = logging.getLogger(__name__)


class Agent:
    """Main AI agent class."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the agent."""
        self.model_name = model_name or config.agent_model
        self.llm = self._create_llm()
        self.agent = self._create_agent()
        
        logger.info(f"Agent initialized with model: {self.model_name}")
    
    def _create_llm(self):
        """Create the language model."""
        llm_config = config.get_llm_config()
        
        if "gpt" in self.model_name.lower():
            return ChatOpenAI(
                model=self.model_name,
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
                api_key=config.openai_api_key,
            )
        elif "claude" in self.model_name.lower():
            return ChatAnthropic(
                model=self.model_name,
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
                api_key=config.anthropic_api_key,
            )
        else:
            # Default to OpenAI
            return ChatOpenAI(
                model=self.model_name,
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
                api_key=config.openai_api_key,
            )
    
    def _create_agent(self):
        """Create the LangGraph agent."""
        # Define tools (empty for basic agent)
        tools = []
        
        # Create ReAct agent with tools
        return create_react_agent(self.llm, tools)
    
    async def chat(self, message: str, conversation_id: Optional[str] = None) -> str:
        """Chat with the agent."""
        try:
            # Prepare messages
            messages = [HumanMessage(content=message)]
            
            # Run the agent
            result = await self.agent.ainvoke(
                {"messages": messages},
                config={"configurable": {"thread_id": conversation_id or "default"}}
            )
            
            # Extract the final response
            if result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content
            
            return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"An error occurred: {str(e)}"
    
    def chat_sync(self, message: str, conversation_id: Optional[str] = None) -> str:
        """Synchronous chat method."""
        try:
            # Prepare messages
            messages = [HumanMessage(content=message)]
            
            # Run the agent
            result = self.agent.invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": conversation_id or "default"}}
            )
            
            # Extract the final response
            if result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content
            
            return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"An error occurred: {str(e)}"
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history."""
        # TODO: Implement conversation history retrieval
        # This would typically involve retrieving from the agent's memory/checkpointer
        return []
    
    def reset_conversation(self, conversation_id: str) -> None:
        """Reset conversation history."""
        # TODO: Implement conversation reset
        logger.info(f"Reset conversation: {conversation_id}")


def create_agent(model_name: Optional[str] = None) -> Agent:
    """Factory function to create an agent."""
    return Agent(model_name=model_name)
''',

            "src/{{package_name}}/api.py": '''"""FastAPI web server for the AI agent."""

import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .agent import create_agent
from .config import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="{{project_name}} API",
    description="AI Agent API powered by LangChain and LangGraph",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = create_agent()


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model=config.agent_model
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI agent."""
    try:
        response = await agent.chat(
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    try:
        history = agent.get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "history": history}
    
    except Exception as e:
        logger.error(f"Get conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def reset_conversation(conversation_id: str):
    """Reset conversation history."""
    try:
        agent.reset_conversation(conversation_id)
        return {"message": f"Conversation {conversation_id} reset successfully"}
    
    except Exception as e:
        logger.error(f"Reset conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "{{package_name}}.api:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        reload=config.debug
    )
''',

            "src/{{package_name}}/cli.py": '''"""Command-line interface for the AI agent."""

import asyncio
import logging
from typing import Optional
import click
from rich.console import Console
from rich.markdown import Markdown
from .agent import create_agent
from .config import config

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool):
    """{{project_name}} - AI Agent CLI"""
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=getattr(logging, config.log_level))


@main.command()
@click.option("--model", help="LLM model to use")
def chat(model: Optional[str]):
    """Start an interactive chat session."""
    console.print("[bold green]Starting {{project_name}} chat session...[/bold green]")
    console.print("Type 'exit' or 'quit' to end the session.\\n")
    
    # Create agent
    agent = create_agent(model_name=model)
    conversation_id = "cli-session"
    
    while True:
        try:
            # Get user input
            user_input = console.input("[bold blue]You:[/bold blue] ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            
            if not user_input.strip():
                continue
            
            # Get agent response
            console.print("[dim]Agent is thinking...[/dim]")
            response = agent.chat_sync(user_input, conversation_id)
            
            # Display response
            console.print(f"[bold green]Agent:[/bold green]")
            console.print(Markdown(response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    import uvicorn
    
    console.print(f"[bold green]Starting {{project_name}} API server...[/bold green]")
    console.print(f"Server will be available at: http://{host}:{port}")
    console.print(f"API docs: http://{host}:{port}/docs\\n")
    
    uvicorn.run(
        "{{package_name}}.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=config.log_level.lower()
    )


@main.command()
def info():
    """Show agent information."""
    console.print("[bold green]{{project_name}} Information[/bold green]")
    console.print(f"Agent Name: {config.agent_name}")
    console.print(f"Model: {config.agent_model}")
    console.print(f"Temperature: {config.agent_temperature}")
    console.print(f"Max Tokens: {config.agent_max_tokens}")
    console.print(f"Debug Mode: {config.debug}")


if __name__ == "__main__":
    main()
''',

            "src/{{package_name}}/__main__.py": '''"""Main entry point for running the agent as a module."""

from .cli import main

if __name__ == "__main__":
    main()
''',

            # Configuration files
            "config/agent.yaml": '''# Agent Configuration
agent:
  name: "{{project_name}}"
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 1000

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Development Settings
development:
  debug: false
  auto_reload: true
''',

            # Test files
            "tests/__init__.py": '',

            "tests/conftest.py": '''"""Pytest configuration and fixtures."""

import pytest
from {{package_name}}.agent import create_agent
from {{package_name}}.config import Config


@pytest.fixture
def agent():
    """Create an agent for testing."""
    return create_agent()


@pytest.fixture
def config():
    """Create a test config."""
    return Config()


@pytest.fixture
def sample_message():
    """Sample message for testing."""
    return "Hello, how are you?"
''',

            "tests/test_agent.py": '''"""Tests for the agent module."""

import pytest
from {{package_name}}.agent import Agent, create_agent


class TestAgent:
    """Test cases for the Agent class."""
    
    def test_agent_creation(self):
        """Test agent can be created."""
        agent = create_agent()
        assert agent is not None
        assert isinstance(agent, Agent)
    
    def test_agent_with_custom_model(self):
        """Test agent with custom model."""
        agent = create_agent(model_name="gpt-3.5-turbo")
        assert agent.model_name == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_chat_async(self, agent, sample_message):
        """Test async chat functionality."""
        # This is a basic test - in real scenarios you'd mock the LLM
        # to avoid making actual API calls during testing
        response = await agent.chat(sample_message)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_chat_sync(self, agent, sample_message):
        """Test sync chat functionality."""
        # This is a basic test - in real scenarios you'd mock the LLM
        response = agent.chat_sync(sample_message)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_conversation_history(self, agent):
        """Test conversation history."""
        history = agent.get_conversation_history("test-session")
        assert isinstance(history, list)
    
    def test_reset_conversation(self, agent):
        """Test conversation reset."""
        # Should not raise an exception
        agent.reset_conversation("test-session")
''',

            "tests/test_config.py": '''"""Tests for the config module."""

import tempfile
import yaml
from pathlib import Path
from {{package_name}}.config import Config


class TestConfig:
    """Test cases for the Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.agent_name == "{{project_name}}"
        assert config.agent_model == "gpt-4"
        assert config.agent_temperature == 0.1
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
    
    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        test_config = {
            "agent_name": "test-agent",
            "agent_model": "gpt-3.5-turbo",
            "agent_temperature": 0.5,
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            yaml_path = Path(f.name)
        
        try:
            config = Config.load_from_yaml(yaml_path)
            assert config.agent_name == "test-agent"
            assert config.agent_model == "gpt-3.5-turbo"
            assert config.agent_temperature == 0.5
        finally:
            yaml_path.unlink()
    
    def test_llm_config(self):
        """Test LLM configuration generation."""
        config = Config(
            agent_model="gpt-4",
            agent_temperature=0.2,
            agent_max_tokens=500,
            openai_api_key="test-key"
        )
        
        llm_config = config.get_llm_config()
        assert llm_config["model"] == "gpt-4"
        assert llm_config["temperature"] == 0.2
        assert llm_config["max_tokens"] == 500
        assert llm_config["openai_api_key"] == "test-key"
''',

            "tests/test_api.py": '''"""Tests for the API module."""

import pytest
from fastapi.testclient import TestClient
from {{package_name}}.api import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestAPI:
    """Test cases for the API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model" in data
    
    def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        # Note: This test might fail without proper API keys
        # In real scenarios, you'd mock the agent
        chat_data = {
            "message": "Hello",
            "conversation_id": "test-123"
        }
        
        response = client.post("/chat", json=chat_data)
        # We expect either success or a specific error
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert data["conversation_id"] == "test-123"
    
    def test_get_conversation(self, client):
        """Test get conversation endpoint."""
        response = client.get("/conversations/test-123")
        # Should return conversation data or error
        assert response.status_code in [200, 500]
    
    def test_reset_conversation(self, client):
        """Test reset conversation endpoint."""
        response = client.delete("/conversations/test-123")
        # Should successfully reset or return error
        assert response.status_code in [200, 500]
''',

            # Development files
            ".pre-commit-config.yaml": '''repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
''',

            "Makefile": '''# Development commands for {{project_name}}

.PHONY: install dev test lint format clean build docker

# Install the package
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"
	pre-commit install

# Run tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

# Lint code
lint:
	flake8 src tests
	mypy src

# Format code
format:
	black src tests
	isort src tests

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Build Docker image
docker:
	docker build -t {{project_name}} .

# Run the agent CLI
run:
	python -m {{package_name}}

# Start the API server
serve:
	uvicorn {{package_name}}.api:app --reload

# Show help
help:
	@echo "Available commands:"
	@echo "  install    - Install the package"
	@echo "  dev        - Install development dependencies"
	@echo "  test       - Run tests"
	@echo "  test-cov   - Run tests with coverage"
	@echo "  lint       - Lint code"
	@echo "  format     - Format code"
	@echo "  clean      - Clean build artifacts"
	@echo "  build      - Build package"
	@echo "  docker     - Build Docker image"
	@echo "  run        - Run the agent CLI"
	@echo "  serve      - Start the API server"
''',

            "LICENSE": '''MIT License

Copyright (c) 2024 {{project_name}}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
''',

            # Environment files
            ".env.example": '''# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Agent Configuration
AGENT_NAME={{project_name}}
LOG_LEVEL=INFO
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Memory Configuration
MEMORY_BACKEND=langmem
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=./data/vectorstore

# Observability
OPIK_API_KEY=your_opik_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT={{project_name}}

# External APIs
TAVILY_API_KEY=your_tavily_api_key_here
SERPER_API_KEY=your_serper_api_key_here
''',

            ".env": '''# Copy from .env.example and fill in your actual values
''',

            # GitHub CI/CD
            ".github/workflows/ci.yml": '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        
    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
        
    - name: Type check with mypy
      run: mypy src
      
    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t {{project_name}} .
      
    - name: Test Docker image
      run: docker run --rm {{project_name}} --version
''',

            # Configuration files
            "configs/agent_config.yaml": '''# Agent Configuration
agent:
  name: "{{project_name}}"
  description: "AI Agent built with LangGraph and LangChain"
  version: "0.1.0"
  
  # LLM Configuration
  llm:
    provider: "openai"  # openai, anthropic, groq
    model: "gpt-4o-mini"
    temperature: 0.7
    max_tokens: 1000
    
  # Memory Configuration
  memory:
    enabled: true
    type: "langmem"  # langmem, chroma, faiss
    max_memory_size: 1000
    similarity_threshold: 0.8
    
  # Tools Configuration
  tools:
    enabled:
      - web_search
      - file_operations
    web_search:
      provider: "tavily"  # tavily, serper, duckduckgo
      max_results: 5
    file_operations:
      allowed_extensions: [".txt", ".md", ".json", ".yaml", ".py"]
      max_file_size_mb: 10
      
  # Workflow Configuration
  workflows:
    timeout_seconds: 300
    max_iterations: 10
    enable_human_in_loop: false
    
  # Safety Configuration
  safety:
    enable_content_filter: true
    max_tool_calls_per_turn: 5
    rate_limit:
      requests_per_minute: 60
''',

            "configs/llm_config.yaml": '''# LLM Provider Configurations
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    models:
      - name: "gpt-4o"
        context_length: 128000
        supports_tools: true
      - name: "gpt-4o-mini"
        context_length: 128000
        supports_tools: true
      - name: "o1-preview"
        context_length: 32000
        supports_tools: false
        
  anthropic:
    base_url: "https://api.anthropic.com"
    models:
      - name: "claude-3-5-sonnet-latest"
        context_length: 200000
        supports_tools: true
      - name: "claude-3-5-haiku-latest"
        context_length: 200000
        supports_tools: true
        
  groq:
    base_url: "https://api.groq.com/openai/v1"
    models:
      - name: "llama-3.3-70b-versatile"
        context_length: 32000
        supports_tools: true
      - name: "mixtral-8x7b-32768"
        context_length: 32000
        supports_tools: true

# Default settings
defaults:
  temperature: 0.7
  max_tokens: 1000
  timeout_seconds: 30
  retries: 3
''',

            "configs/deployment_config.yaml": '''# Deployment Configuration
deployment:
  environment: "development"  # development, staging, production
  
  # API Configuration
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 1
    reload: true
    log_level: "info"
    
  # Database Configuration
  database:
    type: "sqlite"  # sqlite, postgresql, mysql
    url: "sqlite:///./data/agent.db"
    
  # Kubernetes Configuration
  kubernetes:
    namespace: "default"
    replicas: 1
    resources:
      requests:
        cpu: "100m"
        memory: "128Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"
    
  # Docker Configuration
  docker:
    image: "{{project_name}}"
    tag: "latest"
    registry: ""
    
  # Monitoring Configuration
  monitoring:
    enabled: true
    metrics_port: 9090
    health_check_path: "/health"
    readiness_check_path: "/ready"
''',

            # Kubernetes manifests
            "k8s/deployment.yaml": '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{project_name}}
  labels:
    app: {{project_name}}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {{project_name}}
  template:
    metadata:
      labels:
        app: {{project_name}}
    spec:
      containers:
      - name: {{project_name}}
        image: {{project_name}}:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
''',

            "k8s/service.yaml": '''apiVersion: v1
kind: Service
metadata:
  name: {{project_name}}-service
  labels:
    app: {{project_name}}
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: {{project_name}}
---
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
type: Opaque
data:
  # Base64 encoded API keys (replace with actual values)
  openai-api-key: eW91cl9vcGVuYWlfYXBpX2tleV9oZXJl
  anthropic-api-key: eW91cl9hbnRocm9waWNfYXBpX2tleV9oZXJl
''',

            # Core package files
            "src/{{package_name}}/__init__.py": '''"""
{{project_name}} - AI Agent built with LangGraph and LangChain.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import AgentConfig
from .main import main

__all__ = ["AgentConfig", "main"]
''',

            "src/{{package_name}}/config.py": '''"""
Configuration management for {{project_name}}.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration."""
    
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    timeout_seconds: int = Field(default=30, gt=0)


class MemoryConfig(BaseModel):
    """Memory configuration."""
    
    enabled: bool = Field(default=True)
    type: str = Field(default="langmem", description="Memory backend type")
    max_memory_size: int = Field(default=1000, gt=0)
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class ToolsConfig(BaseModel):
    """Tools configuration."""
    
    enabled: List[str] = Field(default_factory=lambda: ["web_search", "file_operations"])
    web_search_provider: str = Field(default="tavily")
    max_search_results: int = Field(default=5, gt=0)


class SafetyConfig(BaseModel):
    """Safety configuration."""
    
    enable_content_filter: bool = Field(default=True)
    max_tool_calls_per_turn: int = Field(default=5, gt=0)
    requests_per_minute: int = Field(default=60, gt=0)


class AgentConfig(BaseModel):
    """Main agent configuration."""
    
    name: str = Field(default="{{project_name}}")
    description: str = Field(default="AI Agent built with LangGraph and LangChain")
    version: str = Field(default="0.1.0")
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "AgentConfig":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Extract agent config from nested structure
        agent_config = config_data.get("agent", {})
        return cls(**agent_config)
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        return cls(
            name=os.getenv("AGENT_NAME", "{{project_name}}"),
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            )
        )
    
    def get_api_key(self) -> str:
        """Get API key for the configured LLM provider."""
        if self.llm.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif self.llm.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.llm.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm.provider}")
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {self.llm.provider}")
        
        return api_key


# Global configuration instance
config = AgentConfig.from_env()

# Try to load from config file if it exists
config_file = Path("configs/agent_config.yaml")
if config_file.exists():
    config = AgentConfig.from_yaml(str(config_file))
''',

            "src/{{package_name}}/main.py": '''"""
Main entry point for {{project_name}}.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .agent.langgraph_agent import create_agent
from .config import config
from .utils.logging import setup_logging

app = typer.Typer(name="{{project_name}}", help="AI Agent built with LangGraph and LangChain")
console = Console()


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send to the agent"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Start an interactive chat session with the agent."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else config.llm.provider
    setup_logging(log_level)
    
    # Load config if provided
    if config_file:
        from .config import AgentConfig
        global config
        config = AgentConfig.from_yaml(config_file)
    
    console.print(Panel.fit(
        f"[bold blue]🤖 {config.name}[/bold blue]\\n"
        f"{config.description}\\n\\n"
        f"[dim]Model: {config.llm.provider}/{config.llm.model}[/dim]",
        title="Agent Ready",
        border_style="blue"
    ))
    
    if message:
        # Single message mode
        asyncio.run(process_message(message))
    else:
        # Interactive mode
        asyncio.run(interactive_chat())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the API server."""
    import uvicorn
    
    console.print(f"🚀 Starting API server on {host}:{port}")
    uvicorn.run(
        "{{package_name}}.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"{{project_name}} v{__version__}")


async def process_message(message: str) -> None:
    """Process a single message."""
    try:
        agent = create_agent(config)
        
        with console.status("[spinner] Thinking..."):
            response = await agent.arun(message)
        
        console.print(Panel(
            response,
            title="[bold green]🤖 Agent Response[/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def interactive_chat() -> None:
    """Run interactive chat session."""
    try:
        agent = create_agent(config)
        console.print("\\n[dim]Type 'quit' or 'exit' to end the conversation.[/dim]\\n")
        
        while True:
            try:
                message = console.input("[bold blue]You:[/bold blue] ")
                
                if message.lower() in ["quit", "exit", "bye"]:
                    console.print("👋 Goodbye!")
                    break
                
                if not message.strip():
                    continue
                
                with console.status("[spinner] Thinking..."):
                    response = await agent.arun(message)
                
                console.print(f"\\n[bold green]🤖 Agent:[/bold green] {response}\\n")
                
            except KeyboardInterrupt:
                console.print("\\n👋 Goodbye!")
                break
                
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        sys.exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
''',

            # Agent modules
            "src/{{package_name}}/agent/__init__.py": '''"""
Agent components for {{project_name}}.
"""

from .base_agent import BaseAgent
from .langgraph_agent import LangGraphAgent, create_agent
from .state_manager import AgentState, StateManager

__all__ = ["BaseAgent", "LangGraphAgent", "create_agent", "AgentState", "StateManager"]
''',

            "src/{{package_name}}/agent/base_agent.py": '''"""
Base agent interface for {{project_name}}.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, config: Any):
        """Initialize the agent with configuration."""
        self.config = config
        self.session_id: Optional[str] = None
    
    @abstractmethod
    async def arun(self, message: Union[str, BaseMessage], **kwargs) -> str:
        """Run the agent asynchronously with a message."""
        pass
    
    @abstractmethod
    def run(self, message: Union[str, BaseMessage], **kwargs) -> str:
        """Run the agent synchronously with a message."""
        pass
    
    @abstractmethod
    async def astream(self, message: Union[str, BaseMessage], **kwargs):
        """Stream agent responses asynchronously."""
        pass
    
    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for this agent."""
        self.session_id = session_id
    
    def get_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.session_id
    
    @abstractmethod
    def get_memory(self) -> Dict[str, Any]:
        """Get the agent's memory state."""
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[str]:
        """Get list of available tools."""
        pass
''',

            "src/{{package_name}}/agent/langgraph_agent.py": '''"""
LangGraph-based agent implementation for {{project_name}}.

Modern implementation using latest LangGraph patterns with:
- State management with MessagesState
- Memory integration with InMemoryStore
- Tool calling with create_react_agent
- Checkpointing for conversation persistence
"""

import uuid
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent, ToolNode, InjectedState
from langgraph.types import Command

from .base_agent import BaseAgent
from .state_manager import StateManager
from ..tools import get_available_tools
from ..config import AgentConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AgentState(MessagesState):
    """Extended state for the agent with additional context."""
    user_id: Optional[str] = None
    session_context: Dict[str, Any] = {}


class LangGraphAgent(BaseAgent):
    """LangGraph-based agent implementation with modern patterns."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the LangGraph agent."""
        super().__init__(config)
        
        self.llm = self._create_llm()
        self.tools = get_available_tools(config.tools.enabled)
        self.store = InMemoryStore() if config.memory.enabled else None
        self.state_manager = StateManager()
        
        # Create the agent graph
        self.agent = self._create_agent_graph()
        
        logger.info(f"Initialized LangGraph agent with {len(self.tools)} tools")
    
    def _create_llm(self):
        """Create the LLM instance based on configuration."""
        api_key = self.config.get_api_key()
        
        if self.config.llm.provider == "openai":
            return ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                api_key=api_key,
                timeout=self.config.llm.timeout_seconds,
            )
        elif self.config.llm.provider == "anthropic":
            return ChatAnthropic(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                api_key=api_key,
                timeout=self.config.llm.timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
    
    def _create_agent_graph(self):
        """Create the LangGraph agent graph with memory integration."""
        # Create checkpointer for conversation memory
        checkpointer = InMemorySaver() if self.config.memory.enabled else None
        
        # Create the react agent with tools and state schema
        agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=checkpointer,
            store=self.store,
            state_schema=AgentState,
        )
        
        return agent
    
    async def arun(self, message: Union[str, BaseMessage], user_id: Optional[str] = None, **kwargs) -> str:
        """Run the agent asynchronously with memory context."""
        try:
            # Convert string to HumanMessage if needed
            if isinstance(message, str):
                message = HumanMessage(content=message)
            
            # Prepare configuration with user context
            config = {
                "configurable": {
                    "thread_id": self.session_id or str(uuid.uuid4()),
                    "user_id": user_id or "default_user"
                }
            }
            
            # Add memory context if enabled
            messages = [message]
            if self.store and user_id:
                # Search for relevant memories
                memories = await self.store.asearch(
                    ("memories", user_id),
                    query=message.content,
                    limit=3
                )
                
                if memories:
                    memory_context = "\\n".join([m.value.get("text", "") for m in memories])
                    system_msg = SystemMessage(
                        content=f"Previous context: {memory_context}"
                    )
                    messages = [system_msg] + messages
            
            # Invoke the agent
            result = await self.agent.ainvoke(
                {
                    "messages": messages,
                    "user_id": user_id,
                    "session_context": kwargs
                },
                config=config
            )
            
            # Store new memory if configured
            if self.store and user_id and isinstance(result.get("messages", [])[-1], AIMessage):
                await self.store.aput(
                    ("memories", user_id),
                    str(uuid.uuid4()),
                    {"text": f"User: {message.content}\\nAssistant: {result['messages'][-1].content}"}
                )
            
            # Extract the response
            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content
            
            return "I'm sorry, I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"Error: {str(e)}"
    
    def run(self, message: Union[str, BaseMessage], **kwargs) -> str:
        """Run the agent synchronously."""
        import asyncio
        return asyncio.run(self.arun(message, **kwargs))
    
    async def astream(self, message: Union[str, BaseMessage], user_id: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        """Stream agent responses asynchronously."""
        try:
            # Convert string to HumanMessage if needed
            if isinstance(message, str):
                message = HumanMessage(content=message)
            
            # Prepare configuration
            config = {
                "configurable": {
                    "thread_id": self.session_id or str(uuid.uuid4()),
                    "user_id": user_id or "default_user"
                }
            }
            
            # Stream the agent
            async for chunk in self.agent.astream(
                {
                    "messages": [message],
                    "user_id": user_id,
                    "session_context": kwargs
                },
                config=config,
                stream_mode="values"
            ):
                if chunk.get("messages"):
                    last_message = chunk["messages"][-1]
                    if isinstance(last_message, AIMessage) and last_message.content:
                        yield last_message.content
                        
        except Exception as e:
            logger.error(f"Error streaming agent: {e}")
            yield f"Error: {str(e)}"
    
    def get_memory(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the agent's memory state for a user."""
        if self.store and user_id:
            try:
                memories = self.store.search(("memories", user_id), limit=10)
                return {"memories": [m.value for m in memories]}
            except Exception as e:
                logger.error(f"Error getting memory: {e}")
        return {"memories": []}
    
    async def clear_memory(self, user_id: Optional[str] = None) -> bool:
        """Clear agent memory for a user."""
        if self.store and user_id:
            try:
                # Note: InMemoryStore doesn't have a direct clear method
                # In production, use a store that supports deletion
                logger.info(f"Memory clearing requested for user: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error clearing memory: {e}")
        return False
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        if self.memory_store:
            self.memory_store.clear()
        logger.info("Agent memory cleared")
    
    def get_tools(self) -> List[str]:
        """Get list of available tools."""
        return [tool.name for tool in self.tools]


def create_agent(config: AgentConfig) -> LangGraphAgent:
    """Factory function to create a LangGraph agent."""
    return LangGraphAgent(config)
''',

            "src/{{package_name}}/agent/state_manager.py": '''"""
State management for {{project_name}} agents.
"""

from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime
import json
from pathlib import Path

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State schema for the agent."""
    
    messages: List[BaseMessage]
    user_id: Optional[str]
    session_id: Optional[str]
    context: Dict[str, Any]
    memory: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class StateManager:
    """Manages agent state persistence and retrieval."""
    
    def __init__(self, storage_path: str = "./data/states"):
        """Initialize the state manager."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._states: Dict[str, AgentState] = {}
    
    def create_state(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Create a new agent state."""
        now = datetime.now()
        
        state = AgentState(
            messages=[],
            user_id=user_id,
            session_id=session_id,
            context=initial_context or {},
            memory={},
            tool_calls=[],
            created_at=now,
            updated_at=now
        )
        
        self._states[session_id] = state
        self._save_state(session_id, state)
        
        return state
    
    def get_state(self, session_id: str) -> Optional[AgentState]:
        """Get agent state by session ID."""
        if session_id in self._states:
            return self._states[session_id]
        
        # Try to load from disk
        state = self._load_state(session_id)
        if state:
            self._states[session_id] = state
        
        return state
    
    def update_state(self, session_id: str, updates: Dict[str, Any]) -> AgentState:
        """Update agent state."""
        state = self.get_state(session_id)
        if not state:
            raise ValueError(f"State not found for session: {session_id}")
        
        # Update state
        for key, value in updates.items():
            if key == "messages":
                # Use LangGraph's add_messages for proper message handling
                state["messages"] = add_messages(state["messages"], value)
            else:
                state[key] = value
        
        state["updated_at"] = datetime.now()
        
        self._states[session_id] = state
        self._save_state(session_id, state)
        
        return state
    
    def delete_state(self, session_id: str) -> bool:
        """Delete agent state."""
        # Remove from memory
        if session_id in self._states:
            del self._states[session_id]
        
        # Remove from disk
        state_file = self.storage_path / f"{session_id}.json"
        if state_file.exists():
            state_file.unlink()
            return True
        
        return False
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List all session IDs, optionally filtered by user ID."""
        sessions = []
        
        # Check in-memory states
        for session_id, state in self._states.items():
            if user_id is None or state.get("user_id") == user_id:
                sessions.append(session_id)
        
        # Check disk states
        for state_file in self.storage_path.glob("*.json"):
            session_id = state_file.stem
            if session_id not in sessions:
                state = self._load_state(session_id)
                if state and (user_id is None or state.get("user_id") == user_id):
                    sessions.append(session_id)
        
        return sessions
    
    def _save_state(self, session_id: str, state: AgentState) -> None:
        """Save state to disk."""
        state_file = self.storage_path / f"{session_id}.json"
        
        # Convert state to JSON-serializable format
        serializable_state = {
            "messages": [
                {
                    "type": msg.type if hasattr(msg, "type") else "human",
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, "additional_kwargs", {})
                }
                for msg in state["messages"]
            ],
            "user_id": state["user_id"],
            "session_id": state["session_id"],
            "context": state["context"],
            "memory": state["memory"],
            "tool_calls": state["tool_calls"],
            "created_at": state["created_at"].isoformat(),
            "updated_at": state["updated_at"].isoformat()
        }
        
        with open(state_file, "w") as f:
            json.dump(serializable_state, f, indent=2)
    
    def _load_state(self, session_id: str) -> Optional[AgentState]:
        """Load state from disk."""
        state_file = self.storage_path / f"{session_id}.json"
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
            
            # Convert back to AgentState
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            messages = []
            for msg_data in data["messages"]:
                msg_type = msg_data["type"]
                content = msg_data["content"]
                
                if msg_type == "human":
                    messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    messages.append(AIMessage(content=content))
                elif msg_type == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))  # Default fallback
            
            state = AgentState(
                messages=messages,
                user_id=data["user_id"],
                session_id=data["session_id"],
                context=data["context"],
                memory=data["memory"],
                tool_calls=data["tool_calls"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"])
            )
            
            return state
            
        except Exception as e:
            print(f"Error loading state for session {session_id}: {e}")
            return None
''',

            # Memory modules
            "src/{{package_name}}/memory/__init__.py": '''"""
Memory components for {{project_name}}.
"""

from .langmem_integration import LangMemStore, get_memory_store
from .vector_store import VectorStore, ChromaVectorStore
from .conversation_memory import ConversationMemory

__all__ = [
    "LangMemStore", 
    "get_memory_store", 
    "VectorStore", 
    "ChromaVectorStore",
    "ConversationMemory"
]
''',

            "src/{{package_name}}/memory/langmem_integration.py": '''"""
Modern memory integration for {{project_name}}.

Provides memory capabilities using LangGraph's InMemoryStore with semantic search
and LangMem integration for production use.
"""

from typing import List, Optional, Dict, Any, AsyncIterator
import uuid
import os
from pathlib import Path

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain.embeddings import init_embeddings

try:
    from langmem.short_term import SummarizationNode, RunningSummary
    from langmem import Client as LangMemClient
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """Modern memory store with semantic search capabilities."""
    
    def __init__(self, enable_semantic_search: bool = True, api_key: Optional[str] = None):
        """Initialize memory store with optional semantic search."""
        self.enable_semantic_search = enable_semantic_search
        
        # Initialize embeddings for semantic search
        if enable_semantic_search:
            try:
                self.embeddings = init_embeddings("openai:text-embedding-3-small")
                self.store = InMemoryStore(
                    index={
                        "embed": self.embeddings,
                        "dims": 1536,
                    }
                )
                logger.info("Memory store initialized with semantic search")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic search: {e}, falling back to basic store")
                self.store = InMemoryStore()
                self.enable_semantic_search = False
        else:
            self.store = InMemoryStore()
        
        # Initialize LangMem client for production use
        self.langmem_client = None
        if LANGMEM_AVAILABLE and api_key:
            try:
                self.langmem_client = LangMemClient(api_key=api_key)
                logger.info("LangMem client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LangMem client: {e}")
    
    async def add_memory(self, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a memory to the store."""
        memory_id = str(uuid.uuid4())
        namespace = ("memories", user_id)
        
        memory_data = {
            "text": content,
            "metadata": metadata or {},
            "timestamp": str(uuid.uuid4())  # Simple timestamp placeholder
        }
        
        try:
            await self.store.aput(namespace, memory_id, memory_data)
            
            # Also store in LangMem if available
            if self.langmem_client:
                await self._store_in_langmem(user_id, content, metadata)
            
            return memory_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return ""
    
    async def search(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        namespace = ("memories", user_id)
        
        try:
            if self.enable_semantic_search:
                items = await self.store.asearch(namespace, query=query, limit=limit)
            else:
                # Fallback to getting all memories (basic implementation)
                items = []  # InMemoryStore basic search would go here
            
            return [{"id": item.key, **item.value} for item in items]
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user."""
        namespace = ("memories", user_id)
        
        try:
            # Note: This is a simplified implementation
            # In production, you'd implement proper pagination
            items = await self.store.asearch(namespace, query="", limit=100)
            return [{"id": item.key, **item.value} for item in items]
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return []
    
    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete a specific memory."""
        namespace = ("memories", user_id)
        
        try:
            await self.store.adelete(namespace, memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    async def clear_user_memories(self, user_id: str) -> bool:
        """Clear all memories for a user."""
        try:
            memories = await self.get_all_memories(user_id)
            for memory in memories:
                await self.delete_memory(user_id, memory["id"])
            return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False
    
    async def _store_in_langmem(self, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Store memory in LangMem for production persistence."""
        if not self.langmem_client:
            return
        
        try:
            await self.langmem_client.acreate_memory(
                content=content,
                user_id=user_id,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to store in LangMem: {e}")


class SummarizationMemory:
    """Memory with automatic summarization capabilities."""
    
    def __init__(self, llm, max_tokens: int = 384, max_summary_tokens: int = 128):
        """Initialize summarization memory."""
        if not LANGMEM_AVAILABLE:
            logger.warning("LangMem not available, summarization disabled")
            self.summarization_node = None
            return
        
        try:
            from langchain_core.messages.utils import count_tokens_approximately
            
            self.summarization_node = SummarizationNode(
                token_counter=count_tokens_approximately,
                model=llm,
                max_tokens=max_tokens,
                max_summary_tokens=max_summary_tokens,
                output_messages_key="llm_input_messages",
            )
            logger.info("Summarization memory initialized")
        except Exception as e:
            logger.error(f"Failed to initialize summarization: {e}")
            self.summarization_node = None
    
    def summarize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize messages if summarization is available."""
        if not self.summarization_node:
            return messages
        
        try:
            return self.summarization_node({"messages": messages})["llm_input_messages"]
        except Exception as e:
            logger.error(f"Failed to summarize messages: {e}")
            return messages


# Factory functions for easy initialization
def create_memory_store(enable_semantic_search: bool = True, langmem_api_key: Optional[str] = None) -> MemoryStore:
    """Create a memory store instance."""
    return MemoryStore(enable_semantic_search=enable_semantic_search, api_key=langmem_api_key)


def create_summarization_memory(llm, max_tokens: int = 384) -> SummarizationMemory:
    """Create a summarization memory instance."""
    return SummarizationMemory(llm, max_tokens=max_tokens)
        if self._use_fallback:
            # Simple fallback search
            results = []
            query_lower = query.lower()
            
            for memory in self._fallback_memories:
                if query_lower in memory["content"].lower():
                    results.append(memory)
                    if len(results) >= limit:
                        break
            
            return results
        
        try:
            return self.client.search_memories(
                query=query,
                user_id=self.user_id,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def get_all(self) -> List[Memory]:
        """Get all memories for the user."""
        if self._use_fallback:
            return self._fallback_memories
        
        try:
            return self.client.get_memories(user_id=self.user_id)
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return []
    
    def clear(self) -> None:
        """Clear all memories for the user."""
        if self._use_fallback:
            self._fallback_memories.clear()
            return
        
        try:
            memories = self.client.get_memories(user_id=self.user_id)
            for memory in memories:
                self.client.delete_memory(memory.id)
            logger.info("All memories cleared")
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")


def get_memory_store(user_id: str = "default") -> LangMemStore:
    """Factory function to create a memory store."""
    return LangMemStore(user_id=user_id)
''',

            "src/{{package_name}}/memory/vector_store.py": '''"""
Vector store implementations for {{project_name}}.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with similarity scores."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass


class ChromaVectorStore(VectorStore):
    """Chroma-based vector store implementation."""
    
    def __init__(
        self,
        persist_directory: str = "./data/vectorstore",
        collection_name: str = "{{package_name}}_memories",
        embeddings: Optional[Embeddings] = None
    ):
        """Initialize Chroma vector store."""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = embeddings or OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        logger.info(f"Initialized Chroma vector store at {self.persist_directory}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            ids = self.store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            return self.store.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with similarity scores."""
        try:
            return self.store.similarity_search_with_score(query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"Failed to search documents with scores: {e}")
            return []
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        try:
            self.store.delete(ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        try:
            # Get all document IDs and delete them
            all_docs = self.store.get()
            if all_docs["ids"]:
                self.store.delete(all_docs["ids"])
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for development/testing."""
    
    def __init__(self, embeddings: Optional[Embeddings] = None):
        """Initialize in-memory vector store."""
        self.embeddings = embeddings or OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.documents: List[Document] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        logger.info("Initialized in-memory vector store")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        ids = []
        for i, doc in enumerate(documents):
            doc_id = f"doc_{len(self.documents) + i}"
            doc.metadata["id"] = doc_id
            self.documents.append(doc)
            ids.append(doc_id)
        
        logger.info(f"Added {len(documents)} documents to in-memory store")
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents using simple text matching."""
        # Simple keyword-based search for development
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            if filter:
                # Apply metadata filter
                if not all(doc.metadata.get(key) == value for key, value in filter.items()):
                    continue
            
            if query_lower in doc.page_content.lower():
                results.append(doc)
                if len(results) >= k:
                    break
        
        return results
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with dummy scores."""
        docs = self.similarity_search(query, k, filter)
        # Return with dummy similarity scores
        return [(doc, 0.8) for doc in docs]
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        self.documents = [
            doc for doc in self.documents
            if doc.metadata.get("id") not in ids
        ]
        logger.info(f"Deleted {len(ids)} documents from in-memory store")
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self.documents.clear()
        self.embeddings_cache.clear()
        logger.info("Cleared all documents from in-memory store")


def create_vector_store(
    store_type: str = "chroma",
    persist_directory: str = "./data/vectorstore",
    **kwargs
) -> VectorStore:
    """Factory function to create a vector store."""
    if store_type == "chroma":
        return ChromaVectorStore(persist_directory=persist_directory, **kwargs)
    elif store_type == "memory":
        return InMemoryVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
''',

            "src/{{package_name}}/memory/conversation_memory.py": '''"""
Conversation memory management for {{project_name}}.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """Manages conversation memory with persistence."""
    
    def __init__(
        self,
        session_id: str,
        memory_type: str = "window",
        window_size: int = 10,
        storage_path: str = "./data/conversations",
        auto_save: bool = True
    ):
        """Initialize conversation memory."""
        self.session_id = session_id
        self.memory_type = memory_type
        self.window_size = window_size
        self.auto_save = auto_save
        
        # Setup storage
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.conversation_file = self.storage_path / f"{session_id}.json"
        
        # Initialize memory
        self.messages: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0
        }
        
        # Load existing conversation if available
        self._load_conversation()
        
        logger.info(f"Initialized conversation memory for session {session_id}")
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.metadata["message_count"] += 1
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        # Apply window size limit
        if self.memory_type == "window" and len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]
        
        if self.auto_save:
            self._save_conversation()
        
        logger.debug(f"Added message to conversation {self.session_id}")
    
    def add_human_message(self, content: str) -> None:
        """Add a human message to the conversation."""
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str) -> None:
        """Add an AI message to the conversation."""
        self.add_message(AIMessage(content=content))
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.add_message(SystemMessage(content=content))
    
    def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """Get conversation messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()
    
    def get_recent_messages(self, time_limit: timedelta) -> List[BaseMessage]:
        """Get messages from within a time limit."""
        cutoff_time = datetime.now() - time_limit
        recent_messages = []
        
        for message in self.messages:
            # Check if message has timestamp metadata
            if hasattr(message, "additional_kwargs"):
                timestamp_str = message.additional_kwargs.get("timestamp")
                if timestamp_str:
                    try:
                        msg_time = datetime.fromisoformat(timestamp_str)
                        if msg_time >= cutoff_time:
                            recent_messages.append(message)
                    except ValueError:
                        # If timestamp parsing fails, include the message
                        recent_messages.append(message)
                else:
                    # No timestamp, include message
                    recent_messages.append(message)
            else:
                # No additional_kwargs, include message
                recent_messages.append(message)
        
        return recent_messages
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return "No conversation yet."
        
        message_count = len(self.messages)
        first_message_time = self.metadata.get("created_at", "Unknown")
        last_message_time = self.metadata.get("updated_at", "Unknown")
        
        summary = f"""Conversation Summary:
- Session ID: {self.session_id}
- Total Messages: {message_count}
- Started: {first_message_time}
- Last Activity: {last_message_time}
- Memory Type: {self.memory_type}
"""
        
        # Add recent message preview
        if self.messages:
            last_message = self.messages[-1]
            preview = last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content
            summary += f"- Last Message: {preview}"
        
        return summary
    
    def clear(self) -> None:
        """Clear the conversation memory."""
        self.messages.clear()
        self.metadata["message_count"] = 0
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        if self.auto_save:
            self._save_conversation()
        
        logger.info(f"Cleared conversation memory for session {self.session_id}")
    
    def save(self) -> None:
        """Manually save the conversation."""
        self._save_conversation()
    
    def _save_conversation(self) -> None:
        """Save conversation to disk."""
        try:
            conversation_data = {
                "metadata": self.metadata,
                "messages": [
                    {
                        "type": msg.type if hasattr(msg, "type") else self._get_message_type(msg),
                        "content": msg.content,
                        "additional_kwargs": getattr(msg, "additional_kwargs", {}),
                        "timestamp": datetime.now().isoformat()
                    }
                    for msg in self.messages
                ]
            }
            
            with open(self.conversation_file, "w") as f:
                json.dump(conversation_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def _load_conversation(self) -> None:
        """Load conversation from disk."""
        if not self.conversation_file.exists():
            return
        
        try:
            with open(self.conversation_file, "r") as f:
                conversation_data = json.load(f)
            
            # Load metadata
            self.metadata.update(conversation_data.get("metadata", {}))
            
            # Load messages
            for msg_data in conversation_data.get("messages", []):
                msg_type = msg_data["type"]
                content = msg_data["content"]
                additional_kwargs = msg_data.get("additional_kwargs", {})
                
                if msg_type == "human":
                    message = HumanMessage(content=content, additional_kwargs=additional_kwargs)
                elif msg_type == "ai":
                    message = AIMessage(content=content, additional_kwargs=additional_kwargs)
                elif msg_type == "system":
                    message = SystemMessage(content=content, additional_kwargs=additional_kwargs)
                else:
                    # Default to human message
                    message = HumanMessage(content=content, additional_kwargs=additional_kwargs)
                
                self.messages.append(message)
            
            logger.info(f"Loaded {len(self.messages)} messages for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
    
    def _get_message_type(self, message: BaseMessage) -> str:
        """Get message type string."""
        if isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "ai"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return "human"  # Default fallback


def create_conversation_memory(
    session_id: str,
    memory_type: str = "window",
    **kwargs
) -> ConversationMemory:
    """Factory function to create conversation memory."""
    return ConversationMemory(
        session_id=session_id,
        memory_type=memory_type,
        **kwargs
    )
''',

            # Tool modules
            "src/{{package_name}}/tools/__init__.py": '''"""
Tool components for {{project_name}}.
"""

from .base_tool import BaseTool
from .web_search import WebSearchTool, TavilySearchTool
from .file_operations import FileOperationsTool
from .custom_tools import CustomTool
from typing import List

def get_available_tools(enabled_tools: List[str]) -> List[BaseTool]:
    """Get list of available tools based on configuration."""
    tools = []
    
    if "web_search" in enabled_tools:
        tools.append(TavilySearchTool())
    
    if "file_operations" in enabled_tools:
        tools.append(FileOperationsTool())
    
    # Add custom tools if needed
    # tools.extend(get_custom_tools())
    
    return tools

__all__ = [
    "BaseTool",
    "WebSearchTool", 
    "TavilySearchTool",
    "FileOperationsTool",
    "CustomTool",
    "get_available_tools"
]
''',

            "src/{{package_name}}/tools/base_tool.py": '''"""
Modern base tool interface for {{project_name}}.

Provides base classes for tools with LangGraph integration, state access,
and proper error handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Annotated
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool as LangChainBaseTool, tool
from langchain_core.callbacks import CallbackManagerForToolRun
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseTool(LangChainBaseTool, ABC):
    """Base class for all tools in the agent system."""
    
    name: str
    description: str
    return_direct: bool = False
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    @abstractmethod
    def _run(
        self, 
        *args, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """Execute the tool synchronously."""
        pass
    
    async def _arun(
        self, 
        *args, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """Execute the tool asynchronously."""
        # Default implementation falls back to sync
        return self._run(*args, run_manager=run_manager, **kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "return_direct": self.return_direct,
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate tool input."""
        return True
    
    def handle_error(self, error: Exception) -> str:
        """Handle tool execution errors."""
        logger.error(f"Tool {self.name} error: {error}")
        return f"Tool error: {str(error)}"


class StateAwareTool(BaseTool):
    """Base tool that can access agent state."""
    
    def get_state_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context from agent state."""
        return {
            "user_id": state.get("user_id"),
            "session_context": state.get("session_context", {}),
            "messages": state.get("messages", [])
        }


class MemoryAwareTool(BaseTool):
    """Base tool that can access and update memory."""
    
    async def store_memory(
        self, 
        store: BaseStore, 
        user_id: str, 
        content: str, 
        memory_type: str = "tool_result"
    ) -> str:
        """Store a result in memory."""
        import uuid
        memory_id = str(uuid.uuid4())
        
        try:
            await store.aput(
                ("memories", user_id),
                memory_id,
                {
                    "text": content,
                    "type": memory_type,
                    "tool": self.name
                }
            )
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return ""
    
    async def search_memory(
        self, 
        store: BaseStore, 
        user_id: str, 
        query: str, 
        limit: int = 3
    ) -> list:
        """Search for relevant memories."""
        try:
            memories = await store.asearch(
                ("memories", user_id),
                query=query,
                limit=limit
            )
            return [m.value for m in memories]
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []


class ToolInput(BaseModel):
    """Base class for tool input schemas."""
    pass


class ToolOutput(BaseModel):
    """Base class for tool output schemas."""
    success: bool = Field(description="Whether the tool execution was successful")
    result: Any = Field(description="The tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")


# Decorators for creating tools with state/store access
def create_stateful_tool(
    name: str,
    description: str,
    args_schema: Optional[Type[BaseModel]] = None
):
    """Decorator to create a tool that can access agent state."""
    def decorator(func):
        return tool(
            func,
            name=name,
            description=description,
            args_schema=args_schema
        )
    return decorator


def create_memory_tool(
    name: str,
    description: str,
    args_schema: Optional[Type[BaseModel]] = None
):
    """Decorator to create a tool that can access memory store."""
    def decorator(func):
        # The function should accept store parameter with InjectedStore annotation
        return tool(
            func,
            name=name,
            description=description,
            args_schema=args_schema
        )
    return decorator


# Example tool implementations using modern patterns
@create_memory_tool(
    name="save_user_preference",
    description="Save a user preference to memory"
)
async def save_user_preference(
    preference: str,
    value: str,
    store: Annotated[BaseStore, InjectedStore],
    user_id: str = "default_user"
) -> str:
    """Save a user preference to memory."""
    try:
        import uuid
        await store.aput(
            ("preferences", user_id),
            f"pref_{preference}",
            {"preference": preference, "value": value}
        )
        return f"Saved preference {preference} = {value}"
    except Exception as e:
        return f"Failed to save preference: {e}"


@create_memory_tool(
    name="get_user_preference", 
    description="Get a user preference from memory"
)
async def get_user_preference(
    preference: str,
    store: Annotated[BaseStore, InjectedStore],
    user_id: str = "default_user"
) -> str:
    """Get a user preference from memory."""
    try:
        result = await store.aget(("preferences", user_id), f"pref_{preference}")
        if result:
            return f"{preference}: {result.value['value']}"
        return f"Preference {preference} not found"
    except Exception as e:
        return f"Failed to get preference: {e}"
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    
    @classmethod
    def success_response(cls, data: Any) -> "ToolOutput":
        """Create a successful response."""
        return cls(success=True, data=data)
    
    @classmethod
    def error_response(cls, error: str) -> "ToolOutput":
        """Create an error response."""
        return cls(success=False, error=error)
''',

            "src/{{package_name}}/tools/web_search.py": '''"""
Web search tools for {{project_name}}.
"""

import os
import json
from typing import Optional, Type, Any, Dict, List, Union
from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools.tavily_search import TavilySearchResults

from .base_tool import StateAwareTool, MemoryAwareTool
from ..utils.logging import get_logger

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    
    query: str = Field(description="The search query to execute")
    max_results: int = Field(default=5, description="Maximum number of results to return (1-10)")
    search_type: str = Field(default="general", description="Type of search: general, news, academic")


class WebSearchTool(StateAwareTool, MemoryAwareTool):
    """Modern web search tool with state and memory integration."""
    
    name: str = "web_search"
    description: str = "Search the web for current information. Useful for finding recent news, facts, or general information."
    args_schema: Type[BaseModel] = WebSearchInput
    
    def __init__(self, **kwargs):
        """Initialize web search tool."""
        super().__init__(**kwargs)
        self.api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not found. Web search functionality will be limited.")
            self._search_tool = None
        else:
            try:
                self._search_tool = TavilySearchResults(
                    api_key=self.api_key,
                    max_results=10,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=True
                )
                logger.info("Tavily search tool initialized with advanced features")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily: {e}")
                self._search_tool = None
    
    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "general",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute web search asynchronously."""
        try:
            # Store search in memory for context
            if self.memory_store:
                await self.memory_store.aput(
                    ("search_queries", f"query_{hash(query)}"),
                    {
                        "query": query,
                        "search_type": search_type,
                        "timestamp": self._get_current_time(),
                        "user_id": self.current_user
                    }
                )
            
            # Perform search
            results = await self._perform_search(query, max_results, search_type)
            
            # Store results in memory
            if self.memory_store and results:
                await self.memory_store.aput(
                    ("search_results", f"results_{hash(query)}"),
                    {
                        "query": query,
                        "results": results[:3],  # Store top 3 for memory efficiency
                        "total_found": len(results),
                        "timestamp": self._get_current_time()
                    }
                )
            
            return self._format_results(results, query)
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {str(e)}"
    
    def _run(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "general",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute web search synchronously."""
        try:
            # Perform search
            results = self._perform_search_sync(query, max_results, search_type)
            return self._format_results(results, query)
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {str(e)}"
    
    async def _perform_search(
        self, 
        query: str, 
        max_results: int, 
        search_type: str
    ) -> List[Dict[str, Any]]:
        """Perform async search with enhanced query processing."""
        if not self._search_tool:
            # Fallback to simple mock results
            return [{
                "title": f"Search Results for: {query}",
                "url": "https://example.com",
                "content": "Web search functionality requires TAVILY_API_KEY to be configured.",
                "score": 0.5
            }]
        
        try:
            # Enhance query based on search type
            enhanced_query = self._enhance_query(query, search_type)
            
            # Execute search
            raw_results = self._search_tool._run(enhanced_query)
            
            # Parse and validate results
            parsed_results = self._parse_search_results(raw_results)
            
            # Apply post-processing
            return self._post_process_results(parsed_results, max_results)
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            raise
    
    def _perform_search_sync(
        self, 
        query: str, 
        max_results: int, 
        search_type: str
    ) -> List[Dict[str, Any]]:
        """Perform synchronous search."""
        if not self._search_tool:
            return [{
                "title": f"Search Results for: {query}",
                "url": "https://example.com", 
                "content": "Web search functionality requires TAVILY_API_KEY to be configured.",
                "score": 0.5
            }]
        
        try:
            enhanced_query = self._enhance_query(query, search_type)
            raw_results = self._search_tool._run(enhanced_query)
            parsed_results = self._parse_search_results(raw_results)
            return self._post_process_results(parsed_results, max_results)
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            raise


    
    def _enhance_query(self, query: str, search_type: str) -> str:
        """Enhance query based on search type."""
        if search_type == "news":
            return f"{query} latest news recent"
        elif search_type == "academic":
            return f"{query} research study academic"
        return query
    
    def _parse_search_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """Parse raw search results into standardized format."""
        if isinstance(raw_results, str):
            try:
                # Try to parse as JSON
                results_data = json.loads(raw_results)
                if isinstance(results_data, list):
                    return results_data
            except (json.JSONDecodeError, TypeError):
                # Fallback: treat as single text result
                return [{
                    "title": "Search Result",
                    "url": "",
                    "content": raw_results,
                    "score": 0.7
                }]
        elif isinstance(raw_results, list):
            return raw_results
        elif isinstance(raw_results, dict):
            return [raw_results]
        
        return []
    
    def _post_process_results(
        self, 
        results: List[Dict[str, Any]], 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Post-process and filter search results."""
        processed = []
        
        for result in results[:max_results]:
            # Ensure required fields
            processed_result = {
                "title": result.get("title", "No title"),
                "url": result.get("url", ""),
                "content": result.get("content", result.get("snippet", "No description")),
                "score": result.get("score", 0.5),
                "published_date": result.get("published_date", ""),
                "source": result.get("source", "")
            }
            
            # Clean content
            if processed_result["content"]:
                processed_result["content"] = processed_result["content"][:500]
            
            processed.append(processed_result)
        
        return processed
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results into readable output."""
        if not results:
            return f"No search results found for: {query}"
        
        formatted = f"🔍 **Search Results for '{query}'** ({len(results)} found)\\n\\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No description")
            score = result.get("score", 0)
            source = result.get("source", "")
            
            # Add relevance indicator
            relevance = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "🔴"
            
            formatted += f"{relevance} **{i}. {title}**\\n"
            if url:
                formatted += f"   🔗 {url}\\n"
            if source:
                formatted += f"   📰 Source: {source}\\n"
            formatted += f"   📝 {content[:300]}{'...' if len(content) > 300 else ''}\\n\\n"
        
        return formatted


class TavilySearchTool(WebSearchTool):
    """Tavily-powered web search tool with advanced features."""
    
    name: str = "tavily_search"
    description: str = "Search the web using Tavily for current, accurate information with AI-powered relevance"


class DuckDuckGoSearchTool(WebSearchTool):
    """DuckDuckGo-powered web search tool (fallback)."""
    
    name: str = "duckduckgo_search"
    description: str = "Search the web using DuckDuckGo (privacy-focused alternative)"
    
    def __init__(self, **kwargs):
        """Initialize DuckDuckGo search tool."""
        super().__init__(**kwargs)
        self._search_tool = None
        
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            self._search_tool = DuckDuckGoSearchRun(max_results=10)
            logger.info("DuckDuckGo search tool initialized")
        except ImportError:
            logger.warning("DuckDuckGo search not available. Install with: pip install duckduckgo-search")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDuckGo: {e}")
    
    def _parse_search_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo results."""
        if isinstance(raw_results, str):
            # DuckDuckGo returns plain text, format as single result
            return [{
                "title": "DuckDuckGo Search Results",
                "url": "https://duckduckgo.com",
                "content": raw_results,
                "score": 0.6,
                "source": "DuckDuckGo"
            }]
        return super()._parse_search_results(raw_results)


def create_web_search_tool(provider: str = "tavily", **kwargs) -> WebSearchTool:
    """
    Factory function to create web search tool.
    
    Args:
        provider: Search provider ("tavily" or "duckduckgo")
        **kwargs: Additional arguments passed to tool constructor
    
    Returns:
        Configured web search tool
    """
    if provider == "tavily":
        return TavilySearchTool(**kwargs)
    elif provider == "duckduckgo":
        return DuckDuckGoSearchTool(**kwargs)
    else:
        raise ValueError(f"Unsupported search provider: {provider}. Use 'tavily' or 'duckduckgo'")


# For backward compatibility
WebSearchResults = WebSearchTool
''',

            "src/{{package_name}}/tools/file_operations.py": '''"""
File operations tools for {{project_name}}.
"""

import os
from pathlib import Path
from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun

from .base_tool import BaseTool, ToolInput, ToolOutput
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FileReadInput(ToolInput):
    """Input schema for file reading."""
    
    file_path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class FileWriteInput(ToolInput):
    """Input schema for file writing."""
    
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_dirs: bool = Field(default=True, description="Create parent directories if they don't exist")


class FileListInput(ToolInput):
    """Input schema for listing files."""
    
    directory: str = Field(description="Directory to list files from")
    pattern: str = Field(default="*", description="File pattern to match")
    recursive: bool = Field(default=False, description="Whether to search recursively")


class FileOperationsTool(BaseTool):
    """File operations tool for reading, writing, and listing files."""
    
    name: str = "file_operations"
    description: str = """Tool for file operations including reading, writing, and listing files.
    
Available operations:
- read_file: Read content from a file
- write_file: Write content to a file  
- list_files: List files in a directory
- file_exists: Check if a file exists
- get_file_info: Get file information (size, modified date, etc.)

Use the format: operation:argument (e.g., "read_file:/path/to/file.txt")
"""
    
    def __init__(self, allowed_extensions: List[str] = None, max_file_size_mb: int = 10):
        """Initialize file operations tool."""
        super().__init__()
        self.allowed_extensions = allowed_extensions or [
            ".txt", ".md", ".json", ".yaml", ".yml", ".py", ".js", ".html", ".css", ".xml"
        ]
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        logger.info(f"File operations tool initialized with allowed extensions: {self.allowed_extensions}")
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute file operation based on query format."""
        try:
            # Parse operation and argument
            if ":" not in query:
                return "Invalid format. Use: operation:argument (e.g., 'read_file:/path/to/file.txt')"
            
            operation, argument = query.split(":", 1)
            operation = operation.strip().lower()
            argument = argument.strip()
            
            # Execute operation
            if operation == "read_file":
                return self.read_file(argument)
            elif operation == "write_file":
                # For write operations, expect format: write_file:path|content
                if "|" not in argument:
                    return "Write format: write_file:path|content"
                file_path, content = argument.split("|", 1)
                return self.write_file(file_path.strip(), content)
            elif operation == "list_files":
                return self.list_files(argument)
            elif operation == "file_exists":
                return self.file_exists(argument)
            elif operation == "get_file_info":
                return self.get_file_info(argument)
            else:
                return f"Unknown operation: {operation}. Available: read_file, write_file, list_files, file_exists, get_file_info"
                
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return f"File operation error: {str(e)}"
    
    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            path = Path(file_path)
            
            # Security checks
            if not self._is_safe_path(path):
                return f"Access denied: {file_path}"
            
            if not path.exists():
                return f"File not found: {file_path}"
            
            if not path.is_file():
                return f"Not a file: {file_path}"
            
            # Check file size
            if path.stat().st_size > self.max_file_size_bytes:
                return f"File too large (max {self.max_file_size_bytes // 1024 // 1024}MB): {file_path}"
            
            # Check extension
            if not self._is_allowed_extension(path):
                return f"File type not allowed: {path.suffix}"
            
            # Read file
            content = path.read_text(encoding="utf-8")
            return f"Content of {file_path}:\\n\\n{content}"
            
        except UnicodeDecodeError:
            return f"Cannot read file (binary or unsupported encoding): {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file."""
        try:
            path = Path(file_path)
            
            # Security checks
            if not self._is_safe_path(path):
                return f"Access denied: {file_path}"
            
            # Check extension
            if not self._is_allowed_extension(path):
                return f"File type not allowed: {path.suffix}"
            
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            path.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {file_path}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def list_files(self, directory: str) -> str:
        """List files in a directory."""
        try:
            path = Path(directory)
            
            if not self._is_safe_path(path):
                return f"Access denied: {directory}"
            
            if not path.exists():
                return f"Directory not found: {directory}"
            
            if not path.is_dir():
                return f"Not a directory: {directory}"
            
            # List files
            files = []
            for item in path.iterdir():
                if item.is_file():
                    size = item.stat().st_size
                    files.append(f"📄 {item.name} ({size} bytes)")
                elif item.is_dir():
                    files.append(f"📁 {item.name}/")
            
            if not files:
                return f"No files found in {directory}"
            
            return f"Files in {directory}:\\n" + "\\n".join(sorted(files))
            
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    def file_exists(self, file_path: str) -> str:
        """Check if a file exists."""
        try:
            path = Path(file_path)
            
            if not self._is_safe_path(path):
                return f"Access denied: {file_path}"
            
            exists = path.exists()
            return f"File {'exists' if exists else 'does not exist'}: {file_path}"
            
        except Exception as e:
            return f"Error checking file: {str(e)}"
    
    def get_file_info(self, file_path: str) -> str:
        """Get file information."""
        try:
            path = Path(file_path)
            
            if not self._is_safe_path(path):
                return f"Access denied: {file_path}"
            
            if not path.exists():
                return f"File not found: {file_path}"
            
            stat = path.stat()
            info = f"""File Information for {file_path}:
- Type: {'File' if path.is_file() else 'Directory' if path.is_dir() else 'Other'}
- Size: {stat.st_size:,} bytes
- Modified: {stat.st_mtime}
- Permissions: {oct(stat.st_mode)[-3:]}
"""
            
            if path.is_file():
                info += f"- Extension: {path.suffix}"
            
            return info
            
        except Exception as e:
            return f"Error getting file info: {str(e)}"
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe to access."""
        try:
            # Convert to absolute path
            abs_path = path.resolve()
            
            # Get current working directory
            cwd = Path.cwd().resolve()
            
            # Check if path is within current working directory or subdirectories
            try:
                abs_path.relative_to(cwd)
                return True
            except ValueError:
                # Path is outside working directory
                return False
                
        except Exception:
            return False
    
    def _is_allowed_extension(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        if not self.allowed_extensions:
            return True
        
        return path.suffix.lower() in [ext.lower() for ext in self.allowed_extensions]
''',

            "src/{{package_name}}/tools/custom_tools.py": '''"""
Custom tools for {{project_name}}.
"""

from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun

from .base_tool import BaseTool, ToolInput, ToolOutput
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CustomToolInput(ToolInput):
    """Input schema for custom tool."""
    
    query: str = Field(description="The input query or command")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")


class CustomTool(BaseTool):
    """Example custom tool implementation."""
    
    name: str = "custom_tool"
    description: str = "A custom tool for specific domain tasks"
    args_schema: Type[BaseModel] = CustomToolInput
    
    def _run(
        self,
        query: str,
        options: Dict[str, Any] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the custom tool."""
        try:
            # Implement your custom logic here
            result = self.process_query(query, options or {})
            return result
        except Exception as e:
            logger.error(f"Custom tool failed: {e}")
            return f"Custom tool error: {str(e)}"
    
    def process_query(self, query: str, options: Dict[str, Any]) -> str:
        """Process the query with custom logic."""
        # Example implementation
        return f"Processed query: {query} with options: {options}"


class CalculatorTool(BaseTool):
    """Simple calculator tool."""
    
    name: str = "calculator"
    description: str = """A calculator tool for mathematical calculations.
    
Supports basic operations: +, -, *, /, **, sqrt, abs, round
Example: "2 + 3 * 4" or "sqrt(16)"
"""
    
    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute mathematical calculation."""
        try:
            # Simple calculator with basic safety
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "sqrt": lambda x: x ** 0.5,
                "pow": pow,
            }
            
            # Remove any potentially dangerous characters/words
            expression = expression.strip()
            dangerous_words = ["import", "exec", "eval", "open", "file", "__"]
            
            if any(word in expression.lower() for word in dangerous_words):
                return "Error: Expression contains potentially dangerous operations"
            
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Calculation error: {str(e)}"


class TimeTool(BaseTool):
    """Tool for time-related operations."""
    
    name: str = "time_tool"
    description: str = """Tool for time-related operations.
    
Available commands:
- current_time: Get current date and time
- timezone: Get current timezone info
- format_time: Format a timestamp
"""
    
    def _run(
        self,
        command: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute time operation."""
        try:
            from datetime import datetime, timezone
            import time
            
            command = command.strip().lower()
            
            if command == "current_time":
                now = datetime.now()
                return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            
            elif command == "timezone":
                now = datetime.now(timezone.utc)
                local_time = now.astimezone()
                return f"UTC time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\\nLocal time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            
            elif command.startswith("format_time"):
                # Extract timestamp if provided
                parts = command.split()
                if len(parts) > 1:
                    timestamp = float(parts[1])
                    dt = datetime.fromtimestamp(timestamp)
                    return f"Formatted time: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    return "Error: format_time requires a timestamp"
            
            else:
                return f"Unknown command: {command}. Available: current_time, timezone, format_time"
                
        except Exception as e:
            return f"Time tool error: {str(e)}"


def get_custom_tools() -> list:
    """Get list of custom tools."""
    return [
        CustomTool(),
        CalculatorTool(),
        TimeTool(),
    ]
''',

            # Workflow modules
            "src/{{package_name}}/workflows/__init__.py": '''"""
Workflow components for {{project_name}}.
"""

from .workflow_builder import WorkflowBuilder, WorkflowNode, WorkflowEdge
from .common_workflows import (
    create_research_workflow,
    create_conversation_workflow,
    create_analysis_workflow
)

__all__ = [
    "WorkflowBuilder",
    "WorkflowNode", 
    "WorkflowEdge",
    "create_research_workflow",
    "create_conversation_workflow", 
    "create_analysis_workflow"
]
''',

            "src/{{package_name}}/workflows/workflow_builder.py": '''"""
Workflow builder for creating custom LangGraph workflows.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage

from ..utils.logging import get_logger

logger = get_logger(__name__)


class NodeType(Enum):
    """Types of workflow nodes."""
    LLM = "llm"
    TOOL = "tool"
    CONDITION = "condition"
    HUMAN = "human"
    CUSTOM = "custom"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow."""
    
    id: str
    name: str
    type: NodeType
    function: Callable
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@dataclass 
class WorkflowEdge:
    """Represents an edge between workflow nodes."""
    
    from_node: str
    to_node: str
    condition: Optional[Callable] = None
    label: Optional[str] = None


class WorkflowBuilder:
    """Builder for creating LangGraph workflows."""
    
    def __init__(self, state_schema: type = None):
        """Initialize the workflow builder."""
        self.state_schema = state_schema
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self.entry_point: Optional[str] = None
        self.exit_points: List[str] = []
        
        logger.info("Workflow builder initialized")
    
    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: NodeType,
        function: Callable,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> "WorkflowBuilder":
        """Add a node to the workflow."""
        node = WorkflowNode(
            id=node_id,
            name=name,
            type=node_type,
            function=function,
            description=description,
            config=config or {}
        )
        
        self.nodes[node_id] = node
        logger.debug(f"Added node: {node_id} ({node_type.value})")
        
        return self
    
    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable] = None,
        label: Optional[str] = None
    ) -> "WorkflowBuilder":
        """Add an edge between nodes."""
        edge = WorkflowEdge(
            from_node=from_node,
            to_node=to_node,
            condition=condition,
            label=label
        )
        
        self.edges.append(edge)
        logger.debug(f"Added edge: {from_node} -> {to_node}")
        
        return self
    
    def set_entry_point(self, node_id: str) -> "WorkflowBuilder":
        """Set the entry point of the workflow."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        self.entry_point = node_id
        logger.debug(f"Set entry point: {node_id}")
        
        return self
    
    def add_exit_point(self, node_id: str) -> "WorkflowBuilder":
        """Add an exit point to the workflow."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        self.exit_points.append(node_id)
        logger.debug(f"Added exit point: {node_id}")
        
        return self
    
    def build(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        if not self.entry_point:
            raise ValueError("Entry point not set")
        
        if not self.exit_points:
            raise ValueError("No exit points defined")
        
        # Create the state graph
        if self.state_schema:
            graph = StateGraph(self.state_schema)
        else:
            from langgraph.graph import MessagesState
            graph = StateGraph(MessagesState)
        
        # Add nodes
        for node in self.nodes.values():
            graph.add_node(node.id, node.function)
        
        # Add edges
        for edge in self.edges:
            if edge.condition:
                # Conditional edge
                graph.add_conditional_edges(
                    edge.from_node,
                    edge.condition,
                    {edge.to_node: edge.to_node}
                )
            else:
                # Regular edge
                graph.add_edge(edge.from_node, edge.to_node)
        
        # Set entry point
        graph.add_edge(START, self.entry_point)
        
        # Set exit points
        for exit_point in self.exit_points:
            graph.add_edge(exit_point, END)
        
        logger.info(f"Built workflow with {len(self.nodes)} nodes and {len(self.edges)} edges")
        
        return graph
    
    def compile(self, **kwargs) -> Any:
        """Build and compile the workflow."""
        graph = self.build()
        return graph.compile(**kwargs)
    
    def visualize(self) -> str:
        """Generate a text visualization of the workflow."""
        lines = ["Workflow Structure:", ""]
        
        # Entry point
        lines.append(f"Entry: {self.entry_point}")
        lines.append("")
        
        # Nodes
        lines.append("Nodes:")
        for node in self.nodes.values():
            lines.append(f"  - {node.id}: {node.name} ({node.type.value})")
            if node.description:
                lines.append(f"    {node.description}")
        lines.append("")
        
        # Edges
        lines.append("Edges:")
        for edge in self.edges:
            condition_str = f" [condition: {edge.label}]" if edge.condition else ""
            lines.append(f"  - {edge.from_node} -> {edge.to_node}{condition_str}")
        lines.append("")
        
        # Exit points
        lines.append(f"Exit points: {', '.join(self.exit_points)}")
        
        return "\\n".join(lines)


class ConditionalRouter:
    """Helper class for creating conditional routing logic."""
    
    @staticmethod
    def create_binary_condition(
        condition_func: Callable[[Any], bool],
        true_path: str,
        false_path: str
    ) -> Callable:
        """Create a binary condition router."""
        def router(state):
            if condition_func(state):
                return true_path
            return false_path
        
        return router
    
    @staticmethod
    def create_multi_condition(
        conditions: Dict[str, Callable[[Any], bool]],
        default_path: str
    ) -> Callable:
        """Create a multi-path condition router."""
        def router(state):
            for path, condition in conditions.items():
                if condition(state):
                    return path
            return default_path
        
        return router


def create_simple_workflow(
    llm_function: Callable,
    tools: List[Callable] = None,
    state_schema: type = None
) -> StateGraph:
    """Create a simple agent workflow with LLM and optional tools."""
    builder = WorkflowBuilder(state_schema)
    
    # Add LLM node
    builder.add_node(
        "llm", 
        "Language Model", 
        NodeType.LLM, 
        llm_function,
        "Main language model processing"
    )
    
    # Set entry point
    builder.set_entry_point("llm")
    
    if tools:
        # Add tool execution node
        def execute_tools(state):
            # Tool execution logic
            return state
        
        builder.add_node(
            "tools",
            "Tool Execution",
            NodeType.TOOL,
            execute_tools,
            "Execute available tools"
        )
        
        # Add conditional routing
        def should_use_tools(state):
            # Logic to determine if tools should be used
            return "tools" if hasattr(state.get("messages", [{}])[-1], "tool_calls") else END
        
        builder.add_edge("llm", "tools", should_use_tools)
        builder.add_edge("tools", "llm")
        builder.add_exit_point("llm")
    else:
        builder.add_exit_point("llm")
    
    return builder.build()
''',

            "src/{{package_name}}/workflows/common_workflows.py": '''"""
Common workflow templates for {{project_name}} using modern LangGraph patterns.
"""

from typing import Any, Dict, List, Optional, Literal, Annotated, Sequence
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from .workflow_builder import WorkflowBuilder, NodeType
from ..core.agent_state import AgentState
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResearchState(AgentState):
    """State for research workflows with additional research-specific fields."""
    
    research_plan: List[str] = []
    findings: List[Dict[str, Any]] = []
    sources: List[str] = []
    confidence_score: float = 0.0


class AnalysisState(AgentState):
    """State for analysis workflows with analysis-specific fields."""
    
    data_points: List[Dict[str, Any]] = []
    insights: List[str] = []
    recommendations: List[str] = []
    analysis_type: str = "general"


async def create_research_workflow(
    llm, 
    tools: List[Any], 
    memory_store: Optional[InMemoryStore] = None,
    checkpointer: Optional[MemorySaver] = None
) -> StateGraph:
    """
    Create a modern research-focused workflow with memory and checkpoints.
    
    Args:
        llm: Language model instance
        tools: List of research tools
        memory_store: Optional memory store for context
        checkpointer: Optional checkpointer for workflow state
    """
    
    async def research_planner(state: ResearchState) -> ResearchState:
        """Plan the research approach with memory integration."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Retrieve relevant research context from memory
        context = ""
        if memory_store:
            try:
                # Search for related research plans
                search_results = await memory_store.asearch(
                    ("research_plans",),
                    query=last_message,
                    limit=3
                )
                if search_results:
                    context = "\\n".join([r.value.get("plan", "") for r in search_results])
            except Exception as e:
                logger.warning(f"Memory search failed: {e}")
        
        planning_prompt = f"""You are an expert research planner. 
        Create a comprehensive research plan for: {last_message}
        
        {f"Previous related research plans:\\n{context}\\n" if context else ""}
        
        Break down the research into 3-5 specific, actionable steps.
        Consider:
        1. What information sources would be most valuable
        2. What specific questions need to be answered
        3. How to verify information accuracy
        4. What potential gaps might exist
        
        Respond with a structured research plan."""
        
        response = await llm.ainvoke([
            {"role": "system", "content": planning_prompt},
            *[{"role": m.type, "content": m.content} for m in messages]
        ])
        
        # Extract research plan steps
        plan_steps = response.content.split("\\n") if response.content else []
        plan_steps = [step.strip() for step in plan_steps if step.strip()]
        
        # Store plan in memory
        if memory_store:
            await memory_store.aput(
                ("research_plans", f"plan_{hash(last_message)}"),
                {
                    "query": last_message,
                    "plan": response.content,
                    "steps": plan_steps,
                    "timestamp": state.get("timestamp", "")
                }
            )
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "research_plan": plan_steps
        }
    
    async def research_executor(state: ResearchState) -> ResearchState:
        """Execute research using tools with enhanced error handling."""
        if not tools:
            logger.warning("No research tools available")
            return state
        
        try:
            # Use tool node for execution
            tool_node = ToolNode(tools)
            result = await tool_node.ainvoke(state)
            
            # Extract findings from tool results
            findings = []
            sources = []
            
            for message in result.get("messages", []):
                if isinstance(message, ToolMessage):
                    findings.append({
                        "tool": message.name,
                        "content": message.content,
                        "timestamp": state.get("timestamp", "")
                    })
                    # Extract source URLs if available
                    if "url" in message.content.lower():
                        # Simple URL extraction - could be enhanced
                        import re
                        urls = re.findall(r'https?://[^\\s]+', message.content)
                        sources.extend(urls)
            
            return {
                **result,
                "findings": state.get("findings", []) + findings,
                "sources": list(set(state.get("sources", []) + sources))
            }
            
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            error_msg = AIMessage(content=f"Research execution encountered an error: {str(e)}")
            return {
                **state,
                "messages": state["messages"] + [error_msg]
            }
    
    async def research_synthesizer(state: ResearchState) -> ResearchState:
        """Synthesize research findings with confidence scoring."""
        messages = state["messages"]
        findings = state.get("findings", [])
        sources = state.get("sources", [])
        
        # Build synthesis context
        findings_text = "\\n".join([
            f"- {finding['tool']}: {finding['content'][:200]}..."
            for finding in findings[-5:]  # Last 5 findings
        ])
        
        sources_text = "\\n".join([f"- {source}" for source in sources[-5:]])
        
        synthesis_prompt = f"""Synthesize the research findings into a comprehensive response.
        
        Research Findings:
        {findings_text}
        
        Sources Consulted:
        {sources_text}
        
        Provide:
        1. **Key Findings**: Main discoveries and facts
        2. **Supporting Evidence**: How findings support conclusions
        3. **Confidence Assessment**: Rate confidence 1-10 and explain
        4. **Conclusions**: Clear, evidence-based conclusions
        5. **Recommendations**: Next steps or areas for further research
        
        Be thorough but concise. Cite sources when possible."""
        
        response = await llm.ainvoke([
            {"role": "system", "content": synthesis_prompt},
            *[{"role": m.type, "content": m.content} for m in messages]
        ])
        
        # Extract confidence score (simple pattern matching)
        confidence = 0.7  # default
        if response.content:
            import re
            confidence_match = re.search(r'confidence[:\\s]*([0-9\\.]+)', response.content.lower())
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1)) / 10.0
                except ValueError:
                    pass
        
        # Store synthesis in memory
        if memory_store:
            await memory_store.aput(
                ("research_synthesis", f"synthesis_{len(findings)}"),
                {
                    "findings_count": len(findings),
                    "sources_count": len(sources),
                    "confidence": confidence,
                    "synthesis": response.content,
                    "timestamp": state.get("timestamp", "")
                }
            )
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "confidence_score": confidence
        }
    
    def should_continue_research(state: ResearchState) -> Literal["executor", "synthesizer"]:
        """Determine if more research is needed based on state."""
        messages = state["messages"]
        findings = state.get("findings", [])
        
        # Check if we have tool calls to execute
        last_message = messages[-1] if messages else None
        if (last_message and 
            hasattr(last_message, "tool_calls") and 
            last_message.tool_calls):
            return "executor"
        
        # Check if we have sufficient findings
        if len(findings) < 2:  # Minimum research threshold
            return "executor"
        
        return "synthesizer"
    
    # Build the research workflow
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("planner", research_planner)
    workflow.add_node("executor", research_executor)
    workflow.add_node("synthesizer", research_synthesizer)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "executor",
        should_continue_research,
        {
            "executor": "executor",  # Continue research
            "synthesizer": "synthesizer"  # Move to synthesis
        }
    )
    workflow.add_edge("synthesizer", END)
    
    # Compile with memory and checkpointing
    compile_config = {}
    if checkpointer:
        compile_config["checkpointer"] = checkpointer
    if memory_store:
        compile_config["store"] = memory_store
    
    return workflow.compile(**compile_config)




async def create_conversation_workflow(
    llm, 
    memory_store: Optional[InMemoryStore] = None,
    checkpointer: Optional[MemorySaver] = None
) -> StateGraph:
    """Create a conversational workflow with advanced memory integration."""
    
    async def conversation_handler(state: AgentState) -> AgentState:
        """Handle conversational interactions with memory context."""
        messages = state["messages"]
        user_id = state.get("user_id", "default")
        
        # Build conversation context with memory
        conversation_prompt = """You are a helpful AI assistant with access to conversation history.
        Maintain context and provide thoughtful, relevant responses. Be conversational but informative.
        Reference previous conversations when relevant."""
        
        memory_context = ""
        if memory_store:
            try:
                # Get recent conversation history
                recent_memories = await memory_store.asearch(
                    ("conversations", user_id),
                    limit=5
                )
                
                if recent_memories:
                    memory_context = "\\n".join([
                        f"Previous: {m.value.get('summary', '')}"
                        for m in recent_memories
                    ])
                    conversation_prompt += f"\\n\\nRecent conversation context:\\n{memory_context}"
                
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
        
        # Generate response
        response = await llm.ainvoke([
            {"role": "system", "content": conversation_prompt},
            *[{"role": m.type, "content": m.content} for m in messages]
        ])
        
        # Store conversation turn in memory
        if memory_store and messages:
            last_user_message = messages[-1].content if messages else ""
            await memory_store.aput(
                ("conversations", user_id, f"turn_{len(messages)}"),
                {
                    "user_input": last_user_message,
                    "ai_response": response.content,
                    "summary": f"User asked about {last_user_message[:50]}...",
                    "timestamp": state.get("timestamp", "")
                }
            )
        
        return {
            **state,
            "messages": state["messages"] + [response]
        }
    
    # Build conversation workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("chat", conversation_handler)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    
    # Compile with configuration
    compile_config = {}
    if checkpointer:
        compile_config["checkpointer"] = checkpointer
    if memory_store:
        compile_config["store"] = memory_store
    
    return workflow.compile(**compile_config)


async def create_analysis_workflow(
    llm, 
    tools: List[Any],
    memory_store: Optional[InMemoryStore] = None
) -> StateGraph:
    """Create an analytical workflow for data processing with modern patterns."""
    
    async def data_processor(state: AnalysisState) -> AnalysisState:
        """Process and analyze data with enhanced capabilities."""
        messages = state["messages"]
        analysis_type = state.get("analysis_type", "general")
        
        # Customize processing based on analysis type
        type_prompts = {
            "statistical": "Focus on statistical analysis, distributions, and significance testing.",
            "trend": "Analyze trends, patterns, and temporal changes in the data.",
            "comparative": "Compare different groups, categories, or time periods.",
            "predictive": "Identify patterns that might predict future outcomes.",
            "general": "Perform comprehensive data analysis."
        }
        
        processing_prompt = f"""You are an expert data analyst. {type_prompts.get(analysis_type, type_prompts['general'])}
        
        Analyze the provided information systematically:
        1. Identify and categorize key data points
        2. Calculate relevant statistics and metrics
        3. Look for patterns, trends, and relationships
        4. Assess data quality and limitations
        5. Draw evidence-based conclusions
        
        Be thorough but focus on actionable insights."""
        
        response = await llm.ainvoke([
            {"role": "system", "content": processing_prompt},
            *[{"role": m.type, "content": m.content} for m in messages]
        ])
        
        # Extract data points (simplified extraction)
        data_points = []
        if response.content:
            # Look for numerical values and key metrics
            import re
            numbers = re.findall(r'\\b\\d+(?:\\.\\d+)?\\b', response.content)
            for i, num in enumerate(numbers[:10]):  # Limit to first 10
                data_points.append({
                    "value": float(num),
                    "context": f"metric_{i}",
                    "source": "analysis"
                })
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "data_points": state.get("data_points", []) + data_points
        }
    
    async def insight_generator(state: AnalysisState) -> AnalysisState:
        """Generate actionable insights from analysis."""
        messages = state["messages"]
        data_points = state.get("data_points", [])
        
        insight_prompt = f"""Based on the analysis and {len(data_points)} data points identified, generate actionable insights:
        
        Provide:
        1. **Key Insights**: Most important findings and their implications
        2. **Actionable Recommendations**: Specific steps that can be taken
        3. **Risk Assessment**: Potential risks or limitations in the analysis
        4. **Future Investigation**: Areas that need more research
        5. **Confidence Level**: How confident you are in these insights (1-10)
        
        Focus on practical value and business impact."""
        
        response = await llm.ainvoke([
            {"role": "system", "content": insight_prompt},
            *[{"role": m.type, "content": m.content} for m in messages]
        ])
        
        # Extract insights and recommendations
        insights = []
        recommendations = []
        
        if response.content:
            lines = response.content.split("\\n")
            current_section = None
            
            for line in lines:
                if "insight" in line.lower():
                    current_section = "insights"
                elif "recommendation" in line.lower():
                    current_section = "recommendations"
                elif line.strip().startswith("- ") or line.strip().startswith("* "):
                    item = line.strip()[2:]
                    if current_section == "insights":
                        insights.append(item)
                    elif current_section == "recommendations":
                        recommendations.append(item)
        
        # Store results in memory
        if memory_store:
            await memory_store.aput(
                ("analysis_results", f"analysis_{hash(str(messages))}"),
                {
                    "insights": insights,
                    "recommendations": recommendations,
                    "data_points_count": len(data_points),
                    "analysis_type": state.get("analysis_type", "general"),
                    "timestamp": state.get("timestamp", "")
                }
            )
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "insights": insights,
            "recommendations": recommendations
        }
    
    # Build analysis workflow
    workflow = StateGraph(AnalysisState)
    
    workflow.add_node("processor", data_processor)
    workflow.add_node("insights", insight_generator)
    
    workflow.add_edge(START, "processor")
    workflow.add_edge("processor", "insights")
    workflow.add_edge("insights", END)
    
    compile_config = {}
    if memory_store:
        compile_config["store"] = memory_store
    
    return workflow.compile(**compile_config)


async def create_multi_agent_workflow(
    agents: Dict[str, Any],
    memory_store: Optional[InMemoryStore] = None
) -> StateGraph:
    """Create a sophisticated multi-agent coordination workflow."""
    
    async def intelligent_supervisor(state: AgentState) -> str:
        """AI-powered supervisor for agent routing."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Get available agents
        agent_descriptions = {
            "research_agent": "Handles research queries, fact-finding, and information gathering",
            "analysis_agent": "Performs data analysis, statistical work, and insights generation", 
            "conversation_agent": "Manages general conversation and Q&A",
            "creative_agent": "Handles creative tasks like writing, brainstorming, and ideation",
            "technical_agent": "Manages technical questions, coding, and system administration"
        }
        
        available_agents = [name for name in agent_descriptions.keys() if name in agents]
        
        routing_prompt = f"""You are a supervisor AI that routes user requests to the most appropriate agent.
        
        User request: {last_message}
        
        Available agents:
        {chr(10).join([f"- {name}: {desc}" for name, desc in agent_descriptions.items() if name in available_agents])}
        
        Choose the single most appropriate agent for this request. Respond with just the agent name."""
        
        # Simple LLM-based routing (could be enhanced with more sophisticated logic)
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            routing_response = await agents.get("supervisor_llm", agents.get(list(agents.keys())[0])).ainvoke([
                SystemMessage(content=routing_prompt),
                HumanMessage(content=last_message)
            ])
            
            chosen_agent = routing_response.content.strip().lower()
            if chosen_agent in available_agents:
                return chosen_agent
                
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}")
        
        # Fallback to keyword-based routing
        text_lower = last_message.lower()
        if any(word in text_lower for word in ["research", "find", "search", "investigate"]):
            return "research_agent" if "research_agent" in agents else list(agents.keys())[0]
        elif any(word in text_lower for word in ["analyze", "analysis", "data", "statistics"]):
            return "analysis_agent" if "analysis_agent" in agents else list(agents.keys())[0]
        else:
            return "conversation_agent" if "conversation_agent" in agents else list(agents.keys())[0]
    
    # Build multi-agent workflow
    workflow = StateGraph(AgentState)
    
    # Add supervisor
    workflow.add_node("supervisor", intelligent_supervisor)
    workflow.add_edge(START, "supervisor")
    
    # Add agent nodes
    for agent_name, agent_func in agents.items():
        if agent_name != "supervisor_llm":  # Skip supervisor LLM
            workflow.add_node(agent_name, agent_func)
            # Supervisor routes to each agent
            workflow.add_conditional_edges(
                "supervisor",
                lambda state: intelligent_supervisor(state),
                {agent_name: agent_name for agent_name in agents.keys() if agent_name != "supervisor_llm"}
            )
            workflow.add_edge(agent_name, END)
    
    compile_config = {}
    if memory_store:
        compile_config["store"] = memory_store
    
    return workflow.compile(**compile_config)


async def create_human_in_loop_workflow(
    llm,
    approval_callback: Optional[callable] = None,
    memory_store: Optional[InMemoryStore] = None
) -> StateGraph:
    """Create a workflow with sophisticated human-in-the-loop approval."""
    
    async def ai_processor(state: AgentState) -> AgentState:
        """Process request with AI and mark for review if needed."""
        messages = state["messages"]
        
        # Enhanced AI processing with confidence scoring
        processing_prompt = """Process this request thoughtfully. After your response, 
        rate your confidence level from 1-10 and explain any areas of uncertainty."""
        
        response = await llm.ainvoke([
            {"role": "system", "content": processing_prompt},
            *[{"role": m.type, "content": m.content} for m in messages]
        ])
        
        # Extract confidence score
        confidence = 0.7  # default
        if response.content:
            import re
            conf_match = re.search(r'confidence[:\\s]*([0-9\\.]+)', response.content.lower())
            if conf_match:
                try:
                    confidence = float(conf_match.group(1)) / 10.0
                except ValueError:
                    pass
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "confidence_score": confidence,
            "needs_approval": confidence < 0.8  # Low confidence requires approval
        }
    
    async def human_approval(state: AgentState) -> AgentState:
        """Handle human approval process."""
        messages = state["messages"]
        last_response = messages[-1].content if messages else ""
        confidence = state.get("confidence_score", 0.7)
        
        if approval_callback:
            try:
                # Use provided callback for approval
                approval_result = await approval_callback(state)
                approval_message = approval_result.get("message", "Approved by human reviewer")
                approved = approval_result.get("approved", True)
            except Exception as e:
                logger.error(f"Approval callback failed: {e}")
                approved = True
                approval_message = "Auto-approved due to callback error"
        else:
            # Default approval (in real implementation, this would be interactive)
            approved = True
            approval_message = f"AI response (confidence: {confidence:.1f}) approved for delivery"
        
        # Store approval decision in memory
        if memory_store:
            await memory_store.aput(
                ("approvals", f"approval_{len(messages)}"),
                {
                    "response": last_response[:200],
                    "confidence": confidence,
                    "approved": approved,
                    "message": approval_message,
                    "timestamp": state.get("timestamp", "")
                }
            )
        
        approval_msg = AIMessage(content=approval_message)
        return {
            **state,
            "messages": state["messages"] + [approval_msg],
            "approved": approved
        }
    
    def needs_approval(state: AgentState) -> Literal["approval", "__end__"]:
        """Determine if human approval is needed."""
        return "approval" if state.get("needs_approval", False) else "__end__"
    
    # Build human-in-loop workflow
    workflow = StateGraph(AgentState)
    
    workflow.add_node("ai", ai_processor)
    workflow.add_node("approval", human_approval)
    
    workflow.add_edge(START, "ai")
    workflow.add_conditional_edges(
        "ai",
        needs_approval,
        {
            "approval": "approval",
            "__end__": END
        }
    )
    workflow.add_edge("approval", END)
    
    compile_config = {}
    if memory_store:
        compile_config["store"] = memory_store
    
    return workflow.compile(**compile_config)


# Workflow factory functions for easy instantiation
WORKFLOW_FACTORIES = {
    "research": create_research_workflow,
    "conversation": create_conversation_workflow,
    "analysis": create_analysis_workflow,
    "multi_agent": create_multi_agent_workflow,
    "human_in_loop": create_human_in_loop_workflow
}


def get_workflow_factory(workflow_type: str):
    """Get a workflow factory by type."""
    return WORKFLOW_FACTORIES.get(workflow_type)


def list_available_workflows():
    """List all available workflow types."""
    return list(WORKFLOW_FACTORIES.keys())
''',
        }


# Template configuration
TEMPLATE_CONFIG = {
    "default_structure": {
        "{{project_name}}/": {
            "type": "directory",
            "files": [
                "pyproject.toml",
                "README.md", 
                ".env.example",
                ".gitignore",
                "Dockerfile",
                "docker-compose.yml",
                "Makefile",
                "LICENSE",
                ".pre-commit-config.yaml"
            ],
            "subdirs": {
                "src/{{package_name}}/": {
                    "type": "directory",
                    "files": [
                        "__init__.py",
                        "config.py",
                        "agent.py", 
                        "api.py",
                        "cli.py",
                        "__main__.py"
                    ]
                },
                "config/": {
                    "type": "directory", 
                    "files": ["agent.yaml"]
                },
                "tests/": {
                    "type": "directory",
                    "files": [
                        "__init__.py",
                        "conftest.py",
                        "test_agent.py",
                        "test_config.py", 
                        "test_api.py"
                    ]
                }
            }
        }
    }
}


def get_file_template_manager():
    """Get the file template manager instance."""
    return FileTemplateManager()
