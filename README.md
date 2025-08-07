# 🤖 AgentCLI - AI Agent Project Scaffolding Framework

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NPM](https://img.shields.io/badge/npm-%40stamkavid%2Fagent--cli-red.svg)](https://www.npmjs.com/package/@stamkavid/agent-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)

**Scaffold AI agent projects in minutes, not hours. The most focused and simplified project setup experience for AI agent development.**

*⚡ Now available for both Python and NPM ecosystems - choose your preferred installation method!*

<!-- GIF PLACEHOLDER: Add demo GIF here showing CLI in action -->
![Demo GIF Placeholder](./assets/demo.gif)
<!-- Replace with actual GIF path when ready -->

[Quick Start](#-quick-start) • [Project Structure](#️-generated-project-structure) • [CLI Commands](#-cli-commands) • [Examples](#-quick-examples) • [Development](#-development) • [Contributing](#-contributing)

</div>

---

## Quick Start

### Installation Options

#### Python Ecosystem (Recommended for Python developers)

```bash
# Global installation with pipx (recommended) - works everywhere
pipx install agent-cli

# Or use pip globally
pip install --user agent-cli

# Or in current environment
pip install agent-cli
```

#### NPM Ecosystem (For Node.js developers)

```bash
# Global installation with npm - works everywhere!
npm install -g @stamkavid/agent-cli

# Or use with npx (no installation needed)
npx @stamkavid/agent-cli my-awesome-project
```

### Create Your First AI Agent Project

```bash
# Zero-config setup - just like Claude Code! 🎉
agent-cli my-ai-agent-project

# Or even shorter
agent my-project

# Interactive mode
agent-cli
```

**That's it!** Your AI agent project structure is ready for development. 🎯

## Generated Project Structure

Every scaffolded project follows clean architecture principles with the **exact structure** you specified:

```
agent-project/
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment template
├── .env                           # Local environment
├── README.md                      # Project documentation
├── Makefile                       # Build automation
├── Dockerfile                     # Container setup
├── docker-compose.yml             # Multi-container orchestration
├── pyproject.toml                 # Modern Python packaging
│
├── configs/
│   ├── agent_config.yaml         # Agent configuration
│   ├── llm_config.yaml           # LLM provider settings
│   └── deployment_config.yaml    # Deployment settings
│
├── data/
│   ├── knowledge_base/            # Knowledge storage
│   └── evaluation/                # Evaluation datasets
│
├── k8s/
│   ├── deployment.yaml           # Kubernetes deployment
│   └── service.yaml              # Kubernetes service
│
├── notebooks/
│   ├── 01_prompt_engineering.ipynb     # Prompt development
│   ├── 02_prompt_management.ipynb      # Prompt versioning
│   ├── 03_memory_system.ipynb          # Memory experiments
│   ├── 04_tool_integration.ipynb       # Tool development
│   ├── 05_langgraph_workflows.ipynb    # Workflow design
│   └── 06_evaluation.ipynb             # Performance evaluation
│
├── src/agent_project/
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── main.py                    # Application entry point
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Abstract base agent
│   │   ├── langgraph_agent.py     # LangGraph implementation
│   │   └── state_manager.py       # State persistence
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── langmem_integration.py # LangMem integration
│   │   ├── vector_store.py        # Vector storage
│   │   └── conversation_memory.py # Conversation context
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base_tool.py           # Tool interface
│   │   ├── web_search.py          # Web search capability
│   │   ├── file_operations.py     # File handling
│   │   └── custom_tools.py        # Custom tool templates
│   │
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── workflow_builder.py    # Workflow construction
│   │   └── common_workflows.py    # Pre-built workflows
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── prompt_manager.py      # Prompt lifecycle
│   │   ├── prompt_library.py      # Prompt collection
│   │   ├── templates/
│   │   │   ├── system_prompts.py  # System prompt templates
│   │   │   ├── user_prompts.py    # User prompt templates
│   │   │   ├── tool_prompts.py    # Tool prompt templates
│   │   │   └── workflow_prompts.py # Workflow prompts
│   │   └── versions/
│   │       ├── v1_prompts.py      # Version 1 prompts
│   │       └── v2_prompts.py      # Version 2 prompts
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py              # LLM client abstraction
│   │   └── prompt_executor.py     # Prompt execution
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   ├── routes.py              # API endpoints
│   │   └── schemas.py             # Pydantic schemas
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── opik_integration.py    # Opik monitoring
│   │   └── metrics.py             # Custom metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Logging configuration
│       └── helpers.py             # Utility functions
│
├── tools/
│   ├── run_agent.py               # Agent execution script
│   ├── evaluate_agent.py          # Performance evaluation
│   ├── populate_memory.py         # Memory initialization
│   ├── manage_prompts.py          # Prompt management
│   └── deploy.py                  # Deployment utilities
│
├── tests/
│   ├── test_agent.py              # Agent testing
│   ├── test_memory.py             # Memory testing
│   ├── test_tools.py              # Tool testing
│   └── test_workflows.py          # Workflow testing
│
└── docs/
    ├── getting_started.md         # Quick start guide
    ├── configuration.md           # Configuration guide
    └── deployment.md              # Deployment guide
```

---

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `create` | Create new AI agent project scaffold | `agent-cli create chatbot-project` |
| `list-templates` | Show available project templates | `agent-cli list-templates` |
| `validate-templates` | Validate all project templates | `agent-cli validate-templates` |
| `validate` | Validate a project name | `agent-cli validate my-project` |
| `info` | Show CLI information | `agent-cli info` |

---

## Quick Examples

### Conversational Agent Project

```bash
# Create a customer service chatbot project
agent-cli create customer-service-bot

# Navigate to project
cd customer-service-bot

# Run the included quickstart
cd quickstart && python quickstart.py

# Start development
make dev-setup
```

### Research Assistant Project

```bash
# Create a research agent project
agent-cli create research-assistant

# Navigate to project
cd research-assistant

# Explore the generated structure
ls -la

# Check the notebooks for experimentation
jupyter lab notebooks/
```

### Task Automation Project

```bash
# Create an automation agent project
agent-cli create workflow-automation

# Navigate to project
cd workflow-automation

# Review the generated architecture
tree src/

# Start development
make install
```

---

##  Development Workflow

### 1. Scaffold & Setup
```bash
agent-cli my-agent-project
cd my-agent-project
```

### 2. Explore & Customize
```bash
# Review generated structure
tree src/

# Check configuration
cat configs/agent_config.yaml

# Explore notebooks
jupyter lab notebooks/
```

### 3. Develop & Test
```bash
# Install dependencies
pip install -e .

# Run tests
pytest

# Start development
make dev-setup
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/agent_cli --cov-report=html
```

---

## 🔧 Development

```bash
# Clone and setup
git clone https://github.com/StamKavid/agent-cli.git
cd agent-cli
pip install -e .[dev]

# Run tests
pytest

# Format code
black src/ && isort src/
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and ensure they pass
5. Submit a pull request

---

## Troubleshooting

### Common Issues

**Project name validation fails:**
```bash
# Both formats work - CLI handles conversion automatically
agent-cli create my-agent-project  # ✅ Good (creates my-agent-project/ directory)
agent-cli create my_agent_project  # ✅ Good (creates my_agent_project/ directory)
agent-cli create "my agent!"       # ❌ Bad (invalid characters)
```

**Template validation errors:**
```bash
# Validate all templates
agent-cli validate-templates
```

**Permission errors:**
```bash
# Ensure write permissions
chmod +w /path/to/output/directory
```

### NPM-Specific Issues

**Python not found (NPM users):**
```bash
# Install Python first
# Visit: https://python.org/downloads/

# Then install agent-cli via npm
npm install -g @stamkavid/agent-cli
```

**Manual Python package installation:**
```bash
# If auto-installation fails
pip install agent-cli

# Or with pipx (recommended)
pipx install agent-cli
```

**NPM permission issues:**
```bash
# Use npx instead of global installation
npx @stamkavid/agent-cli my-project

# Or install with --unsafe-perm
npm install -g @stamkavid/agent-cli --unsafe-perm
```

### Debug Mode
```bash
# Enable verbose output
agent-cli create my-agent-project --verbose
agent-cli validate-templates --verbose
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Links

- **🐍 Python Package**: [agent-cli on PyPI](https://pypi.org/project/agent-cli/)
- **📦 NPM Package**: [@stamkavid/agent-cli on npm](https://www.npmjs.com/package/@stamkavid/agent-cli)
- **📚 Documentation**: [GitHub Repository](https://github.com/StamKavid/agent-cli)
- **🐛 Issues**: [GitHub Issues](https://github.com/StamKavid/agent-cli/issues)
- **📋 Contributing**: [Contributing Guide](https://github.com/StamKavid/agent-cli/blob/main/CONTRIBUTING.md)

---

## �🙏 Acknowledgments

### Inspiration
This project was inspired by the amazing work of [Miguel Otero Pedrido (@MichaelisTrofficus)](https://github.com/MichaelisTrofficus).

**Check out his work:**
- 🏢 [The Neural Maze](https://github.com/neural-maze) - Hub for ML projects with step-by-step explanations
- 📺 [YouTube Channel](https://www.youtube.com/@theneuralmaze) - Tutorials and project showcases
- 📧 [Newsletter](https://theneuralmaze.substack.com/) - Latest articles and project updates
- 🔗 [LinkedIn](https://linkedin.com/in/migueloteropedrido) - Professional profile

---

<div align="center">

**Made with ❤️ by [Stamatis Kavidopoulos](https://github.com/StamKavid)**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/StamKavid)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/stamatiskavidopoulos)

**Ready to scaffold amazing AI agent projects? Start creating today! 🚀**

</div>
