# 🤖 AgentCLI - AI Agent Project Scaffolding Framework

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)

**Scaffold AI agent projects in minutes, not hours. The most seamless project setup experience for AI agent development.**

[Quick Start](#-quick-start) • [Features](#-features) • [Templates](#-templates) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🚀 Quick Start

### Installation

#### Using UV (Recommended - Faster)
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install agent-cli
uv pip install agent-cli

# Or install with all optional dependencies
uv pip install agent-cli[full]

# For development (if you want to contribute)
git clone https://github.com/StamKavid/agent-cli.git
cd agent-cli
uv sync --dev
```

#### Using pip
```bash
# Install from PyPI (recommended)
pip install agent-cli

# Or install with all optional dependencies
pip install agent-cli[full]

# For development (if you want to contribute)
git clone https://github.com/StamKavid/agent-cli.git
cd agent-cli
pip install -e .[dev]
```

### Create Your First AI Agent Project

```bash
# Interactive setup - the fun way! 🎉
agent-cli create my-ai-agent-project

# Or specify everything upfront
agent-cli create chatbot-project --output ~/projects
```

**That's it!** Your AI agent project structure is ready for development. 🎯

---

## ✨ Features

### 🎨 **Beautiful Interactive CLI**
- **Rich terminal interface** with progress bars, spinners, and emojis
- **Guided setup wizard** - no configuration headaches
- **Real-time feedback** with live progress tracking
- **Smart defaults** that work out of the box

### 🏗️ **Project Template Categories**
- **💬 Conversational Agent Projects** - Chatbots, customer service applications
- **🔍 Research Assistant Projects** - Information gathering, analysis tools
- **⚙️ Task Automation Projects** - Workflow orchestration systems
- **📊 Data Analysis Agent Projects** - Reporting, insights applications
- **🎯 Custom Agent Projects** - Build from scratch with clean architecture

### 🛠️ **Production-Ready Project Structure**
- **Clean Architecture** with proper separation of concerns
- **Multiple LLM Provider Support** (OpenAI, Anthropic, Google, Azure, Local)
- **Memory System Templates** - Short-term and long-term memory patterns
- **Tool Integration Scaffolding** - Web search, file operations, custom tools
- **Monitoring & Observability Setup** - Built-in logging and metrics templates
- **REST API Interface Templates** - Deploy as a service structure

### 🔧 **Developer Experience**
- **Hot Reload Setup** - Development environment configuration
- **Interactive Testing Tools** - Test your agent projects in real-time
- **Debug Mode Configuration** - Comprehensive logging and debugging setup
- **Deployment Ready** - Docker, CI/CD, cloud deployment templates

---

## 🎯 Seamless Project Scaffolding Experience

### Interactive Project Creation

```bash
agent-cli create my-agent-project
```

**What happens:**
1. **🎨 Beautiful welcome screen** with project setup wizard
2. **🏗️ Template selection** - Choose your project type
3. **⚙️ Configuration options** - LLM provider, memory, tools, monitoring
4. **📁 Project structure generation** - Clean, modular architecture
5. **📦 Dependencies setup** - All required packages configured
6. **🚀 Ready for development** - Your project scaffold is ready in seconds!

### Real-Time Progress

```
🤖 Welcome to Agent CLI - AI Agent Development Framework
Let's create your amazing AI agent project!

Selected: 💬 Conversational Agent (Chatbot, Customer Service)
LLM Provider [anthropic/openai/google/local] (anthropic): openai
Enable memory system? [y/n] (y): y
Include common tools (web search, file operations)? [y/n] (y): y
Enable monitoring and observability? [y/n] (y): y
Include REST API interface? [y/n] (y): y

⠋ Creating project structure... ━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  20%
```

### Instant Project Testing

```bash
# Navigate to your project
cd my-agent-project

# Run the included quickstart
cd quickstart
python quickstart.py

# Start development
make dev-setup
```

---

## 🏗️ Generated Project Structure

Every scaffolded project follows clean architecture principles:

```
my-ai-agent-project/
├── 🚀 Quick Start
│   ├── quickstart.py              # Complete working example
│   ├── requirements-quickstart.txt # Minimal dependencies
│   └── README.md                  # Getting started guide
├── 📓 Jupyter Notebooks
│   ├── 01_prompt_engineering_playground.ipynb
│   ├── 02_short_term_memory.ipynb
│   ├── 03_long_term_memory.ipynb
│   └── 04_tool_calling_playground.ipynb
├── 💻 Source Code
│   └── my_ai_agent_project/
│       ├── config.py              # Pydantic configuration
│       ├── main.py                # Application entry point
│       ├── core/                  # Agent logic templates
│       ├── infrastructure/        # External services templates
│       │   ├── llm_clients/       # LLM provider implementations
│       │   ├── vector_database/   # Vector storage templates
│       │   ├── monitoring/        # Logging & metrics setup
│       │   └── api/               # REST API layer templates
│       └── application/           # Business logic templates
├── 🧪 Tests
│   ├── test_agent.py
│   └── test_memory.py
├── 🛠️ Tools
│   ├── run_agent.py
│   ├── populate_memory.py
│   └── evaluate_agent.py
└── 📦 Configuration
    ├── pyproject.toml             # Modern Python config
    ├── Dockerfile                 # Container ready
    ├── Makefile                   # Build automation
    └── .env                       # Environment variables
```

---

## 🎨 CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `create` | Create new AI agent project scaffold | `agent-cli create chatbot-project` |
| `list-templates` | Show available project templates | `agent-cli list-templates` |
| `validate-templates` | Validate all project templates | `agent-cli validate-templates` |
| `validate` | Validate a project name | `agent-cli validate my-project` |
| `info` | Show CLI information | `agent-cli info` |

---

## 🚀 Quick Examples

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

## 🎯 Project Template Categories

### 💬 Conversational Agent Projects
Perfect for chatbots, customer service, and interactive applications.

**Generated Structure:**
- Multi-turn conversation templates
- Context awareness patterns
- Personality customization setup
- Response generation scaffolding
- Intent recognition templates

### 🔍 Research Assistant Projects
Ideal for information gathering, analysis, and research tools.

**Generated Structure:**
- Web search integration templates
- Document analysis patterns
- Data synthesis scaffolding
- Citation tracking setup
- Report generation templates

### ⚙️ Task Automation Projects
Built for workflow orchestration and process automation.

**Generated Structure:**
- Workflow management templates
- Task scheduling patterns
- Error handling scaffolding
- Progress tracking setup
- Integration capabilities templates

### 📊 Data Analysis Agent Projects
Specialized for data processing, reporting, and insights.

**Generated Structure:**
- Data processing pipeline templates
- Statistical analysis patterns
- Visualization generation setup
- Report automation scaffolding
- Trend detection templates

---

## 🔧 Development Workflow

### 1. Scaffold & Setup
```bash
agent-cli create my-agent-project
cd my-agent-project
```

### 2. Explore & Customize
```bash
# Review generated structure
tree src/

# Check configuration
cat config.py

# Explore notebooks
jupyter lab notebooks/
```

### 3. Develop & Test
```bash
# Install dependencies
make install

# Run tests
make test

# Start development
make dev-setup
```

---

## 🛠️ Advanced Project Features

### LLM Provider Templates
- **OpenAI** - GPT-4, GPT-3.5-turbo integration templates
- **Anthropic** - Claude-3, Claude-2 integration templates
- **Google** - Gemini Pro, PaLM integration templates
- **Azure** - Azure OpenAI integration templates
- **Local** - Ollama, LM Studio integration templates

### Memory System Templates
- **Short-term Memory** - Conversation context patterns
- **Long-term Memory** - Persistent knowledge templates
- **Vector Storage** - Semantic search setup
- **Memory Management** - Automatic cleanup patterns

### Tool Integration Templates
- **Web Search** - Real-time information integration
- **File Operations** - Read/write file patterns
- **API Calls** - External service integration templates
- **Custom Tools** - Extend with your own patterns

### Monitoring & Observability Templates
- **Structured Logging** - Comprehensive logging setup
- **Metrics Collection** - Performance tracking templates
- **Error Tracking** - Automatic error reporting setup
- **Health Checks** - System monitoring patterns

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git
- Your preferred LLM API key

### Installation
```bash
# Clone the repository
git clone https://github.com/StamKavid/agent-cli.git
cd agent-cli

# Install in development mode
pip install -e .[dev]

# Verify installation
agent-cli info
```

### Your First AI Agent Project
```bash
# Create a conversational agent project
agent-cli create my-first-agent-project

# Navigate to your project
cd my-first-agent-project

# Run the quickstart
cd quickstart
python quickstart.py
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=src/agent_cli --cov-report=html

# Run specific tests
pytest tests/unit/test_cli.py -v
```

---

## 🔧 Development

### Setup Development Environment
```bash
# Install dependencies with UV (recommended)
uv sync --dev

# Or install with pip
make install-dev

# Setup development environment
make dev-setup

# Run all checks
make check-all
```

### Code Quality
```bash
# Lint code
make lint

# Format code
make format

# Validate templates
make validate-templates
```

---

## 🤝 Contributing

We love contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Standards
- Follow PEP 8 style guidelines
- Use type hints throughout
- Write comprehensive docstrings
- Maintain high test coverage
- Use pre-commit hooks

---

## 🐛 Troubleshooting

### Common Issues

**Project name validation fails:**
```bash
# Use valid Python package names
agent-cli create my_agent_project  # ✅ Good
agent-cli create my-agent-project  # ❌ Bad
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

### Debug Mode
```bash
# Enable verbose output
agent-cli create my-agent-project --verbose
agent-cli validate-templates --verbose
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

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
