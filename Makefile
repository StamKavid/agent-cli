# Agent Project CLI Makefile

.PHONY: help install install-dev test test-create clean example lint validate-templates list-templates

help:
	@echo "Available commands:"
	@echo "  install           - Install the CLI tool (production)"
	@echo "  install-dev       - Install with development dependencies"
	@echo "  test              - Run the test suite"
	@echo "  test-create       - Test project creation"
	@echo "  example           - Create an example project"
	@echo "  clean             - Clean up temporary files"
	@echo "  lint              - Run linting and formatting"
	@echo "  validate-templates - Validate all templates"
	@echo "  list-templates    - List available templates"
	@echo "  info              - Show CLI information"

install:
	uv pip install -e .

install-dev:
	uv pip install -e .[dev]

# Alternative UV commands (faster)
install-uv:
	uv sync --dev

test:
	pytest

lint:
	ruff src/
	black --check src/
	mypy src/

format:
	black src/
	isort src/

validate-templates:
	agent-cli validate-templates

list-templates:
	agent-cli list-templates

info:
	agent-cli info

test-create:
	python -c "from agent_cli import AgentProjectCLI; cli = AgentProjectCLI(); cli.create_project('example-agent', 'test_projects')"

example:
	@echo "Creating example project..."
	@mkdir -p examples
	agent-cli create example-agent-project --output examples
	@echo "Example created in examples/example-agent-project/"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf test_projects/
	rm -rf examples/

# Quick commands for different project types
create-chatbot:
	agent-cli create chatbot-agent --output .

create-research:
	agent-cli create research-agent --output .

create-automation:
	agent-cli create automation-agent --output .

# Development workflow
dev-setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Development environment ready!"

check-all: lint test validate-templates
	@echo "All checks passed! ✨"

# CI/CD helpers
ci-test:
	pytest --cov=src/agent_cli --cov-report=xml --cov-report=term-missing

ci-lint:
	ruff check src/
	black --check src/
	mypy src/

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here

# Release helpers
bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major

build:
	uv build

publish:
	uv publish
