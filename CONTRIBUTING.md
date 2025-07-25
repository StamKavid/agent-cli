# Contributing to Agent CLI

Thank you for your interest in contributing to Agent CLI! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/agent-cli.git`
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Install** dependencies: `make install-dev` or `uv sync --dev`
5. **Make** your changes
6. **Test** your changes: `make test`
7. **Commit** your changes: `git commit -m 'Add amazing feature'`
8. **Push** to your branch: `git push origin feature/amazing-feature`
9. **Open** a Pull Request

## 🛠️ Development Setup

### Prerequisites
- Python 3.10+
- UV (recommended) or pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/StamKavid/agent-cli.git
cd agent-cli

# Install with UV (recommended)
uv sync --dev

# Or install with pip
make install-dev
```

### Development Commands

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Validate templates
make validate-templates

# Run all checks
make check-all
```

## 📝 Code Standards

### Python Style Guide
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [mypy](https://mypy.readthedocs.io/) for type checking

### Code Quality
- Write comprehensive docstrings for all public functions and classes
- Use type hints throughout the codebase
- Write unit tests for new functionality
- Maintain test coverage above 80%
- Follow SOLID principles and clean code practices

### Commit Messages
Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Branch Naming
- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation updates
- `refactor/description`: Code refactoring

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/agent_cli --cov-report=html

# Run specific test file
pytest tests/unit/test_cli.py

# Run with verbose output
pytest -v
```

### Writing Tests
- Write tests for all new functionality
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Test both success and error cases

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings for all public functions and classes
- Include type hints in docstrings
- Document exceptions that may be raised
- Provide usage examples for complex functions

### README Updates
- Update README.md for new features
- Add usage examples
- Update installation instructions if needed
- Keep the quick start section current

## 🔧 Template Development

### Adding New Templates
1. Create template content in `src/agent_cli/templates/`
2. Add template to the appropriate category
3. Update template validation
4. Add tests for the new template
5. Update documentation

### Template Guidelines
- Use Jinja2 syntax for template variables
- Provide clear variable names
- Include helpful comments in templates
- Test template rendering with various inputs

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages and stack traces

## 💡 Feature Requests

When requesting features, please include:
- Description of the feature
- Use case and motivation
- Proposed implementation approach (if any)
- Examples of similar features in other projects

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow conventional format

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Other (please describe)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Additional Notes
Any additional information or context
```

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow the project's code of conduct

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Join our discussions for general questions
- Check existing issues and pull requests

Thank you for contributing to Agent CLI! 🚀 