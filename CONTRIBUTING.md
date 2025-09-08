# Contributing to Bedrock Prompt Optimizer

We welcome contributions to the Bedrock Prompt Optimizer! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)
7. [Development Standards](#development-standards)
8. [Testing Guidelines](#testing-guidelines)
9. [Documentation](#documentation)
10. [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to help us maintain a welcoming and inclusive community.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- AWS account with Bedrock access
- Basic understanding of prompt engineering and LLMs

### First-time Contributors

If you're new to contributing to open source projects:

1. Start by reading this contributing guide
2. Look for issues labeled `good first issue` or `help wanted`
3. Join our community discussions
4. Ask questions if you need help

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/bedrock-prompt-optimizer.git
cd bedrock-prompt-optimizer

# Add the original repository as upstream
git remote add upstream https://github.com/example/bedrock-prompt-optimizer.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest

# Run linting
black --check .
flake8 .
mypy .

# Test CLI functionality
bedrock-optimizer --help
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in existing code
- **Feature additions**: Add new functionality
- **Documentation improvements**: Enhance docs, examples, or comments
- **Performance optimizations**: Improve speed or resource usage
- **Test improvements**: Add or improve test coverage
- **Best practices**: Add new prompt engineering techniques

### Before You Start

1. **Check existing issues**: Look for existing issues or discussions about your idea
2. **Create an issue**: For significant changes, create an issue to discuss the approach
3. **Get feedback**: Engage with maintainers and community members
4. **Start small**: Begin with smaller contributions to get familiar with the codebase

### Branch Naming

Use descriptive branch names:

- `feature/add-custom-evaluators`
- `bugfix/fix-session-persistence`
- `docs/update-cli-guide`
- `refactor/improve-agent-coordination`

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Add and commit changes
git add .
git commit -m "Add feature: your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### 2. Create Pull Request

1. Go to GitHub and create a pull request from your fork
2. Use a clear, descriptive title
3. Fill out the pull request template
4. Link any related issues
5. Add appropriate labels

### 3. Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues
Fixes #(issue number)

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or breaking changes documented)
```

### 4. Review Process

1. **Automated checks**: Ensure all CI checks pass
2. **Code review**: Address feedback from maintainers
3. **Testing**: Verify functionality works as expected
4. **Documentation**: Update docs if needed
5. **Approval**: Get approval from maintainers

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 1.0.0]
- AWS region: [e.g., us-east-1]

**Additional Context**
Any other relevant information.

**Logs**
```
Relevant log output
```
```

### Feature Requests

For feature requests, please include:

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How do you envision this feature working?

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other relevant information.
```

## Development Standards

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning

```bash
# Format code
black .
isort .

# Check linting
flake8 .

# Type checking
mypy .

# Security scan
bandit -r .
```

### Code Quality Guidelines

1. **Follow PEP 8**: Use Python style guidelines
2. **Type hints**: Add type hints to all functions
3. **Docstrings**: Document all public functions and classes
4. **Error handling**: Handle errors gracefully
5. **Logging**: Use structured logging
6. **Comments**: Add comments for complex logic

### Example Code Style

```python
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ExampleClass:
    """Example class demonstrating code style.
    
    This class shows the expected code style including type hints,
    docstrings, and error handling.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the example class.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("Configuration cannot be empty")
        
        self.config = config
        logger.info("ExampleClass initialized")
    
    def process_data(self, 
                    data: List[str], 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data with optional configuration.
        
        Args:
            data: List of strings to process
            options: Optional processing options
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        options = options or {}
        
        try:
            # Process data here
            result = {"processed": len(data), "options": options}
            logger.debug(f"Processed {len(data)} items")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process data: {e}")
            raise
```

## Testing Guidelines

### Test Structure

We use pytest for testing with the following structure:

```
tests/
├── unit/                   # Unit tests
│   ├── test_agents.py
│   ├── test_bedrock.py
│   └── test_evaluation.py
├── integration/            # Integration tests
│   ├── test_end_to_end.py
│   └── test_workflows.py
├── performance/            # Performance tests
│   └── test_performance.py
└── fixtures/              # Test fixtures
    ├── conftest.py
    └── sample_data.py
```

### Writing Tests

1. **Test naming**: Use descriptive test names
2. **Test isolation**: Each test should be independent
3. **Fixtures**: Use fixtures for common test data
4. **Mocking**: Mock external dependencies
5. **Coverage**: Aim for high test coverage

### Example Test

```python
import pytest
from unittest.mock import Mock, patch
from your_module import YourClass

class TestYourClass:
    """Test suite for YourClass."""
    
    @pytest.fixture
    def sample_config(self):
        """Provide sample configuration for tests."""
        return {
            "setting1": "value1",
            "setting2": 42
        }
    
    @pytest.fixture
    def your_class(self, sample_config):
        """Create YourClass instance for testing."""
        return YourClass(sample_config)
    
    def test_initialization_success(self, sample_config):
        """Test successful initialization."""
        instance = YourClass(sample_config)
        assert instance.config == sample_config
    
    def test_initialization_failure(self):
        """Test initialization with invalid config."""
        with pytest.raises(ValueError, match="Configuration cannot be empty"):
            YourClass({})
    
    @patch('your_module.external_service')
    def test_process_with_mock(self, mock_service, your_class):
        """Test processing with mocked external service."""
        mock_service.return_value = {"status": "success"}
        
        result = your_class.process_data(["test"])
        
        assert result["processed"] == 1
        mock_service.assert_called_once()
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_agents.py

# Run tests matching pattern
pytest -k "test_agent"

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Documentation

### Documentation Types

1. **Code documentation**: Docstrings and comments
2. **User guides**: How-to guides and tutorials
3. **API reference**: Detailed API documentation
4. **Architecture docs**: System design and architecture

### Writing Documentation

1. **Clear and concise**: Use simple, clear language
2. **Examples**: Include practical examples
3. **Up-to-date**: Keep documentation current with code changes
4. **Accessible**: Consider different skill levels

### Documentation Structure

```
docs/
├── CLI_USAGE.md           # CLI usage guide
├── BEST_PRACTICES_GUIDE.md # Best practices documentation
├── DEPLOYMENT_GUIDE.md    # Deployment instructions
├── tutorials/             # Step-by-step tutorials
│   ├── GETTING_STARTED.md
│   └── ADVANCED_EXAMPLES.md
└── api/                   # API reference
    ├── agents.md
    └── orchestration.md
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Community Forum**: [https://community.example.com/bedrock-optimizer](https://community.example.com/bedrock-optimizer)
- **Email**: support@example.com

### Getting Help

1. **Check documentation**: Look at existing docs first
2. **Search issues**: Check if your question has been asked
3. **Ask questions**: Don't hesitate to ask for help
4. **Be specific**: Provide context and details

### Helping Others

- Answer questions in discussions
- Review pull requests
- Improve documentation
- Share your experiences

## Recognition

We appreciate all contributions and recognize contributors in several ways:

- **Contributors list**: Listed in README and releases
- **Special mentions**: Highlighted in release notes
- **Community recognition**: Featured in community updates

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Questions?

If you have questions about contributing, please:

1. Check this guide and existing documentation
2. Search existing issues and discussions
3. Create a new discussion or issue
4. Contact maintainers directly if needed

Thank you for contributing to Bedrock Prompt Optimizer! 🚀