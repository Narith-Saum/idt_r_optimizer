# Contributing to IDT-R Optimizer

Thank you for your interest in contributing to the IDT-R Optimizer project!

## How to Contribute

### Reporting Bugs
- Check existing issues first
- Provide a minimal reproducible example
- Include:
  - Python version
  - Package versions
  - Steps to reproduce
  - Actual vs expected behavior

### Suggesting Enhancements
- Use GitHub Discussions for feature ideas
- Clearly describe the enhancement
- Explain why it would be useful

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourname/idt_r_optimizer.git
   cd idt_r_optimizer
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Keep code clean and readable
   - Follow PEP 8 style guidelines
   - Add type hints where possible

5. **Add tests**
   ```bash
   pytest tests/
   ```

6. **Format code**
   ```bash
   black idt_r/
   flake8 idt_r/
   ```

7. **Push and submit PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Generate coverage report
pytest tests/ --cov=idt_r --cov-report=html
```

## Code Style

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Aim for 80-100 character line lengths
- Use meaningful variable names

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", etc.
- Example: "Add support for categorical parameters"

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions
- Include example usage where appropriate

## Testing Requirements

- Aim for >80% code coverage
- Write unit tests for new features
- Test edge cases
- Include integration tests where applicable

## Areas for Contribution

- Algorithm improvements
- Performance optimizations
- Additional examples
- Documentation enhancements
- Type hints
- Test coverage
- Bug fixes

Thank you! 🙏
