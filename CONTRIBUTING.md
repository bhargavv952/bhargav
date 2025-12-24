# Contributing to Bhargav

Thank you for your interest in contributing to the Bhargav project! We welcome contributions from the community and appreciate your efforts to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions with other community members.

## Getting Started

1. **Fork the repository** - Click the "Fork" button at the top right of the repository page
2. **Clone your fork** - `git clone https://github.com/YOUR_USERNAME/bhargav.git`
3. **Add upstream remote** - `git remote add upstream https://github.com/bhargavv952/bhargav.git`
4. **Create a new branch** - `git checkout -b feature/your-feature-name`

## How to Contribute

You can contribute in several ways:

- **Report bugs** - Open an issue describing the problem
- **Suggest features** - Share your ideas for improvements
- **Improve documentation** - Help us improve existing docs
- **Submit code changes** - Fix bugs or implement new features
- **Review pull requests** - Help review others' contributions

## Development Setup

1. Ensure you have the necessary dependencies installed
2. Follow the project's setup instructions in the README.md
3. Create a virtual environment if applicable
4. Install development dependencies
5. Run tests to ensure everything works

## Submitting Changes

### Before You Start

- Check if an issue already exists for your proposed change
- Create an issue first to discuss significant changes
- Keep changes focused and manageable in scope

### Creating a Branch

```bash
git checkout -b feature/descriptive-branch-name
```

Branch naming conventions:
- `feature/description` - for new features
- `bugfix/description` - for bug fixes
- `docs/description` - for documentation updates
- `refactor/description` - for code refactoring

## Coding Standards

- Follow the existing code style and conventions
- Write clean, readable code with meaningful variable names
- Include comments for complex logic
- Keep functions small and focused on a single responsibility
- Write tests for new functionality

## Commit Messages

Write clear and descriptive commit messages:

```
[Type] Brief description of changes

More detailed explanation of what changed and why,
if necessary. Keep lines under 72 characters.

Fixes #123 (if applicable)
```

Types:
- `feat:` - A new feature
- `fix:` - A bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a pull request** on GitHub with:
   - A descriptive title
   - A clear description of changes
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots or examples if applicable

4. **Address feedback** - Be responsive to code review comments

5. **Ensure tests pass** - All CI/CD checks must pass

6. **Keep commits clean** - Squash commits if requested by reviewers

## Reporting Issues

When reporting bugs, please include:

- A clear, descriptive title
- A detailed description of the problem
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Screenshots or error logs if applicable
- Your environment details (OS, version, etc.)

## Community

- Be respectful and constructive in all discussions
- Help others who are new to the project
- Share knowledge and best practices
- Celebrate contributions from all community members

## Questions?

If you have questions or need clarification, feel free to:
- Open a GitHub issue
- Reach out to the maintainers
- Check existing documentation

---

Thank you for contributing to Bhargav! We look forward to working with you. 🎉
