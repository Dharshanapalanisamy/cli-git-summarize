# git-summarize

**AI-powered Git commit message generator using conventional commits**

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/git-summarize/git-summarize)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automatically generate meaningful, conventional commit messages from your staged Git changes using AI. Supports multiple AI providers including Anthropic Claude, OpenAI GPT, Google Gemini, and local Ollama models.

## ✨ Features

- 🤖 **Multi-Provider AI Support** - Choose from Claude, OpenAI, Gemini, or Ollama
- 📝 **Conventional Commits** - Generates standardized commit messages automatically
- 🎯 **Smart Context** - Reads staged diffs, branch info, and recent commit history
- 💡 **Multiple Suggestions** - Get several options and pick the best fit
- 🎨 **Beautiful CLI** - Rich, interactive UI with spinners and formatted output
- ⚡ **Fully Automated** - Stage, commit, and push with a single command
- 🔧 **Configurable** - Environment variables, `.env` files, or CLI flags
- 🏠 **Local Option** - Use Ollama for 100% offline, private AI

## 🚀 Quick Start

### Installation

```bash
pip install git-summarize
```

Or install from source:

```bash
git clone https://github.com/git-summarize/git-summarize.git
cd git-summarize
pip install -e .
```

### Basic Usage

1. **Stage your changes:**
   ```bash
   git add .
   ```

2. **Generate a commit message:**
   ```bash
   gcms generate
   ```

3. **Select or edit** the suggested commit message

4. **Review and confirm** to create the commit

### One-Command Workflow

Stage, commit, and push automatically:

```bash
gcms generate --apply --push
```

## 🤖 AI Providers

### Available Providers

| Provider | Description | Default Model | API Key Required |
|----------|-------------|---------------|------------------|
| **Claude** | Anthropic's Claude AI | `claude-3-sonnet-20240229` | Yes |
| **OpenAI** | OpenAI's GPT models | `gpt-4-turbo-preview` | Yes |
| **Gemini** | Google's Gemini AI (free tier: 60 req/min) | `gemini-1.5-flash` | Yes |
| **Ollama** | Local AI server | `llama2` | No |

### Setup API Keys

**Option 1: Environment Variables**

```bash
# For Claude
export ANTHROPIC_API_KEY="your-key-here"

# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Gemini (get free key: https://aistudio.google.com/app/apikey)
export GEMINI_API_KEY="your-key-here"
```

**Option 2: `.env` File**

Create a `.env` file in your project:

```env
GCM_PROVIDER=gemini
GCM_GEMINI_API_KEY=your-key-here
GCM_NUM_SUGGESTIONS=3
```

### Selecting a Provider

```bash
# Use Claude
gcm generate --provider claude

# Use Gemini (free tier)
gcm generate --provider gemini

# Use local Ollama
gcm generate --provider ollama
```

## 📋 CLI Commands

### `gcm generate`

Generate and apply a commit message from staged changes.

**Options:**

| Flag | Description |
|------|-------------|
| `-p, --provider` | AI provider to use (claude, openai, ollama, gemini) |
| `-m, --model` | Model name (overrides provider default) |
| `-n, --num-suggestions` | Number of suggestions (1-10, default: 3) |
| `-a, --auto` | Auto-select first suggestion without interaction |
| `--preview` | Preview suggestions without committing |
| `--apply` | Apply first suggestion directly |
| `--push` | Push after commit (with branch selection) |
| `--no-add` | Skip `git add .` (use already staged files) |
| `-r, --repo` | Path to Git repository (default: current directory) |

**Examples:**

```bash
# Generate 5 suggestions and interactively select
gcm generate -n 5

# Auto-generate and commit without interaction
gcm generate --auto --apply

# Generate, commit, and push with branch selection
gcm generate --apply --push

# Preview only (no commit)
gcm generate --preview

# Use specific model
gcm generate --provider openai --model gpt-4
```

### `gcm config`

Manage git-summarize configuration.

**Options:**

| Flag | Description |
|------|-------------|
| `--show` | Show current configuration |
| `-p, --provider` | Set default provider |
| `-m, --model` | Set default model |

**Examples:**

```bash
# View current config
gcm config --show

# Set default provider
gcm config --provider gemini
```

### `gcm providers`

List all available AI providersers and their status (API keys configured, default models, etc.).

```bash
gcm providers
```

## 🔄 Workflows

### Interactive Mode (Default)

```bash
gcm generate
```

1. Stages all changes (unless `--no-add`)
2. Reads Git context (diffs, branch, history)
3. Generates suggestions via AI
4. Shows formatted suggestions
5. Lets you **select**, **edit**, **regenerate**, or **quit**
6. Commits with selected message

### Fully Automated

```bash
gcm generate --auto --apply --push
```

Complete workflow: stage → generate → commit → push (with branch selection)

### Preview Only

```bash
gcm generate --preview
```

See what the AI would suggest without making any Git changes.

## ⚙️ Configuration

### Environment Variables

All settings can be configured via environment variables with `GCM_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `GCM_PROVIDER` | Default AI provider | `claude` |
| `GCM_MODEL` | Model name override | - |
| `GCM_NUM_SUGGESTIONS` | Number of suggestions (1-10) | `3` |
| `GCM_ANTHROPIC_API_KEY` | Claude API key | - |
| `GCM_OPENAI_API_KEY` | OpenAI API key | - |
| `GCM_GEMINI_API_KEY` | Gemini API key | - |
| `GCM_OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `GCM_AUTO` | Auto-select first suggestion | `false` |
| `GCM_PREVIEW` | Preview mode | `false` |
| `GCM_APPLY` | Apply first suggestion | `false` |
| `GCM_PUSH` | Push after commit | `false` |

### Config File

Configuration is loaded from `~/.git-summarize/config.toml` if it exists.

### Priority Order

1. CLI flags (highest)
2. Environment variables
3. `.env` file
4. Config file
5. Defaults (lowest)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         CLI Layer                        │
│                    (Typer + main.py)                     │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌────────▼──────────┐
    │   Configuration     │  │        UI         │
    │   (pydantic)        │  │   (Rich console)  │
    └─────────────────────┘  └───────────────────┘
               │
    ┌──────────▼──────────────────────────────────┐
    │            Core Workflow                     │
    │                                              │
    │  ┌─────────┐    ┌──────────┐    ┌────────┐  │
    │  │ Git Ops │───▶│ Git      │───▶│ Prompt │  │
    │  │(stage)  │    │ Reader   │    │Builder │  │
    │  └─────────┘    └──────────┘    └────┬───┘  │
    │                                      │       │
    │  ┌─────────┐    ┌──────────┐    ┌────▼───┐  │
    │  │ Commit  │◀───│ Response │◀───│  AI    │  │
    │  │ & Push  │    │ Parser   │    │Provider│  │
    │  └─────────┘    └──────────┘    └────────┘  │
    └──────────────────────────────────────────────┘
```

## 📦 Project Structure

```
cli-git-summarizer/
├── src/git_summarize/
│   ├── main.py              # Entry point
│   ├── cli.py               # CLI commands (Typer)
│   ├── config.py            # Configuration management
│   ├── git_ops.py           # Git operations (stage, commit, push)
│   ├── git_reader.py        # Git diff/context reading
│   ├── parser.py            # AI response parsing
│   ├── prompt_builder.py    # AI prompt construction
│   ├── ui.py                # Rich UI components
│   └── providers/           # AI provider implementations
│       ├── base.py          # Base provider interface
│       ├── claude.py        # Anthropic Claude
│       ├── openai.py        # OpenAI GPT
│       ├── gemini.py        # Google Gemini
│       └── ollama.py        # Local Ollama
├── tests/                   # Test suite
├── pyproject.toml           # Project metadata
└── README.md                # This file
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/git-summarize/git-summarize.git
cd git-summarize

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/git_summarize --cov-report=term-missing

# Run specific test file
pytest tests/test_cli.py -v
```

### Code Quality

```bash
# Lint with ruff
ruff check src/git_summarize

# Type check with mypy
mypy src/git_summarize
```

### Build Package

```bash
pip install build
python -m build
```

## 🗺️ Roadmap

### v1.2.0 (Planned)
- [ ] Pre-commit hook integration
- [ ] Interactive file staging
- [ ] Recent commits history view

### v1.3.0 (Planned)
- [ ] Custom prompt templates
- [ ] Retry logic for API failures
- [ ] Offline mode with prompt caching

### v2.0.0 (Future)
- [ ] Plugin system for third-party providers
- [ ] GitHub/GitLab PR description generation
- [ ] GPG commit signing support

## 📚 Additional Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Gemini API](https://ai.google.dev/)
- [Ollama Documentation](https://ollama.ai/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/) for CLI framework
- UI powered by [Rich](https://github.com/Textualize/rich)
- Configuration managed by [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- Inspired by the need for better Git commit workflows
