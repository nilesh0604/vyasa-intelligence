# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2026-03-03

### Added
- Python 3.11.14 installed via Homebrew (resolving dependency conflicts)
- New virtual environment venv-python311 with Python 3.11
- All 250+ project dependencies successfully installed
- Pre-commit hooks installed and configured (black, isort, bandit, ruff)
- Ollama integration test script (test_ollama.py)
- Startup script (start.sh) for easy application launch
- FastAPI application verified and working correctly

### Fixed
- Python version compatibility issue (3.11 required, had 3.13)
- Bandit security warning for binding to all interfaces

### Verified
- All dependencies imported successfully
- Ollama LLM integration working (7.49s generation time)
- Pre-commit hooks passing on all code
- FastAPI routes: /health, /query, /docs, /openapi.json

## [0.0.2] - 2026-03-03

### Added
- FastAPI application with health check and query endpoints
- LLM factory abstraction for Ollama and Groq providers
- Comprehensive README.md with setup instructions and API documentation
- Environment configuration template (.env.example)
- Initial module structure (api, evaluation, generation, ingestion, retrieval, llm)
- Data directories setup (raw, processed, chroma)

### Updated
- README.md with complete project overview and quick start guide
- Project structure documentation

## [0.0.1] - 2026-03-03

### Added
- M0: Development environment setup
- Python 3.11.14 with uv package manager
- All required dependencies installed (250+ packages)
- Pre-commit hooks configured (black, isort, bandit, ruff)
- Project structure created (src/, tests/, data/, k8s/)
- Ollama with llama3.2 model installed and verified
- Rancher Desktop Kubernetes enabled and verified
- Local LLM module for LangChain integration
- Environment variables template (.env)
- LangSmith API key configured for tracing
- M0 validation test script

### Verified
- Python version >= 3.11 ✓
- All critical package imports ✓
- Environment variables configuration ✓
- Ollama serving llama3.2 model ✓
- LangChain-Ollama integration ✓
- Kubernetes cluster running ✓
