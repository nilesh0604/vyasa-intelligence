# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
