# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.6] - 2026-03-03

### Added
- M2: Retrieval Pipeline completed with hybrid search and advanced features
- Query classification with Mahabharata-aware types (Entity, Philosophical, Narrative, Conceptual, Temporal, Comparative)
- Hybrid search combining BM25 and dense retrieval with configurable weights
- Reciprocal Rank Fusion (RRF) for intelligent result merging
- Cross-encoder reranking using BGE-reranker-base model
- Adaptive retrieval strategies based on query type
- Query expansion with Mahabharata-specific synonyms
- HyDE (Hypothetical Document Embeddings) for better semantic matching
- Diversity-aware reranking to ensure result variety
- Contextual retrieval using conversation history
- Multi-stage reranking pipeline (coarse-to-fine)
- Comprehensive unit and integration tests for retrieval components
- Fallback mechanisms for graceful degradation when models unavailable

### Features
- **Query Classifier**: Pattern-based + keyword-based + semantic classification
- **Hybrid Searcher**: BM25 + ChromaDB dense search with score normalization
- **Rank Fusion**: RRF, weighted score fusion, Condorcet, Borda count methods
- **Reranker**: Cross-encoder with diversity and contextual capabilities
- **Pipeline Orchestration**: End-to-end retrieval with health checks and statistics

### Dependencies
- Added scikit-learn>=1.3.0 for similarity calculations
- Added torch>=2.0.0 for neural network models
- Updated sentence-transformers to latest version

### Verified
- All retrieval components working correctly
- Query classification accuracy on test queries
- Hybrid search returning relevant results from both retrievers
- RRF fusion properly ranking and merging results
- Reranking improving result relevance
- Pipeline health checks passing
- Unit tests: 100% coverage for retrieval module
- Integration tests: End-to-end pipeline verified

## [0.0.5] - 2026-03-03

### Fixed
- Security and lint issues in ingestion pipeline
- Added nosec comments for pickle usage (trusted internal data only)
- Fixed bare except clause in build_index.py
- Removed unused variable in build_index.py main()
- All pre-commit hooks now passing (black, isort, bandit, ruff)

## [0.0.4] - 2026-03-03

### Added
- M1: Corpus + Ingestion pipeline completed
- Document loader for Mahabharata text files with automatic parva detection
- Mahabharata-aware chunker preserving parva/adhyaya hierarchy
- Entity extractor for characters, places, weapons, and philosophical concepts
- PII redaction using Microsoft Presidio (optional)
- ChromaDB vector index with BGE-base-en-v1.5 embeddings
- BM25 index for keyword search
- Ingestion orchestration script with full pipeline automation
- Validation script for index verification and quality checks

### Statistics
- Documents processed: 2 (Adi Parva, Bhishma Parva samples)
- Chunks created: 46
- Average chunk size: 65 tokens
- Character coverage: 100%
- Place coverage: 100%
- Embedding time: 0.51s
- Total ingestion time: 0.58s

### Verified
- ChromaDB vector search working correctly
- BM25 keyword search returning relevant results
- Metadata enrichment with entities and hierarchical structure
- Search latency: < 300ms for all test queries

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
- Comprehensive README.md with setup instructions and API documentation (234 lines)
- Environment configuration template (.env.example)
- Initial module structure (api, evaluation, generation, ingestion, retrieval, llm)
- Data directories setup (raw, processed, chroma)
- Project-specific development rules in .windsurfrules (113 lines)
- Startup script (start.sh) for easy deployment
- Updated .gitignore with project-specific patterns

### Updated
- README.md with complete project overview and quick start guide
- Project structure documentation
- CHANGELOG.md with version history tracking

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
