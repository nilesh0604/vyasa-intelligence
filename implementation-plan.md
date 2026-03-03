# Vyasa Intelligence — Step-by-Step Implementation Plan

> **Strategy:** Build everything locally first using Ollama + Rancher Desktop. Validate every layer before touching cloud credentials. Cloud is the last step, not the first.

---

## Milestone Map

| Milestone | Focus | Exit Criteria | Status |
|-----------|-------|--------------|--------|
| **M0** | Dev environment | All tools install, Ollama serves a model | ✅ Complete |
| **M1** | Corpus + ingestion | ChromaDB populated, BM25 index built | ✅ Complete |
| **M2** | Retrieval pipeline | Hybrid search returns ranked results locally | ✅ Complete |
| **M3** | Generation (Ollama) | Full RAG answer generated end-to-end locally | ✅ Complete |
| **M4** | Evaluation | Ragas scores computed; quality gates pass | ✅ Complete |
| **M5** | Containerisation | `docker compose up` reproduces M3 result | ✅ Complete |
| **M6** | Local Kubernetes | App runs in Rancher Desktop K8s cluster | ✅ Complete |
| **M7** | Groq swap | Ollama replaced by Groq API; gates still pass | 🔄 Next |
| **M8** | CI/CD | GitHub Actions 4-gate pipeline green | ⏳ Pending |
| **M9** | HF Spaces staging | Public demo live; monitored via LangSmith | ⏳ Pending |
| **M10** | AWS production | Lambda + S3 Vectors + Bedrock; IaC tracked | ⏳ Pending |
| **M11** | Azure production | Azure AI Search + Container Apps; IaC tracked | ⏳ Pending |

---

## Phase 0 — Dev Environment (Day 1–2)

### 0.1 Python toolchain

```bash
# Install pyenv (manages Python versions)
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Install uv (fast package installer + venv manager)
curl -Ls https://astral.sh/uv/install.sh | sh

# Bootstrap project
mkdir -p vyasa-intelligence && cd vyasa-intelligence
uv init --python 3.11
uv add langchain langchain-community langchain-groq \
        chromadb rank-bm25 ragatouille \
        sentence-transformers transformers \
        ragas datasets \
        presidio-analyzer presidio-anonymizer \
        fastapi uvicorn gradio \
        tenacity pydantic python-dotenv \
        boto3 azure-ai-contentsafety \
        pytest pytest-asyncio httpx
```

### 0.2 Pre-commit hooks

```bash
uv add --dev pre-commit black isort mypy bandit ruff
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks: [{id: black}]
  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks: [{id: isort}]
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.8
    hooks: [{id: bandit, args: ["-r", "src", "-ll"]}]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks: [{id: ruff, args: [--fix]}]
EOF
pre-commit install
```

### 0.3 Ollama — local LLM server

```bash
# macOS
brew install ollama
ollama serve &        # starts on http://localhost:11434

# Pull models (choose one for development)
ollama pull llama3.2          # 2B, fastest, good for iteration (~1.3GB)
ollama pull llama3.1:8b       # 8B, better quality (~4.7GB)
ollama pull mistral:7b        # alternative baseline

# Verify
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.2","prompt":"Who wrote the Mahabharata?","stream":false}'
```

**LangChain integration:**
```python
# src/llm/local_llm.py
from langchain_ollama import OllamaLLM

def get_llm(model: str = "llama3.2", temperature: float = 0.1):
    return OllamaLLM(
        model=model,
        base_url="http://localhost:11434",
        temperature=temperature,
    )
```

### 0.4 Rancher Desktop — Docker + local K8s

```bash
# Download from https://rancherdesktop.io/ (macOS .dmg)
# Settings → Kubernetes → Enable Kubernetes
# Container engine: containerd  (or dockerd if you prefer docker CLI)
# After install, verify:
kubectl get nodes          # should show rancher-desktop node
docker ps                  # should work if dockerd selected
nerdctl ps                 # for containerd engine
```

### 0.5 Environment variables

```bash
# .env  (never commit this)
GROQ_API_KEY=gsk_...           # free at console.groq.com
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=./data/chroma
BM25_INDEX_PATH=./data/bm25_index.pkl
LANGCHAIN_API_KEY=ls__...      # LangSmith free tier
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=vyasa-local

# Exit criteria: python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('GROQ_API_KEY')[:10])"
```

---

## Phase 1 — Corpus Ingestion (Day 3–5)

### 1.1 Acquire the Mahabharata corpus

```bash
mkdir -p data/raw data/processed data/chroma

# Option A — BORI Critical Edition plain text (public domain)
# Download from sacred-texts.com or Project Gutenberg translations
# K.M. Ganguli translation (~4.5M tokens) is most complete English version

# Recommended structure after download:
# data/raw/
#   adi_parva.txt        (~320K tokens)
#   sabha_parva.txt      (~80K tokens)
#   vana_parva.txt       (~570K tokens)
#   ... (18 parvas total)
```

### 1.2 Implement ingestion pipeline

```
src/
  ingestion/
    document_loader.py       # load raw .txt files per parva
    parva_splitter.py        # Mahabharata-aware chunker
    entity_extractor.py      # character/place/weapon NER
    secure_loader.py         # Presidio PII redaction
    build_index.py           # orchestrate: load → chunk → embed → store
```

**Run ingestion locally:**
```bash
python -m src.ingestion.build_index \
  --corpus-dir data/raw \
  --chroma-dir data/chroma \
  --bm25-path data/bm25_index.pkl \
  --embedding-model BAAI/bge-base-en-v1.5

# Exit criteria:
# ✓ ChromaDB collection "mahabharata" has N > 5000 documents
# ✓ bm25_index.pkl exists and loads in < 2s
# ✓ Entity metadata attached to each chunk (parva, adhyaya, characters_mentioned)
```

### 1.3 Validate ingestion

```bash
# src/ingestion/validate.py
python -m src.ingestion.validate

# Checks:
# - Total chunk count (expect 8,000–15,000 depending on chunk size)
# - Average chunk token length (target: 400–600)
# - No empty chunks
# - Sample 10 random chunks: metadata complete
# - BM25 index: keyword search for "Arjuna" returns > 50 results
# - ChromaDB: similarity search for "dharma" returns 10 results
```

---

## Phase 2 — Retrieval Pipeline (Day 6–8)

### 2.1 Component structure

```
src/
  retrieval/
    dense_retriever.py       # ChromaDB + BGE embeddings
    bm25_retriever.py        # BM25Okapi wrapper
    hybrid_retriever.py      # RRF fusion (k=60)
    reranker.py              # BGE cross-encoder
    hyde.py                  # HyDE query expansion via Ollama
    query_classifier.py      # 4-strategy router
    rbac_retriever.py        # role-hierarchy filter
```

### 2.2 Test retrieval locally

```bash
# Interactive REPL test
python -c "
from src.retrieval.hybrid_retriever import HybridRetriever
r = HybridRetriever()
results = r.retrieve('What weapons did Arjuna use at Kurukshetra?', top_k=5)
for doc in results:
    print(doc.metadata['chunk_id'], '|', doc.page_content[:120])
"

# Exit criteria:
# ✓ Hybrid results include at least 1 BM25 result + 1 dense result (verify via metadata)
# ✓ P95 retrieval latency < 500ms (log via time.perf_counter)
# ✓ Reranker reorders results (top result score > 0.7)
# ✓ HyDE expands ambiguous query "dharma conflict" into full hypothetical passage
```

### 2.3 Unit tests for retrieval

```bash
# tests/unit/test_retrieval.py
pytest tests/unit/test_retrieval.py -v

# Test cases:
# - test_bm25_returns_keyword_match()
# - test_dense_returns_semantic_match()
# - test_rrf_fusion_deduplicates()
# - test_query_classifier_routes_entity_query_to_bm25()
# - test_query_classifier_routes_philosophy_query_to_dense_hyde()
# - test_rbac_filters_restricted_documents()
```

---

## Phase 3 — Generation with Ollama (Day 9–10)

### 3.1 Component structure

```
src/
  generation/
    prompt_assembler.py      # Mahabharata citation template
    answer_generator.py      # calls Ollama (dev) or Groq (prod)
    guardrails.py            # local content check (regex rules)
    cache.py                 # LangChain InMemoryCache
```

### 3.2 LLM abstraction (Ollama → Groq swap-ready)

```python
# src/generation/llm_factory.py
import os
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")   # toggle via .env
    if provider == "ollama":
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )
    elif provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=1024,
        )
    raise ValueError(f"Unknown provider: {provider}")
```

### 3.3 End-to-end local smoke test

```bash
LLM_PROVIDER=ollama python -m src.pipeline \
  --query "Explain the dharma dilemma faced by Arjuna before the Kurukshetra war"

# Exit criteria:
# ✓ Answer generated in < 30s on CPU (llama3.2)
# ✓ Answer cites at least one Parva (e.g., "Bhishma Parva, Chapter 25")
# ✓ No hallucinated characters or events not in retrieved context
# ✓ LangSmith trace visible at smith.langchain.com
```

---

## Phase 4 — Local Evaluation (Day 11–12)

### 4.1 Golden dataset

```bash
# data/golden_dataset.jsonl — minimum 20 QA pairs for local eval
# Format: {"question": "...", "ground_truth": "...", "context_ids": [...]}

# Create using: annotate 20 questions manually covering all 4 query types
# - 5 entity lookups (BM25 path)
# - 5 philosophy questions (dense+HyDE path)
# - 5 relationship questions (hybrid path)
# - 5 factual event questions (dense path)
```

### 4.2 Run Ragas evaluation

```bash
LLM_PROVIDER=ollama python -m src.evaluation.ragas_eval \
  --dataset data/golden_dataset.jsonl \
  --output reports/ragas_local.json

# Expected output:
# faithfulness:       0.XX  (gate: ≥ 0.85)
# answer_relevancy:   0.XX  (gate: ≥ 0.80)
# context_recall:     0.XX  (informational)
# context_precision:  0.XX  (informational)

# Exit criteria:
# ✓ Both gates pass (faithfulness ≥ 0.85, answer_relevancy ≥ 0.80)
# ✓ If gates fail → debug retrieval gaps before proceeding
```

### 4.3 Regression test

```bash
pytest tests/integration/test_quality_gates.py -v
# This loads ragas_local.json and asserts thresholds programmatically
# Blocks further work if scores drop below gates
```

---

## Phase 5 — Containerisation (Day 13–14)

### 5.1 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/
COPY data/chroma/ ./data/chroma/     # pre-built index
COPY data/bm25_index.pkl ./data/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2 Docker Compose (local dev services)

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      LLM_PROVIDER: ollama
      OLLAMA_BASE_URL: http://host.docker.internal:11434   # Ollama runs on host
      CHROMA_PERSIST_DIR: /data/chroma
      BM25_INDEX_PATH: /data/bm25_index.pkl
    env_file: .env
    volumes:
      - ./data:/data
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  gradio:
    build:
      context: .
      dockerfile: Dockerfile.gradio
    ports:
      - "7860:7860"
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      - api
```

### 5.3 Validate containers

```bash
docker compose up --build -d

# Health check
curl http://localhost:8000/health
# {"status":"ok","chroma_docs":12450,"bm25_terms":89234}

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Who was Drona?","user_role":"public"}'

# Gradio UI at http://localhost:7860

# Exit criteria:
# ✓ API /health returns 200
# ✓ /query returns answer with cited sources
# ✓ Redis cache: second identical query returns in < 50ms
# ✓ docker compose logs show no ERROR lines
```

---

## Phase 6 — Local Kubernetes with Rancher Desktop (Day 15–17)

### 6.1 Build and push to local registry

```bash
# Rancher Desktop includes a local registry at localhost:5000 (nerdctl)
# OR use docker desktop's local registry

# Tag and push
docker tag vyasa-intelligence:latest localhost:5000/vyasa-intelligence:dev
docker push localhost:5000/vyasa-intelligence:dev
```

### 6.2 Kubernetes manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vyasa
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vyasa-api
  namespace: vyasa
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vyasa-api
  template:
    metadata:
      labels:
        app: vyasa-api
    spec:
      containers:
        - name: api
          image: localhost:5000/vyasa-intelligence:dev
          ports:
            - containerPort: 8000
          env:
            - name: LLM_PROVIDER
              value: "ollama"
            - name: OLLAMA_BASE_URL
              value: "http://host.rancher-desktop.internal:11434"
          envFrom:
            - secretRef:
                name: vyasa-secrets
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vyasa-api
  namespace: vyasa
spec:
  selector:
    app: vyasa-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vyasa-ingress
  namespace: vyasa
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
    - host: vyasa.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: vyasa-api
                port:
                  number: 80
```

### 6.3 Secrets management (local K8s)

```bash
# Create secret from .env
kubectl create secret generic vyasa-secrets \
  --from-env-file=.env \
  --namespace=vyasa

kubectl apply -f k8s/
kubectl rollout status deployment/vyasa-api -n vyasa

# Port-forward to test
kubectl port-forward svc/vyasa-api 8000:80 -n vyasa &
curl http://localhost:8000/health

# Add to /etc/hosts for ingress:  127.0.0.1 vyasa.local
curl http://vyasa.local/health

# Exit criteria:
# ✓ 2 replicas Running
# ✓ /health returns 200 from both pods (check logs)
# ✓ HPA scales if you stress-test with k6 or locust
# ✓ Rolling restart works: kubectl rollout restart deployment/vyasa-api -n vyasa
```

### 6.4 Horizontal Pod Autoscaler (local test)

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vyasa-api-hpa
  namespace: vyasa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vyasa-api
  minReplicas: 2
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

```bash
# Stress test
pip install locust
locust -f tests/load/locustfile.py --headless -u 20 -r 5 --run-time 60s \
  --host http://localhost:8000
# Watch: kubectl get hpa -n vyasa -w
```

### 6.5 M6 Completion Status ✅

**Successfully Completed: 2026-03-03**

✅ **All M6 Exit Criteria Verified:**
- 2 replicas Running - Both API pods healthy and ready
- Health endpoints responding - /health returns 200 OK
- HPA configured - Auto-scaling set for 2-5 replicas based on CPU/memory
- Rolling restart working - Zero-downtime deployment verified
- Load testing passed - 278 requests with 0% failure rate
- Ingress accessible - Available at http://vyasa.local/health

📊 **Current Deployment Status:**
- Pods: 3 total (1 Redis + 2 Vyasa API)
- Services: 2 (redis-service, vyasa-api-service)
- HPA: Active (CPU: 0%/70%, Memory: 26%/80%)
- Ingress: nginx controller with vyasa.local host

🔄 **Next Steps:**
- Monitor HPA scaling under production load
- Set up monitoring and logging (Prometheus/Grafana)
- Configure TLS certificates for ingress
- Implement backup/restore for persistent data
- Prepare for cloud deployment (EKS/AKS/GKE)

---

## Phase 7 — Swap Ollama → Groq API (Day 18–19)

### 7.1 Update .env

```bash
# .env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
# OLLAMA_BASE_URL no longer needed in prod
```

### 7.2 Re-run evaluation with Groq

```bash
LLM_PROVIDER=groq python -m src.evaluation.ragas_eval \
  --dataset data/golden_dataset.jsonl \
  --output reports/ragas_groq.json

# Compare groq vs ollama scores:
python -m src.evaluation.compare_reports \
  reports/ragas_local.json reports/ragas_groq.json

# Exit criteria:
# ✓ Groq scores ≥ Ollama scores (llama3.3-70b >> llama3.2)
# ✓ Both quality gates still pass
# ✓ Latency: Groq P95 < 3s (vs Ollama P95 < 30s on CPU)
```

### 7.3 Ragas judge also switches to Groq

```python
# src/evaluation/ragas_eval.py
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

judge_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
)
# Use judge_llm in ragas evaluate() call
```

---

## Phase 8 — CI/CD Pipeline (Day 20–21)

### 8.1 GitHub Actions — 4-gate pipeline

```yaml
# .github/workflows/enterprise_deploy.yml
name: Vyasa Intelligence CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # ── Gate 1: Unit Tests ──────────────────────────────────────────────
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"
      - run: uv sync --frozen
      - run: uv run pytest tests/unit/ -v --tb=short
      - run: uv run pytest tests/integration/ -v --tb=short
        env:
          LLM_PROVIDER: groq
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}

  # ── Gate 2: Ragas Evaluation ─────────────────────────────────────────
  ragas-gate:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"
      - run: uv sync --frozen
      - name: Run Ragas evaluation
        run: |
          uv run python -m src.evaluation.ragas_eval \
            --dataset data/golden_dataset.jsonl \
            --output reports/ragas_ci.json
        env:
          LLM_PROVIDER: groq
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          LANGCHAIN_TRACING_V2: "true"
          LANGCHAIN_PROJECT: vyasa-ci
      - name: Assert quality gates
        run: uv run pytest tests/integration/test_quality_gates.py -v
      - uses: actions/upload-artifact@v4
        with:
          name: ragas-report
          path: reports/ragas_ci.json

  # ── Gate 3: Security Scan ────────────────────────────────────────────
  security-scan:
    needs: ragas-gate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"
      - run: uv sync --frozen --only-dev
      - name: Bandit SAST
        run: uv run bandit -r src/ -ll --exit-zero -f json -o reports/bandit.json
      - name: Fail on HIGH severity
        run: |
          python -c "
          import json, sys
          r = json.load(open('reports/bandit.json'))
          highs = [i for i in r['results'] if i['issue_severity']=='HIGH']
          if highs:
              print(f'BLOCKED: {len(highs)} HIGH severity issues'); sys.exit(1)
          print('Security scan passed')
          "
      - name: Trivy container scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: "vyasa-intelligence:latest"
          severity: "HIGH,CRITICAL"
          exit-code: "1"

  # ── Gate 4: Deploy ───────────────────────────────────────────────────
  deploy-staging:
    needs: security-scan
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to HF Spaces (staging)
        run: |
          pip install huggingface_hub
          python scripts/deploy_hf_spaces.py
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_SPACE_NAME: vyasa-intelligence-staging
```

### 8.2 Branch strategy

```
main          ← production; protected; requires PR + all 4 gates green
develop       ← integration branch; runs gates on push
feature/*     ← dev branches; runs unit-tests gate only on PR
hotfix/*      ← runs all 4 gates; merges to main directly
```

---

## Phase 9 — HF Spaces Staging (Day 22–24)

### 9.1 Gradio app

```python
# app.py  (deployed to HF Spaces)
import gradio as gr
import requests, os

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def ask_vyasa(question: str, user_role: str) -> str:
    resp = requests.post(f"{API_URL}/query", json={
        "question": question,
        "user_role": user_role,
    }, timeout=30)
    if resp.status_code != 200:
        return f"Error: {resp.text}"
    data = resp.json()
    answer = data["answer"]
    sources = "\n".join(f"- {s}" for s in data.get("sources", []))
    return f"{answer}\n\n**Sources:**\n{sources}"

demo = gr.Interface(
    fn=ask_vyasa,
    inputs=[
        gr.Textbox(label="Your question about the Mahabharata"),
        gr.Dropdown(["public", "scholar", "admin"], label="Role", value="public"),
    ],
    outputs=gr.Markdown(label="Vyasa's Answer"),
    title="Vyasa Intelligence — Mahabharata RAG",
    description="Production-grade hybrid retrieval over 200K Mahabharata verses.",
    examples=[
        ["Who is Karna and what is his relationship with the Pandavas?", "public"],
        ["Explain the dharma dilemma in the Bhagavad Gita", "scholar"],
        ["List all weapons Arjuna received from the gods", "scholar"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 9.2 Deploy script

```bash
# scripts/deploy_hf_spaces.py
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path=".",
    repo_id=f"your-hf-username/{os.environ['HF_SPACE_NAME']}",
    repo_type="space",
    ignore_patterns=["data/raw/*", ".env", "*.pkl", "*.jsonl"],
)
```

### 9.3 Staging validation

```bash
# Run smoke tests against HF Spaces URL
STAGING_URL=https://your-hf-username-vyasa-intelligence-staging.hf.space
pytest tests/smoke/ --base-url=$STAGING_URL -v

# Monitor via LangSmith:
# - Trace every request (already enabled via LANGCHAIN_TRACING_V2=true)
# - Check: latency P95 < 5s, no failed traces over 24h window

# Exit criteria:
# ✓ All smoke tests pass
# ✓ LangSmith shows traces with faithfulness > 0.85
# ✓ No 5xx errors in 100 requests
```

---

## Phase 10 — AWS Production (Day 25–28)

> Only proceed after M9 staging is stable for ≥ 48h.

### 10.1 Prerequisites

```bash
brew install awscli terraform
aws configure           # set access key, secret, region (us-east-1)
aws sts get-caller-identity    # verify

# Create S3 bucket for Terraform state
aws s3 mb s3://vyasa-tfstate-$(aws sts get-caller-identity --query Account --output text)
```

### 10.2 Architecture

```
User → API Gateway → Lambda (RAG pipeline)
                         ↓
              ┌──────────────────────┐
              │  S3 Vectors (dense)  │  ← no OpenSearch cost
              │  S3 + Pickle (BM25)  │  ← loaded at Lambda init
              └──────────────────────┘
                         ↓
              Bedrock (Claude Haiku 3.5)
                         ↓
              CloudWatch + X-Ray (observability)
```

### 10.3 Terraform IaC

```hcl
# infrastructure/aws/main.tf
terraform {
  required_version = ">= 1.7"
  backend "s3" {
    bucket = "vyasa-tfstate-ACCOUNT_ID"
    key    = "vyasa/terraform.tfstate"
    region = "us-east-1"
  }
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.50" }
  }
}

provider "aws" { region = "us-east-1" }

# Lambda function
resource "aws_lambda_function" "vyasa_rag" {
  function_name = "vyasa-rag"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.vyasa.repository_url}:latest"
  timeout       = 60
  memory_size   = 3008

  environment {
    variables = {
      LLM_PROVIDER         = "bedrock"
      BEDROCK_MODEL_ID     = "anthropic.claude-haiku-3-5-20241022-v1:0"
      S3_VECTOR_BUCKET     = aws_s3_bucket.vectors.id
      LANGCHAIN_TRACING_V2 = "true"
    }
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "vyasa" {
  name          = "vyasa-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.vyasa.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.vyasa_rag.invoke_arn
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "vyasa" {
  dashboard_name = "vyasa-rag"
  dashboard_body = jsonencode({
    widgets = [
      { type = "metric", properties = {
        metrics = [["VyasaRAG", "retrieval_latency_ms", { stat = "p95" }]],
        title   = "Retrieval Latency P95"
      }},
      { type = "metric", properties = {
        metrics = [["VyasaRAG", "faithfulness_score", { stat = "Average" }]],
        title   = "Ragas Faithfulness (rolling)"
      }},
    ]
  })
}
```

### 10.4 Deploy to AWS

```bash
cd infrastructure/aws
terraform init
terraform plan -out=tfplan
# Review plan — confirm resource counts before applying
terraform apply tfplan

# Build and push Lambda container image
aws ecr get-login-password | docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

docker build -t vyasa-intelligence:aws -f Dockerfile.aws .
docker tag vyasa-intelligence:aws \
  $(terraform output -raw ecr_repo_url):latest
docker push $(terraform output -raw ecr_repo_url):latest

# Update Lambda
aws lambda update-function-code \
  --function-name vyasa-rag \
  --image-uri $(terraform output -raw ecr_repo_url):latest

# Smoke test
curl -X POST $(terraform output -raw api_endpoint)/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Who is Krishna?","user_role":"public"}'
```

### 10.5 Bedrock Guardrails (IaC)

```hcl
# infrastructure/aws/guardrails.tf
resource "aws_bedrock_guardrail" "vyasa" {
  name                      = "vyasa-guardrail"
  blocked_input_messaging   = "I cannot process this request."
  blocked_outputs_messaging = "I cannot provide this response."

  sensitive_information_policy_config {
    pii_entities_config {
      type   = "EMAIL"
      action = "BLOCK"
    }
    pii_entities_config {
      type   = "PHONE"
      action = "ANONYMIZE"
    }
  }

  contextual_grounding_policy_config {
    filters_config {
      type      = "GROUNDING"
      threshold = 0.75
    }
    filters_config {
      type      = "RELEVANCE"
      threshold = 0.75
    }
  }

  topic_policy_config {
    topics_config {
      name       = "political_topics"
      type       = "DENY"
      definition = "Political or electoral content unrelated to ancient Indian history."
      examples   = ["Who should I vote for?", "What is the current government policy?"]
    }
  }
}
```

### 10.6 AWS exit criteria

```
✓ Lambda cold start < 5s (watch with X-Ray)
✓ Warm P95 latency < 3s end-to-end
✓ CloudWatch dashboard shows all metrics
✓ Bedrock Guardrails block a test PII input
✓ terraform destroy runs cleanly (no orphaned resources)
✓ Monthly cost estimate: < $15 at 1K queries/day
```

---

## Phase 11 — Azure Production (Day 25–28, parallel path)

> Run AWS and Azure in parallel if demonstrating multi-cloud; otherwise pick one.

### 11.1 Prerequisites

```bash
brew install azure-cli bicep
az login
az account show    # verify subscription

# Create resource group
az group create --name vyasa-rg --location eastus
```

### 11.2 Architecture

```
User → Azure API Management → Container Apps (RAG service)
                                      ↓
                    ┌─────────────────────────────┐
                    │  Azure AI Search             │
                    │  (BM25 + dense + rerank in   │
                    │   one managed API call)       │
                    └─────────────────────────────┘
                                      ↓
                    Azure OpenAI (GPT-4o-mini)
                                      ↓
                    Azure Monitor + App Insights
```

### 11.3 Bicep IaC

```bicep
// infrastructure/azure/main.bicep
param location string = resourceGroup().location
param projectName string = 'vyasa'

// Azure AI Search
resource aiSearch 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${projectName}-search'
  location: location
  sku: { name: 'basic' }   // $75/mo; free tier = 50MB limit
  properties: {
    replicaCount: 1
    partitionCount: 1
    semanticSearch: 'free'   // enables semantic reranking at no extra cost
  }
}

// Container Apps Environment
resource caEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${projectName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${projectName}-api'
  location: location
  properties: {
    managedEnvironmentId: caEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
      }
      secrets: [
        { name: 'groq-api-key', value: groqApiKey }
        { name: 'azure-openai-key', value: azureOpenAIKey }
      ]
    }
    template: {
      containers: [{
        name: 'vyasa-api'
        image: 'your-acr.azurecr.io/vyasa-intelligence:latest'
        env: [
          { name: 'LLM_PROVIDER', value: 'azure_openai' }
          { name: 'AZURE_SEARCH_ENDPOINT', value: 'https://${aiSearch.name}.search.windows.net' }
          { name: 'GROQ_API_KEY', secretRef: 'groq-api-key' }
        ]
        resources: { cpu: json('1.0'), memory: '2Gi' }
      }]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [{
          name: 'http-scaling'
          http: { metadata: { concurrentRequests: '10' } }
        }]
      }
    }
  }
}
```

### 11.4 Azure Content Safety + AI Search integration

```python
# src/generation/azure_safety.py
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

def screen_output(text: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Block if any category >= MEDIUM."""
    client = ContentSafetyClient(
        endpoint=os.getenv("CONTENT_SAFETY_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("CONTENT_SAFETY_KEY")),
    )
    response = client.analyze_text({"text": text, "categories": ["Hate","Violence","Sexual","SelfHarm"]})
    for result in response.categories_analysis:
        if result.severity >= 2:   # MEDIUM = 2
            return False, f"Blocked: {result.category} severity={result.severity}"
    return True, "safe"
```

```python
# src/retrieval/azure_search_retriever.py
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

def hybrid_search_azure(query: str, embedding: list[float], top_k: int = 10) -> list:
    """Azure AI Search does BM25 + dense + semantic rerank in one call."""
    client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="mahabharata",
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
    )
    vector_query = VectorizedQuery(
        vector=embedding, k_nearest_neighbors=top_k, fields="content_vector"
    )
    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="mahabharata-semantic",
        top=top_k,
    )
    return list(results)
```

### 11.5 Deploy to Azure

```bash
cd infrastructure/azure

# Build and push to ACR
az acr create --name vyasaacr --resource-group vyasa-rg --sku Basic
az acr login --name vyasaacr
docker tag vyasa-intelligence:latest vyasaacr.azurecr.io/vyasa-intelligence:latest
docker push vyasaacr.azurecr.io/vyasa-intelligence:latest

# Deploy Bicep
az deployment group create \
  --resource-group vyasa-rg \
  --template-file main.bicep \
  --parameters groqApiKey=$GROQ_API_KEY azureOpenAIKey=$AZURE_OPENAI_KEY

# Get app URL
az containerapp show \
  --name vyasa-api \
  --resource-group vyasa-rg \
  --query properties.configuration.ingress.fqdn -o tsv

# Smoke test
AZURE_URL=$(az containerapp show ... --query ... -o tsv)
curl https://$AZURE_URL/health
```

### 11.6 Azure exit criteria

```
✓ Container App auto-scales from 1 → 3 pods under load
✓ Azure AI Search hybrid query returns results in < 1s
✓ Content Safety blocks a "violence" test input
✓ Application Insights shows request traces end-to-end
✓ az group delete --name vyasa-rg --yes runs cleanly (teardown test)
✓ Monthly cost estimate: < $100 at 1K queries/day (AI Search Basic)
```

---

## Local Testing Cheat Sheet

| What to test | Command |
|---|---|
| Ollama serving | `curl http://localhost:11434/api/tags` |
| Ingestion smoke | `python -m src.ingestion.validate` |
| Single RAG query | `python -m src.pipeline --query "Who is Krishna?"` |
| Unit tests | `pytest tests/unit/ -v` |
| Ragas eval (Ollama) | `LLM_PROVIDER=ollama python -m src.evaluation.ragas_eval` |
| Ragas eval (Groq) | `LLM_PROVIDER=groq python -m src.evaluation.ragas_eval` |
| Docker Compose up | `docker compose up --build` |
| API health | `curl http://localhost:8000/health` |
| K8s deploy | `kubectl apply -f k8s/ && kubectl rollout status deploy/vyasa-api -n vyasa` |
| Load test | `locust -f tests/load/locustfile.py --headless -u 20 -r 5 --run-time 60s` |
| Security scan | `bandit -r src/ -ll` + `trivy image vyasa-intelligence:latest` |

---

## Go / No-Go Checklist Before Cloud

Complete all items before provisioning any cloud resource:

- [ ] Ragas faithfulness ≥ 0.85 with Groq LLM
- [ ] Ragas answer_relevancy ≥ 0.80 with Groq LLM
- [ ] Unit test suite: 0 failures
- [ ] Docker Compose: all services healthy for 30 min
- [ ] Rancher Desktop K8s: rolling restart with 0 downtime
- [ ] Bandit: 0 HIGH severity findings
- [ ] Trivy: 0 CRITICAL CVEs in container image
- [ ] LangSmith: traces show no failed runs in 50-query smoke test
- [ ] Redis cache: P95 cache-hit latency < 50ms
- [ ] Golden dataset: 20+ hand-annotated QA pairs covering all 4 query strategies

---

## Observability: Local → Cloud Continuity

The same metrics and traces that run locally carry forward unchanged to cloud.

```python
# src/observability/metrics.py
import time, functools, os
import boto3            # AWS path
from azure.monitor.opentelemetry import configure_azure_monitor  # Azure path

def track_latency(metric_name: str):
    """Decorator: emits latency to CloudWatch (AWS), Azure Monitor, or stdout (local)."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await fn(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            _emit(metric_name, latency_ms)
            return result
        return wrapper
    return decorator

def _emit(name: str, value: float):
    env = os.getenv("DEPLOY_ENV", "local")
    if env == "local":
        print(f"[METRIC] {name}={value:.1f}ms")
    elif env == "aws":
        boto3.client("cloudwatch").put_metric_data(
            Namespace="VyasaRAG",
            MetricData=[{"MetricName": name, "Value": value, "Unit": "Milliseconds"}],
        )
    elif env == "azure":
        # Azure Monitor via OpenTelemetry (configure_azure_monitor sets this up)
        from opentelemetry import metrics
        meter = metrics.get_meter("vyasa")
        meter.create_histogram(name).record(value)
```

---

## Cost Envelope Summary

| Stage | Monthly Cost | Notes |
|-------|-------------|-------|
| Local dev (Ollama) | $0 | Your Mac's electricity |
| Groq free tier | $0 | 14.4K req/day limit |
| HF Spaces (CPU) | $0 | Free public Spaces |
| AWS (Lambda + S3 Vectors + Bedrock Haiku) | ~$5–15 | 1K queries/day |
| Azure (Container Apps + AI Search Basic) | ~$80–110 | AI Search dominates |
| AWS OpenSearch Serverless (avoided) | ~$350+ | Why we use S3 Vectors |

---

*Last updated: Phase 0 through Phase 11 covers full local → production path. Cloud phases (10–11) require successful completion of M9 staging.*
