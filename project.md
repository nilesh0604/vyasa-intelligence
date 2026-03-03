# Vyasa Intelligence — Production RAG System

**Vyasa** (Veda Vyasa) is the author of the Mahabharata. This project builds a production-grade RAG system over the full Mahabharata corpus (~200,000 verses, ~4–5M English tokens), demonstrating every layer of a real enterprise pipeline: structured ingestion, hybrid retrieval, reranking, evaluation gating, observability, and cloud deployment.

---

## Portfolio Context

This system is one of five interconnected projects that form a deliberate engineering stack:

| # | Project | Core Skill |
|---|---------|------------|
| 1 | **Production-Grade RAG** ← this repo | Ingestion → hybrid retrieval → evaluation CI/CD |
| 2 | Local AI Assistant (SLMs) | Benchmark SLMs locally; cut cloud costs |
| 3 | Monitoring & Observability | SRE-grade metrics, trace visualization, regression gating |
| 4 | Specialized Fine-Tuning (SFT + DPO) | LoRA, DPO, GGUF deployment |
| 5 | Real-Time Multimodal Application | Sub-500ms voice pipeline (Deepgram → LLM → ElevenLabs) |

The logs, evaluation pipelines, and performance charts across all five are what signal production-grade thinking — not the UI.

---

## Stack & Cost

### Free / Local Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM Inference | Groq `llama-3.3-70b-versatile` | Free (14.4K req/day) |
| Embeddings | `BAAI/bge-base-en-v1.5` (local HuggingFace) | Free |
| Vector DB | ChromaDB (persistent local) | Free |
| BM25 | `rank_bm25` Python lib | Free |
| Reranker | `BAAI/bge-reranker-v2-m3` (local) or Cohere Rerank (1K/mo free) | Free |
| ColBERT Reranker | `ragatouille` (ColBERT v2) | Free |
| Evaluation | `ragas` OSS package | Free |
| CI/CD | GitHub Actions | Free (2,000 min/mo) |
| Hosting | Hugging Face Spaces (Gradio) or Render free | Free |

> **Raspberry Pi 5 note:** BGE-base and BGE-reranker-base run comfortably on 8GB RAM. Use Groq API for LLM inference rather than local generation to maintain quality.

### Cloud Comparison

| Dimension | Free Stack | AWS Lean Serverless | Azure Cloud-Native |
|-----------|-----------|--------------------|--------------------|
| Monthly cost | ~$0 | ~$5–$40 | ~$10–$75 |
| Time to first deploy | 1–2 days | 2–3 days | 2–3 days |
| Scales to prod | Pi/Render limits | Yes, infinitely | Yes, infinitely |
| Managed infra | None | Mostly | Fully |
| Best for | Learning + demos | AWS-native teams | Azure / MS shops |

> **Never start with Bedrock Knowledge Bases + OpenSearch Serverless** — OpenSearch has a $350/month floor with zero pipeline visibility. AWS S3 Vectors (launched 2025) cuts this to ~$35/month.

---

## Repository Structure

```
vyasa-intelligence/
├── .github/workflows/
│   └── enterprise_deploy.yml    # CI/CD gate + environment promotion
├── data/
│   ├── raw/
│   │   └── mahabharata.pdf
│   └── processed/
│       ├── chunks/              # JSONL chunks with parva metadata
│       └── golden_dataset.jsonl # 50–200 Q&A triplets
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   ├── parva_splitter.py    # Mahabharata-specific hierarchical chunker
│   │   ├── entity_extractor.py  # Characters, places, weapons, concepts
│   │   ├── secure_loader.py     # PII redaction (enterprise)
│   │   └── query_classifier.py  # Routes bm25 / dense / hybrid / hyde
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py  # Reciprocal Rank Fusion
│   │   ├── reranker.py          # BGE cross-encoder / ColBERT
│   │   ├── hyde.py              # Hypothetical Document Embeddings
│   │   └── rbac_retriever.py    # Role-based access filtering (enterprise)
│   ├── generation/
│   │   ├── prompt_assembler.py  # Stable Parva-citing template
│   │   ├── answer_generator.py
│   │   ├── guardrails.py        # Bedrock Guardrails (AWS enterprise)
│   │   └── content_safety.py    # Azure Content Safety (Azure enterprise)
│   ├── evaluation/
│   │   ├── golden_dataset.jsonl
│   │   └── ragas_eval.py
│   ├── observability/
│   │   └── metrics.py
│   └── pipeline.py              # Full pipeline wiring
├── tests/
│   ├── unit/
│   ├── integration/
│   └── test_quality_gates.py
├── terraform/                   # AWS IaC
├── bicep/                       # Azure IaC
├── Dockerfile
└── requirements.txt
```

---

## Phase 1 — Ingestion & Chunking

```bash
pip install langchain langchain-community chromadb rank_bm25 \
  sentence-transformers PyMuPDF unstructured groq ragas ragatouille \
  presidio-analyzer presidio-anonymizer
```

### `src/ingestion/document_loader.py`

```python
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredMarkdownLoader, TextLoader
)
from pathlib import Path

def load_documents(data_dir: str) -> list:
    docs = []
    for path in Path(data_dir).rglob("*"):
        if path.suffix == ".pdf":
            docs.extend(PyMuPDFLoader(str(path)).load())
        elif path.suffix in (".md", ".markdown"):
            docs.extend(UnstructuredMarkdownLoader(str(path)).load())
        elif path.suffix == ".txt":
            docs.extend(TextLoader(str(path)).load())
    return docs
```

### `src/ingestion/parva_splitter.py` — Mahabharata-aware hierarchical chunker

Standard `RecursiveCharacterTextSplitter` blindly cuts mid-verse and across Parva boundaries. The Mahabharata's structure requires respecting its hierarchy:

```
Mahabharata
└── 18 Parvas (Books) + Harivamsa appendix
    └── 100 Sub-Parvas
        └── Adhyayas (Chapters) — ~2,000 total
            └── Shlokas (Verses) — ~200,000 total
```

```python
import re
from langchain.schema import Document

PARVA_NAMES = [
    "Adi Parva", "Sabha Parva", "Vana Parva", "Virata Parva",
    "Udyoga Parva", "Bhishma Parva", "Drona Parva", "Karna Parva",
    "Shalya Parva", "Sauptika Parva", "Stri Parva", "Shanti Parva",
    "Anushasana Parva", "Ashvamedhika Parva", "Ashramavasika Parva",
    "Mausala Parva", "Mahaprasthanika Parva", "Svargarohana Parva"
]

PARVA_PATTERN = re.compile(
    r'(' + '|'.join(PARVA_NAMES) + r'|BOOK\s+[IVXLC]+|SECTION\s+[IVXLC]+\b)',
    re.IGNORECASE
)

ADHYAYA_PATTERN = re.compile(
    r'(SECTION\s+[IVXLC]+|Chapter\s+\d+|Adhyaya\s+\d+)',
    re.IGNORECASE
)

def split_by_parva(full_text: str) -> list[dict]:
    segments = PARVA_PATTERN.split(full_text)
    parvas = []
    current_parva = "Adi Parva"
    for seg in segments:
        if PARVA_PATTERN.match(seg.strip()):
            current_parva = seg.strip()
        elif seg.strip():
            parvas.append({"parva": current_parva, "text": seg.strip()})
    return parvas

def chunk_parva(parva_text: str, parva_name: str,
                chunk_size: int = 600, overlap: int = 100) -> list[Document]:
    """
    Chunk within a Parva at adhyaya boundaries first,
    then fall back to recursive character splitting.
    Priority: adhyaya break > paragraph > sentence > character
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    adhyaya_sections = ADHYAYA_PATTERN.split(parva_text)
    chunks = []
    current_section = "Opening"

    for seg in adhyaya_sections:
        if ADHYAYA_PATTERN.match(seg.strip()):
            current_section = seg.strip()
        elif seg.strip():
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            sub_chunks = splitter.create_documents([seg.strip()])
            for i, chunk in enumerate(sub_chunks):
                chunk.metadata.update({
                    "parva": parva_name,
                    "section": current_section,
                    "chunk_id": f"{parva_name.replace(' ', '_')}_{current_section}_{i:03d}",
                    "source": "mahabharata.pdf"
                })
            chunks.extend(sub_chunks)
    return chunks

def build_mahabharata_chunks(pdf_text: str) -> list[Document]:
    parvas = split_by_parva(pdf_text)
    all_chunks = []
    for parva in parvas:
        chunks = chunk_parva(parva["text"], parva["parva"])
        all_chunks.extend(chunks)
    print(f"Total chunks created: {len(all_chunks)}")
    # At 600 tokens/chunk → ~8,000–10,000 chunks total
    return all_chunks
```

### `src/ingestion/entity_extractor.py`

The Mahabharata has thousands of named entities. Injecting them into chunk metadata dramatically improves BM25 recall for entity queries ("What weapons did Arjuna use?").

```python
import re

MAHABHARATA_ENTITIES = {
    "characters": [
        "Arjuna", "Karna", "Bhima", "Yudhishthira", "Nakula", "Sahadeva",
        "Duryodhana", "Dushashana", "Shakuni", "Drona", "Bhishma", "Kripa",
        "Krishna", "Balarama", "Draupadi", "Kunti", "Gandhari", "Dhritarashtra",
        "Pandu", "Vyasa", "Ashwatthama", "Abhimanyu", "Ghatotkacha",
        "Shikhandi", "Dhrishtadyumna", "Virata", "Drupada", "Hidimba"
    ],
    "weapons": [
        "Gandiva", "Pashupatastra", "Brahmastra", "Brahmanda astra",
        "Narayanastra", "Sudarshana Chakra", "Vijaya", "Kaumodaki"
    ],
    "places": [
        "Kurukshetra", "Hastinapura", "Indraprastha", "Dwarka", "Panchala",
        "Matsya", "Kashi", "Anga", "Mathura"
    ],
    "concepts": [
        "Dharma", "Karma", "Moksha", "Artha", "Kama", "Ahimsa",
        "Bhagavad Gita", "Yoga", "Brahman", "Atman"
    ]
}

ALL_ENTITIES = [e for group in MAHABHARATA_ENTITIES.values() for e in group]
ENTITY_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(e) for e in ALL_ENTITIES) + r')\b')

def extract_entities(text: str) -> dict:
    found = set(ENTITY_PATTERN.findall(text))
    result = {}
    for category, entities in MAHABHARATA_ENTITIES.items():
        matched = [e for e in entities if e in found]
        if matched:
            result[category] = matched
    return result

def enrich_chunk_metadata(chunk) -> None:
    entities = extract_entities(chunk.page_content)
    chunk.metadata["entities"] = entities
    chunk.metadata["entity_names"] = ", ".join(
        e for group in entities.values() for e in group
    )
```

### `src/ingestion/query_classifier.py` — Mahabharata edition

Four distinct query patterns require different retrieval strategies:

```python
import re

# Entity names → BM25 (exact match)
ENTITY_PATTERN = re.compile(
    r'\b(Arjuna|Karna|Bhima|Krishna|Drona|Bhishma|Duryodhana|Gandiva|'
    r'Kurukshetra|Hastinapura|Draupadi|Yudhishthira|Ashwatthama|'
    r'Brahmastra|Bhagavad Gita)\b', re.IGNORECASE
)

# Philosophical/thematic → dense + HyDE
PHILOSOPHY_PATTERN = re.compile(
    r'\b(dharma|karma|moksha|duty|righteousness|soul|atman|brahman|'
    r'yoga|meditation|liberation|sin|virtue|morality|meaning|purpose)\b',
    re.IGNORECASE
)

# Relationship/lineage → multi-hop (full RRF pipeline)
RELATIONSHIP_PATTERN = re.compile(
    r'\b(father|mother|brother|sister|son|daughter|wife|husband|'
    r'teacher|student|guru|disciple|friend|enemy|ally|born|lineage)\b',
    re.IGNORECASE
)

def classify_query(query: str) -> dict:
    """
    Returns routing strategy:
    - 'bm25'       → entity/name keyword search
    - 'dense'      → semantic/philosophical
    - 'dense+hyde' → ambiguous philosophical (HyDE improves recall)
    - 'hybrid'     → relationship/multi-hop (full RRF pipeline)

    Examples:
    "Who is Karna?"                   → bm25
    "What is the meaning of dharma?"  → dense+hyde
    "Who is Arjuna's father?"         → hybrid
    "Describe the Kurukshetra war"    → hybrid
    """
    if ENTITY_PATTERN.search(query):
        return {"strategy": "bm25", "use_hyde": False}
    if PHILOSOPHY_PATTERN.search(query.lower()) and len(query.split()) > 6:
        return {"strategy": "dense", "use_hyde": True}
    if RELATIONSHIP_PATTERN.search(query.lower()):
        return {"strategy": "hybrid", "use_hyde": False}
    return {"strategy": "dense", "use_hyde": False}
```

### `src/ingestion/secure_loader.py` — Enterprise PII redaction

```python
import boto3
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from datetime import datetime

analyzer  = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> tuple[str, list]:
    """Redact PII before embedding — GDPR/HIPAA requirement."""
    results = analyzer.analyze(text=text, language="en",
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                  "CREDIT_CARD", "US_SSN", "LOCATION"])
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, [r.entity_type for r in results]

def load_and_secure(file_path: str, user_role: str) -> dict:
    raw_text = extract_text(file_path)
    clean_text, pii_found = redact_pii(raw_text)
    return {
        "content": clean_text,
        "metadata": {
            "source": file_path,
            "access_role": user_role,
            "pii_types_redacted": pii_found,
            "ingested_at": datetime.utcnow().isoformat(),
            "encryption": "AES-256"
        }
    }
```

---

## Phase 2 — Hybrid Retrieval + Reranking

Run BM25 and dense retrieval in parallel, fuse with Reciprocal Rank Fusion, then rerank with a cross-encoder. Add HyDE for ambiguous philosophical queries. This stack beats embedding-only models on precision by re-scoring candidates with a full attention pass over the query+document pair.

### `src/retrieval/dense_retriever.py`

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

def build_dense_index(chunks: list, persist_dir: str = "./chroma_db") -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

def dense_search(vectorstore: Chroma, query: str, k: int = 20) -> list:
    return vectorstore.similarity_search_with_score(query, k=k)
```

### `src/retrieval/bm25_retriever.py`

```python
from rank_bm25 import BM25Okapi
from langchain.schema import Document

class BM25Index:
    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices]
```

### `src/retrieval/hybrid_retriever.py` — Reciprocal Rank Fusion

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    bm25_results: list, dense_results: list, k: int = 60, top_n: int = 10
) -> list:
    """RRF score = sum(1 / (k + rank)) across ranked lists. k=60 is the empirical default."""
    scores = defaultdict(float)
    doc_map = {}

    for rank, (doc, _) in enumerate(bm25_results):
        cid = doc.metadata["chunk_id"]
        scores[cid] += 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    for rank, (doc, _) in enumerate(dense_results):
        cid = doc.metadata["chunk_id"]
        scores[cid] += 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[cid] for cid, _ in ranked[:top_n]]
```

### `src/retrieval/reranker.py` — BGE Cross-Encoder with ColBERT fallback

```python
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # ~570MB, runs on CPU

class BGEReranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, docs: list, top_k: int = 5) -> list:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

# Optional: ColBERT via ragatouille (higher precision, heavier, better for complex technical docs)
# from ragatouille import RAGPretrainedModel
# colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# results = colbert.rerank(query=query, documents=[d.page_content for d in docs], k=5)
```

### `src/retrieval/hyde.py` — Hypothetical Document Embeddings

Generates a synthetic answer and embeds *that* for retrieval — improves recall on ambiguous philosophical queries.

```python
from groq import Groq

client = Groq()

HYDE_PROMPT = """Write a short passage (2-3 sentences) that would directly answer:
"{query}"
Write only the passage, no preamble."""

def generate_hypothetical_doc(query: str) -> str:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
        max_tokens=200,
        temperature=0.3
    )
    return resp.choices[0].message.content
```

### `src/retrieval/rbac_retriever.py` — Enterprise role-based access filtering

```python
ROLE_HIERARCHY = {
    "admin":   ["admin", "manager", "analyst", "viewer"],
    "manager": ["manager", "analyst", "viewer"],
    "analyst": ["analyst", "viewer"],
    "viewer":  ["viewer"],
}

def get_allowed_roles(user_role: str) -> list[str]:
    return ROLE_HIERARCHY.get(user_role, ["viewer"])

def rbac_search(vectorstore, query: str, user_role: str, k: int = 20):
    """Filter retrieval to only chunks the user's role can access."""
    return vectorstore.similarity_search(
        query, k=k,
        filter={"access_role": {"$in": get_allowed_roles(user_role)}}
    )
```

---

## Phase 3 — Generation

### `src/generation/prompt_assembler.py` — Mahabharata edition

```python
SYSTEM_PROMPT = """You are Vyasa Intelligence, an expert on the Mahabharata.
Answer ONLY using the provided context passages from the Mahabharata.
Every factual claim must be cited using the format [Parva Name, Section].
If the context does not contain sufficient information, say:
"This information is not present in the retrieved passages of the Mahabharata."
Never fabricate events, characters, or verses."""

ANSWER_TEMPLATE = """## Retrieved Passages from the Mahabharata
{context}

## Question
{question}

## Instructions
- Answer using ONLY the passages above.
- Cite each claim as [Parva, Section] — e.g., [Bhishma Parva, Section XXIX].
- For philosophical questions, quote relevant shlokas if present in context.
- End with a **Sources** section listing all Parvas cited.

## Answer"""

def assemble_prompt(question: str, chunks: list) -> tuple[str, list[str]]:
    context_parts = []
    cited_ids = []
    for chunk in chunks:
        cid      = chunk.metadata.get("chunk_id", "unknown")
        parva    = chunk.metadata.get("parva", "Unknown Parva")
        sec      = chunk.metadata.get("section", "Unknown Section")
        entities = chunk.metadata.get("entity_names", "")
        context_parts.append(
            f"[{parva}, {sec}] (entities: {entities})\n{chunk.page_content}"
        )
        cited_ids.append(cid)
    context_block = "\n\n---\n\n".join(context_parts)
    return ANSWER_TEMPLATE.format(context=context_block, question=question), cited_ids
```

### `src/generation/answer_generator.py`

```python
from groq import Groq
from src.generation.prompt_assembler import SYSTEM_PROMPT, assemble_prompt

client = Groq()

def generate_answer(question: str, chunks: list) -> dict:
    user_prompt, cited_ids = assemble_prompt(question, chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    return {
        "answer": response.choices[0].message.content,
        "cited_chunk_ids": cited_ids,
        "model": "llama-3.3-70b-versatile"
    }
```

### `src/generation/guardrails.py` — AWS Bedrock enterprise safety

```python
import boto3, json, os

bedrock_runtime   = boto3.client("bedrock-runtime", region_name="us-east-1")
GUARDRAIL_ID      = "your-guardrail-id"
GUARDRAIL_VERSION = "DRAFT"

def generate_with_guardrails(prompt: str, user_role: str) -> dict:
    """Blocks prompt injection, PII leaks, and answers not grounded in retrieved context."""
    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-haiku-3-5-v1:0",
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
        })
    )
    result = json.loads(response["body"].read())
    if result.get("amazon-bedrock-guardrailAction") == "INTERVENED":
        log_guardrail_intervention(prompt, user_role)
        return {"answer": "This query was blocked by safety policy.", "blocked": True}
    return {"answer": result["content"][0]["text"], "blocked": False}
```

Guardrail Terraform configuration:

```hcl
resource "aws_bedrock_guardrail" "rag_guardrail" {
  name        = "prod-rag-guardrail"
  description = "Enterprise safety for RAG outputs"

  sensitive_information_policy_config {
    pii_entities_config { action = "BLOCK";     type = "EMAIL" }
    pii_entities_config { action = "ANONYMIZE"; type = "NAME"  }
  }

  contextual_grounding_policy_config {
    filters_config { type = "GROUNDING"; threshold = 0.85 }
    filters_config { type = "RELEVANCE"; threshold = 0.80 }
  }

  topic_policy_config {
    topics_config {
      name       = "off-topic-requests"
      definition = "Requests outside the knowledge base domain"
      type       = "DENY"
    }
  }
}
```

### `src/generation/content_safety.py` — Azure Content Safety

```python
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
import os

safety_client = ContentSafetyClient(
    endpoint=os.environ["CONTENT_SAFETY_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["CONTENT_SAFETY_KEY"])
)

def check_output_safety(answer: str) -> dict:
    """Screens LLM output for hate, violence, self-harm, sexual content."""
    response = safety_client.analyze_text(AnalyzeTextOptions(text=answer))
    violations = [
        cat for cat in [
            response.hate_result, response.violence_result,
            response.self_harm_result, response.sexual_result
        ]
        if cat and cat.severity >= 2  # 0=safe, 2=low, 4=medium, 6=high
    ]
    if violations:
        log_safety_violation(answer, violations)
        return {"safe": False, "answer": "Response blocked by content policy."}
    return {"safe": True, "answer": answer}
```

---

## Phase 4 — Evaluation

### `data/processed/golden_dataset.jsonl`

Build 50–200 entries covering all complexity levels. Annotate `complexity` and `query_type` — Ragas degrades differently per level and you want separate metric breakdowns in CI.

```jsonl
{"question": "Who is Karna's biological mother?", "ground_truth": "Karna's biological mother is Kunti, who conceived him through a divine boon from the sun god Surya before her marriage to Pandu.", "reference_chunk_ids": ["Adi_Parva_Section_IV_002"], "complexity": "simple", "query_type": "bm25"}
{"question": "What was the name of Arjuna's bow?", "ground_truth": "Arjuna's legendary bow is called Gandiva, gifted to him by the fire god Agni.", "reference_chunk_ids": ["Adi_Parva_Section_XII_001"], "complexity": "simple", "query_type": "bm25"}
{"question": "What does the Bhagavad Gita teach about performing one's duty without attachment to results?", "ground_truth": "Krishna teaches Nishkama Karma — performing one's duty (svadharma) without attachment to fruits or outcomes, surrendering results to the divine.", "reference_chunk_ids": ["Bhishma_Parva_Section_XXIX_003", "Bhishma_Parva_Section_XXX_001"], "complexity": "complex", "query_type": "dense+hyde"}
{"question": "How are Duryodhana and the Pandavas related?", "ground_truth": "Duryodhana is the son of Dhritarashtra and Gandhari. The Pandavas are sons of Pandu, Dhritarashtra's younger brother, making them cousins.", "reference_chunk_ids": ["Adi_Parva_Section_VI_004"], "complexity": "simple", "query_type": "hybrid"}
{"question": "Compare Bhishma's and Karna's reasons for not becoming king of Hastinapura.", "ground_truth": "Bhishma voluntarily renounced kingship through the Terrible Oath (Bhishma Pratigya) to allow his father Shantanu to remarry. Karna was denied kingship due to his perceived low-caste birth, not by personal choice.", "reference_chunk_ids": ["Adi_Parva_Section_III_002", "Adi_Parva_Section_IX_007"], "complexity": "adversarial", "query_type": "hybrid"}
{"question": "What role did Shakuni play in the game of dice?", "ground_truth": "Shakuni manipulated the dice game against Yudhishthira using loaded dice, causing the Pandavas to lose their kingdom, wealth, and Draupadi.", "reference_chunk_ids": ["Sabha_Parva_Section_II_005", "Sabha_Parva_Section_III_001"], "complexity": "complex", "query_type": "bm25"}
```

### `src/evaluation/ragas_eval.py`

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_recall, context_precision
)
import json

FAITHFULNESS_THRESHOLD    = 0.85
ANSWER_RELEVANCY_THRESHOLD = 0.80

def load_golden_dataset(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def run_ragas_evaluation(pipeline_fn, dataset_path: str) -> dict:
    rows = load_golden_dataset(dataset_path)
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for row in rows:
        result = pipeline_fn(row["question"])
        eval_data["question"].append(row["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c.page_content for c in result["chunks"]])
        eval_data["ground_truth"].append(row["ground_truth"])

    dataset = Dataset.from_dict(eval_data)
    return evaluate(dataset, metrics=[
        faithfulness, answer_relevancy, context_recall, context_precision
    ])

def assert_quality_gates(scores: dict):
    faith    = scores["faithfulness"]
    relevancy = scores["answer_relevancy"]
    print(f"Faithfulness:     {faith:.3f} (gate: ≥{FAITHFULNESS_THRESHOLD})")
    print(f"Answer Relevancy: {relevancy:.3f} (gate: ≥{ANSWER_RELEVANCY_THRESHOLD})")
    if faith < FAITHFULNESS_THRESHOLD:
        raise ValueError(f"GATE FAILED: faithfulness={faith:.3f} < {FAITHFULNESS_THRESHOLD}")
    if relevancy < ANSWER_RELEVANCY_THRESHOLD:
        raise ValueError(f"GATE FAILED: answer_relevancy={relevancy:.3f} < {ANSWER_RELEVANCY_THRESHOLD}")
    print("All quality gates passed.")
```

> **Ragas LLM cost:** Configure Ragas to use Groq Llama 3.3 70B as the judge model via `LangchainLLMWrapper` — keeps evaluation cost at $0.

---

## Pipeline Wiring

### `src/pipeline.py`

```python
import asyncio
from src.ingestion.query_classifier import classify_query
from src.retrieval.bm25_retriever import BM25Index
from src.retrieval.dense_retriever import dense_search
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.retrieval.reranker import BGEReranker
from src.retrieval.hyde import generate_hypothetical_doc
from src.generation.answer_generator import generate_answer

reranker = BGEReranker()

def run_rag_pipeline(question: str, bm25_index, vectorstore) -> dict:
    routing = classify_query(question)
    strategy = routing["strategy"]
    search_query = question

    if routing["use_hyde"]:
        search_query = generate_hypothetical_doc(question)

    if strategy == "bm25":
        bm25_results  = bm25_index.search(question, k=20)
        dense_results = dense_search(vectorstore, question, k=5)
    else:
        bm25_results  = bm25_index.search(question, k=10)
        dense_results = dense_search(vectorstore, search_query, k=20)

    fused    = reciprocal_rank_fusion(bm25_results, dense_results, top_n=15)
    reranked = reranker.rerank(question, fused, top_k=5)

    result = generate_answer(question, reranked)
    result["chunks"] = reranked  # exposed for Ragas eval
    return result

# Production hardening: async parallel retrieval
async def hybrid_retrieve_async(question: str, bm25_index, vectorstore):
    bm25_task  = asyncio.to_thread(bm25_index.search, question, 20)
    dense_task = asyncio.to_thread(dense_search, vectorstore, question, 20)
    bm25_results, dense_results = await asyncio.gather(bm25_task, dense_task)
    return reciprocal_rank_fusion(bm25_results, dense_results)
```

> **Key architecture decision:** Separate the indexing pipeline and query pipeline into two independently deployable services from day one. The indexing pipeline runs on a schedule (or on S3 upload events); the query pipeline is your latency-sensitive API. Coupling them together is the most common architecture mistake requiring a painful refactor at scale.

---

## Observability

### `src/observability/metrics.py`

```python
import time, functools
import boto3  # swap for azure-monitor-opentelemetry on Azure

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

def track_latency(operation_name: str):
    """Decorator — pushes latency to CloudWatch / Azure Monitor."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            cloudwatch.put_metric_data(
                Namespace="RAG/Production",
                MetricData=[{
                    "MetricName": f"{operation_name}Latency",
                    "Value": latency_ms,
                    "Unit": "Milliseconds",
                    "Dimensions": [{"Name": "Environment", "Value": "prod"}]
                }]
            )
            return result
        return wrapper
    return decorator

@track_latency("VectorRetrieval")
def dense_search(vectorstore, query, k=20): ...

@track_latency("LLMGeneration")
def generate_answer(question, chunks): ...
```

LangSmith tracing (free tier, 5K traces/month):

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"]    = "vyasa-intelligence"
# Every retrieval, rerank, and generation step now visible in LangSmith UI
```

### Production SLA Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API P50 latency | < 1s | > 1.5s |
| API P95 latency | < 3s | > 4s |
| API P99 latency | < 5s | > 7s |
| LLM generation time | < 2s | > 3s |
| Vector retrieval time | < 200ms | > 400ms |
| Faithfulness (rolling 24h) | ≥ 0.85 | < 0.82 |
| Answer relevancy (rolling 24h) | ≥ 0.80 | < 0.77 |
| Guardrail intervention rate | < 2% | > 5% |

---

## CI/CD Gate

### `.github/workflows/enterprise_deploy.yml`

Four sequential gates — deploy only reaches prod if all pass:

```yaml
name: Vyasa Intelligence — Promote to Production

on:
  push:
    branches: [main]

jobs:
  # ── GATE 1: Unit + integration tests ──────────────────────
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/unit tests/integration -v --tb=short

  # ── GATE 2: Ragas quality gate on staging ─────────────────
  ragas-gate:
    needs: test
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      - run: pip install -r requirements.txt
      - name: Deploy to staging
        run: ./scripts/deploy_staging.sh
      - name: Run Ragas evaluation against staging endpoint
        env:
          STAGING_API_URL: ${{ secrets.STAGING_API_URL }}
          GROQ_API_KEY:    ${{ secrets.GROQ_API_KEY }}
        run: python scripts/ragas_gate.py --endpoint $STAGING_API_URL
        # Fails build if faithfulness < 0.85 or answer_relevancy < 0.80
      - name: Upload evaluation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: ragas-report
          path: evaluation_report.json

  # ── GATE 3: Security scan ──────────────────────────────────
  security-scan:
    needs: ragas-gate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Bandit SAST scan
        run: pip install bandit && bandit -r src/ -ll
      - name: Trivy container vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'your-ecr/rag-api:${{ github.sha }}'
          severity:  'HIGH,CRITICAL'
          exit-code: '1'

  # ── DEPLOY: Only if all gates pass ────────────────────────
  deploy-prod:
    needs: security-scan
    runs-on: ubuntu-latest
    environment:
      name: production
      url: ${{ steps.deploy.outputs.url }}
    steps:
      - name: Deploy to AWS Lambda / Azure Container Apps
        id: deploy
        run: ./scripts/deploy_prod.sh
      - name: Post-deploy smoke test
        run: python scripts/smoke_test.py --env prod
```

`tests/test_quality_gates.py`:

```python
import pytest, json
from src.evaluation.ragas_eval import run_ragas_evaluation, assert_quality_gates
from src.pipeline import run_rag_pipeline

def test_faithfulness_and_relevancy_gates():
    scores = run_ragas_evaluation(run_rag_pipeline, "data/processed/golden_dataset.jsonl")
    with open("evaluation_report.json", "w") as f:
        json.dump(dict(scores), f, indent=2)
    assert_quality_gates(scores)  # raises on failure → CI fails → no deploy
```

---

## Cloud Deployment

### Free — Gradio on Hugging Face Spaces

```python
# app.py
import gradio as gr
from src.pipeline import run_rag_pipeline

def chat(question, history):
    result = run_rag_pipeline(question, bm25_index, vectorstore)
    return result["answer"]

demo = gr.ChatInterface(fn=chat, title="Vyasa Intelligence")
demo.launch()
```

### AWS Lean Serverless

Key unlock: **S3 Vectors** (2025) cuts vector storage from ~$350/month (OpenSearch Serverless) to ~$35/month — an 87% reduction.

**Architecture:**

```
S3 (raw docs, KMS encrypted)
   │
   ▼
Lambda: Ingestion (Presidio PII → Bedrock Embed)
   ├──► S3 Vectors (dense index, KMS encrypted)
   └──► DynamoDB (BM25 + chunk metadata + RBAC tags)

API Gateway (WAF + JWT Auth via Cognito)
   │
   ▼
Lambda: Query Handler
   ├── Classify query (BM25 / dense)
   ├── RBAC filter on retrieval
   ├── RRF fusion + Bedrock Rerank
   ├── Prompt Assembler + Claude Haiku 3.5
   └── Bedrock Guardrails (output safety)

CloudWatch → Grafana (latency, cost, token metrics)
CloudTrail → S3 (full audit log, 7-year retention)
```

**Terraform IaC:**

```hcl
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

resource "aws_s3_bucket" "rag_docs" {
  bucket = "rag-docs-${var.project_name}"
}

resource "aws_s3_bucket" "vector_store" {
  bucket = "rag-vectors-${var.project_name}"
}

resource "aws_lambda_function" "rag_ingestion" {
  filename      = "ingestion.zip"
  function_name = "rag-ingestion"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.ingest"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 1024
  environment {
    variables = {
      VECTOR_BUCKET  = aws_s3_bucket.vector_store.bucket
      DOCS_BUCKET    = aws_s3_bucket.rag_docs.bucket
      BEDROCK_REGION = var.aws_region
    }
  }
}

resource "aws_lambda_function" "rag_query" {
  filename      = "query.zip"
  function_name = "rag-query"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.query"
  runtime       = "python3.11"
  timeout       = 60
  memory_size   = 512
}

resource "aws_apigatewayv2_api" "rag_api" {
  name          = "rag-api"
  protocol_type = "HTTP"
}
```

**Lambda query handler:**

```python
import boto3, json, os
from src.pipeline import run_rag_pipeline

bedrock = boto3.client("bedrock-runtime", region_name=os.environ["BEDROCK_REGION"])

def query(event, context):
    body     = json.loads(event["body"])
    question = body["question"]
    result   = run_rag_pipeline(
        question=question,
        llm_client=bedrock,
        model_id="anthropic.claude-haiku-3-5-v1:0",
        vector_bucket=os.environ["VECTOR_BUCKET"]
    )
    return {
        "statusCode": 200,
        "body": json.dumps({
            "answer":    result["answer"],
            "citations": result["cited_chunk_ids"]
        })
    }
```

> **Cost reality:** ~1,000 RAG queries/day with Claude Haiku 3.5 (~3K input + 500 output tokens/query) = **$8–$15/month** total including S3 Vectors, Lambda, and API Gateway.

### Azure Cloud-Native

Azure's advantage: **Azure AI Search** handles hybrid BM25 + dense + semantic reranking in one managed call — eliminating `rank_bm25`, `hybrid_retriever.py`, and the reranker code (~40% less pipeline code) at the cost of the $75/month Basic tier.

**Architecture:**

```
Azure Blob Storage (raw docs, CMK via Key Vault)
   │
   ▼
Azure Document Intelligence (PDF parsing, tables, OCR)
   │
   ▼
Azure AI Search (BM25 + dense + semantic rerank in ONE service)
   └── Microsoft Entra RBAC filters at index level

Azure Container Apps (FastAPI — scales to zero = $0 idle)
   ├── Azure AD JWT auth (MSAL middleware)
   ├── HyDE + RBAC-aware query → AI Search
   ├── Azure OpenAI gpt-4o-mini (generation)
   └── Azure Content Safety (output guardrails)

Azure Monitor → Log Analytics Workspace
Microsoft Defender for AI (threat detection)
Azure Key Vault (CMK for all encryption)
```

**FastAPI on Azure Container Apps:**

```python
from fastapi import FastAPI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
import os

app = FastAPI()
search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name="rag-index",
    credential=os.environ["AZURE_SEARCH_KEY"]
)
openai_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2024-10-01"
)

@app.post("/query")
async def query(question: str):
    vector_query = VectorizedQuery(
        vector=get_embedding(question),
        k_nearest_neighbors=20,
        fields="content_vector"
    )
    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="rag-semantic-config",
        top=5
    )
    chunks = [r for r in results]
    prompt, cited_ids = assemble_prompt(question, chunks)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1
    )
    return {"answer": response.choices[0].message.content, "citations": cited_ids}
```

**Azure Bicep IaC:**

```bicep
param location string = 'eastus'
param projectName string = 'vyasa-intelligence'

resource searchService 'Microsoft.Search/searchServices@2024-06-01-preview' = {
  name: '${projectName}-search'
  location: location
  sku: { name: 'basic' }
  properties: {
    replicaCount: 1
    partitionCount: 1
    semanticSearch: 'free'
  }
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${projectName}-api'
  location: location
  properties: {
    configuration: {
      ingress: { external: true, targetPort: 8000 }
    }
    template: {
      containers: [{
        name: 'rag-api'
        image: 'your-acr.azurecr.io/rag-api:latest'
        resources: { cpu: '0.5', memory: '1Gi' }
      }]
      scale: { minReplicas: 0, maxReplicas: 10 }
    }
  }
}
```

### Hosting Options

| Platform | Free Tier | Notes |
|----------|-----------|-------|
| HF Spaces | Always free (CPU) | Best for demos, limited RAM |
| Render | Free (spins down) | Good for REST API backend |
| Fly.io | 3 shared VMs free | Best free option for always-on API |
| AWS Lambda + S3 Vectors | ~$8–15/month | Production-ready, serverless |
| Azure Container Apps + AI Search | ~$0–75/month | Fully managed, scales to zero |
| Raspberry Pi 5 | $0 (already owned) | Best local prod for low traffic |

---

## Execution Roadmap

### Week 1 — Enterprise Foundation

Before writing application code, set up your governance scaffold:

```bash
# AWS: Bootstrap with Control Tower (enables SCPs, CloudTrail, GuardDuty)
aws organizations create-organization
# Create three accounts: dev / staging / prod — never share accounts

# Azure: Resource groups per environment
az group create --name rg-rag-dev     --location eastus
az group create --name rg-rag-staging --location eastus
az group create --name rg-rag-prod    --location eastus
az security pricing create --name "AI" --tier "Standard"
```

Enable **CloudTrail (AWS)** or **Azure Monitor Diagnostic Settings** on day one. Retroactively adding audit logging requires downtime. Every query, retrieval, and guardrail intervention must be logged from the first deploy.

**Goal:** Accounts provisioned, IAM/Entra roles defined, CI/CD pipelines connected.

### Week 2 — Ingestion Pipeline

- Implement `document_loader.py`, `parva_splitter.py`, `entity_extractor.py`, `query_classifier.py`
- Add `secure_loader.py` — PII redaction with Presidio before any document hits the vector store
- Stand up ChromaDB locally + BGE embeddings
- Run one-time embedding locally (25–40 min for full Mahabharata), then upload `chroma_db/` to S3

**Goal:** Full corpus ingested with parva/section/entity metadata, one working end-to-end query.

### Week 3 — Full Retrieval + Generation

- Implement BM25 index, RRF fusion, BGE reranker, HyDE
- Wire `src/pipeline.py` — the complete query flow in one callable function
- Implement RBAC retrieval filter

**Goal:** Hybrid retrieval with reranking producing cited, Parva-attributed answers end-to-end.

### Week 4 — Evaluation + CI/CD Gate

- Build `golden_dataset.jsonl` — start with 50 Q&A triplets (simple / complex / adversarial)
- Run `ragas_eval.py` locally, tune until `faithfulness ≥ 0.85`, `answer_relevancy ≥ 0.80`
- Set up the enterprise GitHub Actions gate: unit → ragas → security scan → deploy

**Goal:** CI pipeline that blocks merge if quality gates fail.

### Week 5 — Cloud Migration

- **AWS:** Terraform → S3 Vectors + Lambda + Bedrock Claude Haiku 3.5 + Bedrock Guardrails (~$8–15/month)
- **Azure:** Bicep → AI Search + Container Apps + Azure OpenAI + Content Safety

**Goal:** Live REST API endpoints on both clouds, same Ragas gate validating both.

### Week 6 — Production Hardening

```python
# 1. Retry with exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_llm(prompt: str) -> str: ...

# 2. Semantic caching — avoid re-querying for identical questions
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())  # swap for Redis in prod
```

Additional hardening:
- p50/p95/p99 latency tracking per query (target: P90 TTFT < 2s)
- Token consumption logging per query for cost forecasting
- Weekly reindexing cron — RAG systems degrade as corpora evolve
- Cache hit rate monitoring — frequent identical queries should never hit the LLM twice

---

## Enterprise Features Summary

| Feature | AWS Service | Azure Service |
|---------|-------------|---------------|
| PII redaction at ingestion | Presidio + Macie | Presidio + Document Intelligence |
| LLM output guardrails | Bedrock Guardrails | Azure Content Safety |
| RBAC on retrieval | IAM + S3 bucket policies | Microsoft Entra + AI Search filters |
| Encryption at rest | KMS (CMK) | Key Vault (CMK) |
| Audit trail | CloudTrail (7-year retention) | Azure Monitor + Log Analytics |
| Compliance certifications | SOC2, HIPAA, ISO 27001 | SOC2, HIPAA, ISO 27001, GDPR |
| Threat detection | GuardDuty + Macie | Defender for AI |
| Latency SLA monitoring | CloudWatch + X-Ray | Azure Monitor + App Insights |
| Disaster recovery | Multi-AZ Lambda + S3 replication | Geo-redundant Container Apps + AI Search |

---

## Scale Estimates (Mahabharata)

| Metric | Estimate |
|--------|----------|
| Total pages (English translation) | ~5,500 pages (12 volumes) |
| Total tokens (English prose) | ~4–5 million tokens |
| Chunks at 600 tokens, 100 overlap | **~8,000–10,000 chunks** |
| BGE embedding time (CPU, local) | ~25–40 minutes (one-time) |
| ChromaDB size on disk | ~150–200 MB |
| S3 Vectors monthly cost (AWS) | ~$3–5/month |
| Azure AI Search Free tier | Fits (10K doc limit) |
| Groq inference per query | $0 (14.4K req/day free) |
| AWS total at 1K queries/day | ~$8–15/month |
