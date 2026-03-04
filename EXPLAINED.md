# Vyasa Intelligence — The Complete Beginner's Guide

> **Who is this for?** Someone who is studying Generative AI and wants to understand not just *what* this project does, but *why* every single piece exists, *how* it works under the hood, and *what problem* it solves. No prior knowledge assumed.

---

## Table of Contents

1. [The Big Picture — What Problem Are We Solving?](#1-the-big-picture)
2. [What is RAG? The Core Idea](#2-what-is-rag)
3. [The Mahabharata Corpus — Our Knowledge Source](#3-the-corpus)
4. [Stage 1 — Ingestion: Preparing the Book](#4-stage-1-ingestion)
   - [Document Loading](#41-document-loading)
   - [Text Chunking (Splitting)](#42-text-chunking)
   - [Entity Extraction](#43-entity-extraction)
   - [Embeddings — Teaching a Computer to Understand Meaning](#44-embeddings)
   - [ChromaDB — The Vector Database](#45-chromadb)
   - [BM25 — The Keyword Index](#46-bm25)
5. [Stage 2 — Retrieval: Finding the Right Pages](#5-stage-2-retrieval)
   - [Query Classification](#51-query-classification)
   - [Dense Search](#52-dense-search)
   - [BM25 Search](#53-bm25-search)
   - [Hybrid Search and Score Fusion](#54-hybrid-search-and-fusion)
   - [Reranking](#55-reranking)
6. [Stage 3 — Generation: Writing the Answer](#6-stage-3-generation)
   - [LLM Factory — Switching Between AI Brains](#61-llm-factory)
   - [Prompt Assembler — Writing Instructions for the AI](#62-prompt-assembler)
   - [Answer Generator — The Final Step](#63-answer-generator)
7. [Stage 4 — Safety and Speed](#7-stage-4-safety-and-speed)
   - [Guardrails — The Safety Net](#71-guardrails)
   - [Caching — Don't Repeat Yourself](#72-caching)
8. [The Pipeline — All Stages Together](#8-the-pipeline)
9. [The API — Talking to the System](#9-the-api)
10. [Evaluation — How Do We Know the AI is Correct?](#10-evaluation)
11. [Deployment — Running in the Real World](#11-deployment)
    - [Docker](#111-docker)
    - [Kubernetes](#112-kubernetes)
    - [CI/CD Pipeline](#113-cicd-pipeline)
12. [Technology Glossary — Every Term Explained](#12-technology-glossary)
13. [How Everything Connects — The Full Data Flow](#13-full-data-flow)

---

## 1. The Big Picture

### What Problem Are We Solving?

Imagine you have a library with a **single book** — the Mahabharata — which is roughly **5 million words long** (about 10 times the length of the Bible). A student comes in and asks:

> *"What was Arjuna's moral dilemma before the Kurukshetra war?"*

A human librarian who has read the entire book could answer this. But what if you want a computer to answer it instantly, accurately, and with references to the exact chapters?

The naive approach is to dump the whole book into an AI like ChatGPT and ask the question. **That doesn't work** because:

- AI models have a **context window limit** — they can only "read" a few thousand words at a time
- Even if they could read it all, it would be **extremely slow and expensive**
- The AI might **hallucinate** (make up answers) if it can't find the relevant part

**Vyasa Intelligence solves this** by building a smart Q&A system that:
1. Pre-processes the book once and stores it in a searchable format
2. When a question comes in, finds **only the 5 most relevant passages** from the book
3. Feeds those 5 passages to an AI and asks it to answer **based only on those passages**
4. Returns the answer with citations like `[Bhishma Parva, Chapter 25]`

This architecture is called **RAG — Retrieval-Augmented Generation**.

---

## 2. What is RAG?

### The Library Analogy

Think of RAG like a very smart research assistant:

```
WITHOUT RAG:
You → "Tell me everything about dharma in the Mahabharata" → AI
The AI has to remember the entire book from its training.
It might mix up facts or make things up.

WITH RAG:
You → question → System finds the 5 most relevant pages from the book
     → Those 5 pages + your question → AI
     → AI writes an answer based ONLY on those 5 pages
     → Answer includes citations so you can verify
```

### Why RAG? The Three Core Problems It Solves

| Problem | Without RAG | With RAG |
|---------|-------------|----------|
| **Knowledge Cutoff** | AI only knows things from its training data (before 2024) | Your own book/database is always up to date |
| **Hallucination** | AI invents facts | AI is grounded in the actual text |
| **Citations** | No source references | Every answer cites specific chapters |

### The RAG Flow (Simplified)

```
[Your Question]
      ↓
[Search Engine finds relevant passages]
      ↓
[Passages + Question → Language Model]
      ↓
[Answer with citations]
```

In code terms, the entire flow lives in [src/pipeline.py](src/pipeline.py) inside the `RAGPipeline` class.

---

## 3. The Corpus

### What is a Corpus?

A **corpus** is just a fancy word for "a collection of text documents used as the source of knowledge." In our case, it's the Mahabharata translation by **K.M. Ganguli** — 18 books (called Parvas), ~200,000 verses, ~5 million English words.

### The 18 Parvas (Books)

The Mahabharata is divided into 18 Parvas. Each is like a volume of a series:

| # | Parva | What it covers |
|---|-------|----------------|
| 1 | Adi Parva | Origin stories, birth of the Pandavas and Kauravas |
| 2 | Sabha Parva | The dice game, Draupadi's humiliation |
| 3 | Vana Parva | 12-year forest exile |
| 6 | Bhishma Parva | First 10 days of Kurukshetra war, includes the Bhagavad Gita |
| 7 | Drona Parva | Days 11-15 of the war |
| 12 | Shanti Parva | Bhishma's teachings on dharma, governance, philosophy |
| ... | ... | ... |

The raw text files live in [data/raw/](data/raw/) — one `.txt` file per Parva.

---

## 4. Stage 1 — Ingestion: Preparing the Book

**The Big Idea:** Before you can search a book intelligently, you need to prepare it — load it, cut it into pieces, understand what's in each piece, and store those pieces in two different kinds of indexes.

Think of this like preparing an encyclopedia:
- **Loading** = getting the books off the shelf
- **Chunking** = cutting out individual encyclopedia entries
- **Embedding** = writing a "meaning summary" for each entry
- **Indexing** = filing everything in a way you can find later

### 4.1 Document Loading

**File:** [src/ingestion/document_loader.py](src/ingestion/document_loader.py)

**What it does:** Reads the raw `.txt` files from disk and creates structured Python objects from them.

**Why it's needed:** Raw text files are just strings. We need to attach metadata (which book is this from? how many words?) to make the text useful.

```python
# document_loader.py:15-31
class MahabharataDocument:
    """Represents a single document (Parva) of the Mahabharata."""

    def __init__(self, content: str, parva: str, source_file: Path, metadata: dict):
        self.content = content    # The actual text
        self.parva = parva        # "Bhishma Parva", "Adi Parva", etc.
        self.source_file = source_file
        self.metadata = metadata  # file size, line count, encoding, etc.
```

The loader tries multiple text **encodings** (UTF-8, Latin-1, CP1252) because old text files can be encoded in different ways. Think of encoding as the "alphabet" used to store the file — different computers have used different alphabets historically.

```python
# document_loader.py:83-86
encodings = ["utf-8", "latin-1", "cp1252"]
for encoding in encodings:
    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
```

It also maps confusing filenames to proper Parva names:

```python
# document_loader.py:125-144
parva_mappings = {
    "adi": "Adi Parva",
    "bhishma": "Bhishma Parva",
    "shanti": "Shanti Parva",
    # ...18 total
}
```

---

### 4.2 Text Chunking

**File:** [src/ingestion/parva_splitter.py](src/ingestion/parva_splitter.py)

**What is a "chunk"?**

Imagine you're studying for an exam and you have a 500-page textbook. You don't keep the entire textbook in your head at once. You cut it into **index cards** — one important concept per card. A "chunk" is exactly that: a small, self-contained piece of text.

**Why not just use the whole Parva?**

- A single Parva can be 500,000+ words
- An LLM can only process ~4,000 words at a time
- If you give it less text, it's faster, cheaper, and more accurate
- Smaller chunks = more precise search results

**What size are our chunks?**

```python
# parva_splitter.py:38-40
self.chunk_size = chunk_size        # 500 tokens (~375 words)
self.chunk_overlap = chunk_overlap  # 50 tokens overlap between adjacent chunks
```

**Why overlap?** Think of it like scanning pages with a sliding window. If you cut exactly at word 500 and 501, you might split a sentence in half. Overlap ensures context isn't lost at boundaries.

```
Chunk 1: Words 1-500
Chunk 2: Words 451-950  ← overlaps with chunk 1 by 50 words
Chunk 3: Words 901-1400 ← overlaps with chunk 2 by 50 words
```

**Mahabharata-aware splitting:** The splitter first looks for chapter markers like "CHAPTER 1" or "Adhyaya 5" before doing word-count splitting. This is smarter than random cutting because chapters are natural story units.

```python
# parva_splitter.py:67-70
self.adhyaya_pattern = re.compile(
    r"(?i)chapter\s+(\d+)|adhyaya\s+(\d+)|section\s+(\d+)",
    re.MULTILINE,
)
```

**Each chunk gets rich metadata:**

```python
# parva_splitter.py:288-301
return {
    "chunk_id": "bhishma_parva_ch25_chunk3",  # Unique ID
    "parva": "Bhishma Parva",
    "chapter_number": 25,
    "characters_mentioned": ["Arjuna", "Krishna"],  # Who appears in this chunk
    "places_mentioned": ["Kurukshetra"],            # Where it takes place
    "is_dialogue": True,                             # Is it a conversation?
    "chunk_type": "philosophy",                     # narrative/dialogue/philosophy/description
    "token_count": 487,
}
```

**Chunk types** are automatically classified:

```python
# parva_splitter.py:356-403
philosophy_keywords = ["dharma", "karma", "moksha", "yoga", "soul", "atman", ...]
battle_keywords = ["battle", "war", "arrow", "sword", "chariot", ...]

if is_dialogue:
    return "dialogue"
elif any(keyword in text_lower for keyword in philosophy_keywords):
    return "philosophy"
elif any(keyword in text_lower for keyword in battle_keywords):
    return "description"
else:
    return "narrative"
```

This metadata becomes searchable later — you can filter to only get "philosophy" chunks when asking about dharma.

---

### 4.3 Entity Extraction

**File:** [src/ingestion/entity_extractor.py](src/ingestion/entity_extractor.py)

**What it does:** Scans each chunk and tags which characters, places, weapons, and concepts appear in it.

**Why?** Metadata filters speed up search dramatically. If someone asks "What did Karna say?", the system can pre-filter to only chunks that mention Karna before doing the expensive vector search.

The system knows about:
- **Characters:** Arjuna, Krishna, Duryodhana, Bhishma, Karna, Draupadi, and 30+ more
- **Places:** Hastinapura, Kurukshetra, Dwarka, Indraprastha, and more
- **Weapons:** Gandiva (Arjuna's bow), Brahmastra, Pashupatastra, and more
- **Concepts:** Dharma, Karma, Moksha, Yoga, Bhagavad Gita

---

### 4.4 Embeddings — Teaching a Computer to Understand Meaning

This is the most important concept in all of modern AI. Understanding embeddings is the key to understanding why RAG works.

**The Problem with Keyword Search**

If you search for "What is duty?" using keyword matching, you'll miss passages that talk about "dharma" (which *means* duty in Sanskrit). The words are different but the meaning is the same.

**The Solution: Embeddings**

An **embedding** converts a sentence into a list of numbers — specifically 768 numbers — that capture its *meaning*, not just its words.

```
"What is duty?" → [0.23, -0.87, 0.12, 0.54, ... 768 numbers total]
"Tell me about dharma" → [0.21, -0.85, 0.14, 0.52, ... 768 numbers]
"How to bake a cake" → [0.91, 0.34, -0.67, 0.88, ... 768 numbers]
```

Notice the first two vectors (lists of numbers) are **similar** — they both hover around the same numerical neighborhood. The third (baking) is in a completely different neighborhood.

**Why 768 numbers?** Think of it as a 768-dimensional map. Every sentence is a point on this map. Sentences with similar meanings are close to each other on the map. This is why it's called a "vector space" — vectors are arrows pointing to locations in this mathematical space.

**How are embeddings computed?**

The model used here is **BAAI/bge-base-en-v1.5** — a pre-trained neural network (like a small version of ChatGPT) that has been specifically trained to encode meaning into vectors. "BGE" stands for "Beijing Academy of AI General Embeddings" — it's a state-of-the-art open-source model.

```python
# build_index.py:73
self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# During indexing:
# build_index.py:228-230
embeddings = self.embedding_model.encode(
    texts, batch_size=32, convert_to_numpy=True
)
# texts = ["In the beginning...", "Arjuna said to Krishna...", ...]
# embeddings = array of shape (num_chunks, 768)
```

---

### 4.5 ChromaDB — The Vector Database

**What is a database?** A way to store and quickly retrieve data.

**What is a *vector* database?** A special database designed to store and search by those 768-dimensional vectors (embeddings). Regular databases search by exact match or sorting. Vector databases search by *similarity* — "find me the vectors closest to this query vector."

**ChromaDB** is an open-source vector database that runs locally on your machine. It's like SQLite but for vectors.

```python
# build_index.py:207-238
# Create the "mahabharata" collection (like a table in a regular database)
self.collection = self.chroma_client.create_collection(
    name="mahabharata",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity for distance
)

# Store chunks with their embeddings
self.collection.add(
    documents=texts,      # The actual text of each chunk
    metadatas=metadatas,  # chunk_id, parva, characters, etc.
    ids=ids,              # Unique identifiers
    embeddings=embeddings.tolist()  # The 768-dim vectors
)
```

**HNSW** (Hierarchical Navigable Small World) is the algorithm ChromaDB uses internally to find nearest neighbors efficiently. It builds a graph structure so searching through millions of vectors takes milliseconds rather than seconds.

**Cosine Similarity** measures the *angle* between two vectors. Two vectors pointing in the same direction have cosine similarity of 1.0 (identical meaning). Two perpendicular vectors have 0.0 (unrelated). Two opposing vectors have -1.0 (opposite meaning). This is why semantic search works even when the words are different — the *direction* of the vectors is what matters.

```
cosine_similarity("What is duty?", "Tell me about dharma") = ~0.85  (very similar)
cosine_similarity("What is duty?", "How to bake a cake") = ~0.12    (unrelated)
```

---

### 4.6 BM25 — The Keyword Index

**File:** [src/ingestion/build_index.py](src/ingestion/build_index.py) (lines 243-271)

**What is BM25?**

BM25 stands for **Best Match 25** — it's the algorithm that powers traditional search engines like early Google, Apache Solr, and Elasticsearch. It's purely keyword-based: it counts how often your search words appear in each document and ranks by frequency.

**Why do we need BM25 if we have vector search?**

Vector search is great at meaning but can miss important specific words. For example:

- Question: "What weapon did Arjuna use?"
- BM25: Will find "Arjuna" and "weapon" and "Gandiva" by exact word match — very precise
- Dense search: Might return philosophical passages about Arjuna that don't mention weapons

They complement each other. BM25 is precise for specific names, facts, weapons. Dense search is better for abstract concepts and meaning.

**How BM25 works (simplified):**

```python
# build_index.py:254-256
tokenized_docs = []
for chunk in chunks:
    tokens = chunk.page_content.lower().split()  # ["in", "the", "beginning", ...]
    tokenized_docs.append(tokens)

bm25_index = BM25Okapi(tokenized_docs)
```

Then at search time:
```python
# hybrid_search.py:135-150
tokens = query.lower().split()  # ["who", "is", "arjuna"]
doc_scores = self.bm25_index.get_scores(tokens)
# Returns a score for each chunk: [0.0, 2.3, 0.0, 5.7, 1.2, ...]
# Higher score = better keyword match
```

The BM25 index is saved as a **pickle file** (`.pkl`) — Python's binary serialization format. Think of it like a ZIP file for Python objects.

---

## 5. Stage 2 — Retrieval: Finding the Right Pages

Once the book is indexed (Stage 1), we're ready to search it. When a user asks a question, retrieval runs to find the most relevant chunks.

### 5.1 Query Classification

**File:** [src/retrieval/query_classifier.py](src/retrieval/query_classifier.py)

**What it does:** Before searching, the system tries to understand *what kind* of question is being asked. Different question types need different search strategies.

**The 6 Query Types:**

```python
# query_classifier.py:19-27
class QueryType(Enum):
    ENTITY = "entity"           # "Who is Karna?" — about a specific person/place/thing
    PHILOSOPHICAL = "philosophical"  # "What is dharma?" — abstract concepts
    NARRATIVE = "narrative"     # "What happened at the dice game?" — story events
    CONCEPTUAL = "conceptual"   # "Relationship between dharma and karma" — abstract ideas
    TEMPORAL = "temporal"       # "When did the 12-year exile end?" — time/sequence
    COMPARATIVE = "comparative" # "Compare Arjuna and Karna" — contrasting things
    UNKNOWN = "unknown"         # Can't figure it out
```

**How classification works — 3 methods combined:**

**Method 1: Pattern Matching (Regex)**
```python
# query_classifier.py:138-163
patterns = {
    QueryType.ENTITY: [
        r"who (?:is|was) (\w+)",           # "Who is Krishna?"
        r"tell me about (\w+)",             # "Tell me about Bhishma"
        r"(\w+)'s (?:role|character|story)" # "Karna's story"
    ],
    QueryType.PHILOSOPHICAL: [
        r"what (?:is|was) (?:the )?(?:concept of )?(dharma|karma|moksha)",
        r"explain (?:the )?(?:concept of )?(dharma|karma)",
        r"philosophy (?:of|in|about)",
    ],
    # ...
}
```

**Method 2: Keyword Matching**
Checks if known keywords appear in the question:
- Entity queries contain character names (Arjuna, Krishna, Karna)
- Philosophical queries contain concepts (dharma, karma, moksha, yoga)
- Narrative queries contain story words (battle, war, happened, story)

**Method 3: Semantic Similarity**
Uses embeddings to compare the question against pre-computed example questions for each type:
```python
# query_classifier.py:189-231
type_examples = {
    QueryType.ENTITY: [
        "Who is Arjuna?", "Tell me about Krishna", "Describe Bhishma"
    ],
    QueryType.PHILOSOPHICAL: [
        "What is dharma?", "Explain the concept of karma"
    ],
    # ...
}
# Compute cosine similarity between incoming question and these examples
```

**Final score = 40% patterns + 30% keywords + 30% semantic similarity**

```python
# query_classifier.py:262-270
confidence_scores[query_type.value] = (
    0.4 * pattern_scores.get(query_type, 0.0)
    + 0.3 * keyword_scores.get(query_type, 0.0)
    + 0.3 * semantic_scores.get(query_type, 0.0)
)
```

**Why classify first?** Different query types work better with different search weights. The system adapts its search strategy:

```python
# query_classifier.py:400-435
if query_type == QueryType.ENTITY:
    # "Who is Karna?" — exact name matching is key → more BM25
    strategy["bm25_weight"] = 0.7
    strategy["dense_weight"] = 0.3

elif query_type == QueryType.PHILOSOPHICAL:
    # "What is moksha?" — meaning is key → more dense search + HyDE
    strategy["bm25_weight"] = 0.3
    strategy["dense_weight"] = 0.7
    strategy["hyde"] = True        # Generate a hypothetical document first

elif query_type == QueryType.CONCEPTUAL:
    # "Relationship between dharma and fate" — mostly semantic
    strategy["bm25_weight"] = 0.2
    strategy["dense_weight"] = 0.8
    strategy["top_k"] = 20         # Retrieve more candidates
```

---

### 5.2 Dense Search

**File:** [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py) (lines 153-183)

Dense search is **semantic/meaning-based search** using the same embedding model from ingestion.

**How it works:**
1. Convert the user's question into a 768-dimensional vector
2. Ask ChromaDB: "Find me the chunks whose vectors are closest to this query vector"
3. ChromaDB uses the HNSW algorithm to search efficiently

```python
# hybrid_search.py:163-183
# Step 1: Convert question to vector
query_embedding = self.embedding_model.encode(
    query, convert_to_numpy=True, normalize_embeddings=True
)
# query_embedding is now a 768-number array like [0.23, -0.87, ...]

# Step 2: Search ChromaDB for the most similar chunks
results = self.chroma_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=top_k,
    include=["metadatas", "distances", "documents"],
)

# Step 3: Convert distance to similarity (ChromaDB returns distance, not similarity)
distance = results["distances"][0][i]
similarity = 1.0 - distance  # distance=0 means identical → similarity=1
```

---

### 5.3 BM25 Search

**File:** [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py) (lines 125-150)

BM25 search is the **keyword-based** search using the pre-built index.

```python
# hybrid_search.py:135-150
# Step 1: Tokenize the query (split into words)
tokens = query.lower().split()
# "Who is Arjuna?" → ["who", "is", "arjuna"]

# Step 2: Get BM25 scores for all chunks
doc_scores = self.bm25_index.get_scores(tokens)
# Returns array: [0.0, 5.7, 2.3, 0.0, 8.1, ...] — one score per chunk

# Step 3: Sort and take top results
top_indices = np.argsort(doc_scores)[::-1][:top_k]
```

---

### 5.4 Hybrid Search and Fusion

**File:** [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py) (lines 81-123, 186-250)

**The Core Idea:** Run both BM25 and dense search independently, then combine their rankings into one final ranking.

**Why combine?**

| Search Type | Good At | Bad At |
|-------------|---------|--------|
| BM25 | Exact names ("Gandiva", "Kurukshetra") | Semantic similarity ("duty" ≠ "dharma") |
| Dense | Meaning ("duty" = "dharma") | Exact rare words, specific names |
| **Hybrid** | **Both!** | — |

**The combining algorithm — Score Normalization + Weighted Sum:**

Step 1: Normalize scores so they're all between 0 and 1 (Min-Max normalization):
```python
# hybrid_search.py:252-276
# Min-Max normalization
normalized = {}
for chunk_id, score in scores.items():
    normalized[chunk_id] = (score - min_score) / (max_score - min_score)
# Now all scores are between 0.0 and 1.0
```

Step 2: Combine with weights:
```python
# hybrid_search.py:221-224
combined_score = bm25_weight * bm25_score + dense_weight * dense_score
# Default: 0.5 * bm25 + 0.5 * dense (equal weight)
# Entity query: 0.7 * bm25 + 0.3 * dense (favor keyword match)
# Philosophy query: 0.3 * bm25 + 0.7 * dense (favor semantic)
```

Step 3: Sort by combined score, take top 5 (or top_k):
```python
# hybrid_search.py:227-229
sorted_results = sorted(
    combined_scores.items(), key=lambda x: x[1], reverse=True
)[:top_k]
```

The file [src/retrieval/rank_fusion.py](src/retrieval/rank_fusion.py) also implements **Reciprocal Rank Fusion (RRF)** — an alternative fusion strategy. Instead of combining raw scores, RRF combines *rankings*: each chunk gets a score of `1/(rank + 60)`. This is more robust because it doesn't depend on the absolute values of scores.

---

### 5.5 Reranking

**File:** [src/retrieval/reranker.py](src/retrieval/reranker.py)

**What is reranking?**

Think of it as a two-stage hiring process:
- Stage 1 (hybrid search): Quickly screen 1000 candidates, pick 20 semifinalists
- Stage 2 (reranker): Carefully evaluate those 20 and pick the best 5

The **reranker** is a **Cross-Encoder** model. Unlike the embedding model (which encodes query and document separately), a cross-encoder sees the query and document *together* and scores their relevance jointly. This is more accurate but slower — which is why it's only used to rerank a small shortlist.

The model used is **BAAI/bge-reranker-v2-m3** (or **Cohere Rerank** as a cloud alternative).

```
Hybrid Search (fast, approximate):
  Query: "What did Arjuna say about duty?"
  Retrieves: 20 candidate chunks

Cross-Encoder Reranker (slow, precise):
  Scores each of the 20 chunks against the full query
  Returns: Top 5, now with more accurate relevance scores
```

---

## 6. Stage 3 — Generation: Writing the Answer

Now we have the 5 most relevant passages from the Mahabharata. Time to write the answer.

### 6.1 LLM Factory — Switching Between AI Brains

**File:** [src/generation/llm_factory.py](src/generation/llm_factory.py)

**What is an LLM?**

LLM = **Large Language Model**. This is the AI that actually writes the answer — models like GPT-4, Llama, Mistral. They're neural networks trained on billions of text examples to predict the next word. By doing this billions of times, they learn grammar, facts, reasoning, and how to write coherently.

**Why two LLM providers?**

| Provider | When Used | Model | Cost |
|----------|-----------|-------|------|
| **Ollama** | Local development | llama3.2 (2B) | Free — runs on your laptop |
| **Groq** | Production | llama-3.3-70b-versatile | Free tier (14,400 req/day) |

**Ollama** is software that runs open-source AI models locally on your computer. No internet needed, no API costs, complete privacy. Great for development and testing.

**Groq** is a cloud service with a free tier that provides access to large, powerful models like Llama 3.3 70B (70 billion parameters — much smarter than local 2B models). In production, we use Groq for higher quality answers.

The **Factory Pattern** is a software design principle: you have one function that creates objects (here: LLM instances) and hides which specific type it's creating. The rest of the code doesn't care whether it's Ollama or Groq — it just calls `llm.invoke(prompt)`.

```python
# llm_factory.py:18-51
def get_llm(provider=None, model=None, temperature=0.1, max_tokens=None):
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama")  # Read from .env file

    if provider == "ollama":
        return OllamaLLM(model="llama3.2", base_url="http://localhost:11434")
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")  # Secret key from .env file
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
```

**Temperature** (0.0 to 1.0) controls randomness. At 0.0, the AI always picks the most probable next word (deterministic, consistent). At 1.0, it's very creative/random. We use `temperature=0.1` — almost deterministic, because we want factual answers, not creative fiction.

---

### 6.2 Prompt Assembler — Writing Instructions for the AI

**File:** [src/generation/prompt_assembler.py](src/generation/prompt_assembler.py)

**What is a Prompt?**

A prompt is the text you send to an LLM. The quality of the answer depends enormously on how you write the prompt. This is called **Prompt Engineering**.

A naive prompt: `"Answer this: What is dharma?"`

Our prompt is much more sophisticated:

```python
# prompt_assembler.py:77-93
system_prompt = """You are Vyasa, an AI assistant specialized in the Mahabharata.
Your role is to answer questions based solely on the provided context from the Mahabharata text.

CRITICAL REQUIREMENTS:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain the answer, say "I cannot find this information..."
3. ALWAYS include citations using the format [Parva, Section]
4. Do not invent or extrapolate information beyond the text
5. Maintain accuracy and fidelity to the source text"""
```

**Role-based prompts:** Different users get different instruction styles:
- `public` → "Provide clear, accessible answers suitable for general readers"
- `scholar` → "Provide detailed answers with specific references, suitable for academic research"
- `admin` → "Provide comprehensive answers with full context and analysis"

**How the full prompt is assembled:**

```python
# prompt_assembler.py:58-66
prompt_parts = [
    system_prompt,           # Who you are + rules
    history_section,         # Previous Q&As (for follow-up questions)
    formatted_context,       # The 5 retrieved passages with citation labels
    f"Question: {question}", # The actual user question
    "Answer:",               # Signal the AI to start writing
]
```

**Context formatting:** Each retrieved chunk is labeled with its source:
```
[Bhishma Parva, 25]
Then Arjuna, overwhelmed with sorrow, his eyes full of tears, spoke to Krishna...

[Bhishma Parva, 26]
Krishna said: "Why have you brought this disgrace upon yourself, Arjuna?..."

Question: What was Arjuna's dilemma before the battle?
Answer:
```

**Citation validation:** After the answer is generated, the system checks that any citations in the answer actually came from the provided context — not from the AI's training data (which would be hallucination).

```python
# prompt_assembler.py:182-211
def validate_answer_citations(self, answer, provided_context):
    answer_citations = set(self.extract_citations_from_answer(answer))
    context_citations = set()  # Build from provided_context

    missing_citations = answer_citations - context_citations  # Hallucinated citations!
    valid_citations = answer_citations & context_citations    # Good citations

    return {
        "valid_citations": list(valid_citations),
        "missing_citations": list(missing_citations),  # Red flag: AI invented a source
        "all_valid": len(missing_citations) == 0,
    }
```

---

### 6.3 Answer Generator — The Final Step

**File:** [src/generation/answer_generator.py](src/generation/answer_generator.py)

This is the component that actually calls the LLM and assembles the final answer object.

```python
# answer_generator.py:112-166
def generate_answer(self, question, context_docs, user_role="public"):
    # Step 1: Build the prompt
    prompt = self.prompt_assembler.assemble_prompt(
        question=question,
        context_docs=context_docs,
        user_role=user_role,
    )

    # Step 2: Call the LLM (Groq or Ollama)
    answer = self.llm.invoke(prompt)

    # Step 3: Extract the text (different LLMs return different response formats)
    if hasattr(answer, "content"):
        answer_text = answer.content  # ChatGroq returns an object with .content
    else:
        answer_text = str(answer)     # OllamaLLM returns a string directly

    # Step 4: Extract and validate citations
    citations = self.prompt_assembler.extract_citations_from_answer(answer_text)
    validation_result = self.prompt_assembler.validate_answer_citations(answer_text, context_docs)

    # Step 5: Build sources list (human-readable)
    sources = self._build_sources_list(context_docs, citations)

    return {
        "answer": answer_text.strip(),
        "citations": citations,   # ["[Bhishma Parva, 25]", "[Bhishma Parva, 26]"]
        "sources": sources,       # Human-readable source strings
        "metadata": {...}         # timing, token counts, etc.
    }
```

---

## 7. Stage 4 — Safety and Speed

### 7.1 Guardrails — The Safety Net

**File:** [src/generation/guardrails.py](src/generation/guardrails.py)

**What are guardrails?**

Guardrails are rules that prevent the system from being misused or generating harmful content. Think of them as bouncers at a club — they check both who comes in (input) and what goes out (output).

**Why do we need them?**

Without guardrails, someone could ask:
- "How do I make a weapon like Brahmastra?" (using the system for harm)
- "Convert me to Hinduism" (religious conversion — not our job)
- Politically charged questions unrelated to the Mahabharata

The system checks both input (question) and output (answer) against a list of blocked patterns:

```python
# guardrails.py:30-42
blocked_patterns = [
    # Hate speech
    r"\b(hate|kill|destroy).*\b(people|group|community|caste|religion)",
    # Violence instructions
    r"\b(how to|instructions for).*(bomb|weapon|violence|attack)",
    # Political content
    r"\b(vote|election|campaign|political party)",
    # Religious conversion
    r"\b(convert).*\b(to christianity|to islam|to hinduism|religion)",
]
```

**Two-stage checking:**

```python
# guardrails.py:60-105 (input check)
def check_input(self, question, user_role="public"):
    # Returns: {"allowed": True/False, "blocked_categories": [...]}

# guardrails.py:107-155 (output check)
def check_output(self, answer, context_used=True):
    # Also checks for: missing citations, inappropriate length, refusal patterns
```

**Role-based restrictions:** `public` users get more restrictions. `admin` users get more access. This mirrors real-world access control (like different login levels in an app).

**Mahabharata context check:** The system even verifies that questions are related to the Mahabharata, not off-topic:

```python
# guardrails.py:45-50
mahabharata_context_patterns = [
    r"\b(kuru|pandava|kaurava|hastinapura|indraprastha)",
    r"\b(dharma|adharma|karma|yoga|veda)",
    r"\b(krishna|arjuna|bhishma|drona|karna|duryodhana)",
]
```

---

### 7.2 Caching — Don't Repeat Yourself

**File:** [src/generation/cache.py](src/generation/cache.py)

**The Problem:** If 1000 users ask "What is dharma?", should we:
- Call the LLM 1000 times and wait 1-2 seconds each time? (slow + expensive)
- Answer once, store the answer, serve from storage for the other 999? (fast + cheap)

The answer is obviously the second option. That's **caching**.

**How the cache key is generated:**

The system generates a unique key based on:
1. The question (normalized to lowercase)
2. A hash of the retrieved documents (same question might get different documents if the index changes)
3. The user role (scholars get different answers than public)

```python
# cache.py:115-135
def _generate_cache_key(self, question, context_hash, user_role):
    normalized_question = question.lower().strip()
    combined = f"{normalized_question}|{context_hash}|{user_role}"
    return hashlib.sha256(combined.encode()).hexdigest()
    # Returns something like: "a3f8b9c2d1e4f5a6b7c8d9e0f1a2b3c4..."
```

**SHA-256 hash** is a mathematical function that converts any string into a fixed-length 64-character string. The same input always produces the same output, but even a tiny change produces a completely different output. It's like a fingerprint — unique to the content.

**Two cache types:**

**In-memory cache** — stores answers in RAM:
```python
# cache.py:16-77
class ResponseCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self._cache = {}  # Dictionary: {cache_key: response_data}
        # Stores up to 1000 answers for 1 hour
```

**Redis cache** — for production with multiple servers:
```python
# cache.py:266-344
class RedisCache(ResponseCache):
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
```

**Redis** is a separate in-memory database server. When you have multiple servers running (horizontal scaling), each server needs to share the same cache. In-memory dict won't work (each server has its own memory). Redis is a shared cache that all servers can access.

**LRU Eviction (Least Recently Used):** When the cache is full, old answers are removed:
```python
# cache.py:137-148
def _evict_lru(self):
    lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
    del self._cache[lru_key]
    # Remove the answer that was accessed longest ago
```

**TTL (Time-to-Live):** Answers expire after 1 hour (configurable). This ensures that if the underlying documents change, stale answers get cleared automatically.

---

## 8. The Pipeline — All Stages Together

**File:** [src/pipeline.py](src/pipeline.py)

The `RAGPipeline` class is the conductor. It connects all the components and defines the exact order of operations.

**Initialization:** When the pipeline starts up, it loads all components:

```python
# pipeline.py:20-84
class RAGPipeline:
    def __init__(self, chroma_dir, bm25_path, llm_provider, ...):
        self.retriever = HybridSearcher(...)     # BM25 + ChromaDB
        self.generator = AnswerGenerator(...)    # LLM + Prompt assembler
        self.cache = ResponseCache(...)          # In-memory or Redis
        self.guardrails = ContentGuardrails()    # Safety checks
```

**The `query()` method — the complete flow:**

```python
# pipeline.py:86-229
def query(self, question, user_role="public", top_k=5):
    # 1. Check if input is safe
    input_check = self.guardrails.check_input(question, user_role)
    if not input_check["allowed"]:
        return {"answer": "Cannot process this question", ...}

    # 2. Do hybrid search to find relevant chunks
    context_docs = self._retrieve_context(question, top_k, ...)

    # 3. Check cache: maybe this question was already answered?
    context_hash = self.cache.generate_context_hash(context_docs)
    cached = self.cache.get(question, context_hash, user_role)
    if cached:
        return cached  # Fast path: return stored answer!

    # 4. Generate new answer using LLM
    generation_result = self.generator.generate_answer(
        question=question,
        context_docs=context_docs,
        user_role=user_role,
    )

    # 5. Check if output is safe
    output_check = self.guardrails.check_output(generation_result["answer"])

    # 6. Save to cache for next time
    self.cache.set(question, context_hash, generation_result, user_role)

    # 7. Return the response
    return {
        "answer": generation_result["answer"],
        "sources": generation_result["sources"],
        "retrieval_time": ...,
        "generation_time": ...,
        "cache_hit": False,
    }
```

**Query type routing within retrieval:**

```python
# pipeline.py:249-280
def _retrieve_context(self, question, top_k, bm25_weight, dense_weight):
    query_type = self._classify_query(question)

    if query_type == "entity":
        # "Who is Bhishma?" → favor keyword search
        return self.retriever.search(query, bm25_weight=0.7, dense_weight=0.3)
    elif query_type == "philosophy":
        # "What is yoga?" → favor semantic search
        return self.retriever.search(query, bm25_weight=0.3, dense_weight=0.7)
    else:
        # Balanced
        return self.retriever.search(query, bm25_weight=0.5, dense_weight=0.5)
```

---

## 9. The API — Talking to the System

**File:** [src/api/main.py](src/api/main.py)

**What is an API?**

API = **Application Programming Interface**. It's a way for programs (or users) to talk to another program over a network. Think of it like a restaurant menu — it defines what you can order (what requests you can make) and what you'll get (what responses to expect).

**FastAPI** is a modern Python framework for building APIs. It automatically generates documentation and validates requests/responses.

**The two endpoints:**

```python
# api/main.py:55-76

# 1. Health check — is the server running?
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Vyasa Intelligence API is running"}

# 2. Query endpoint — ask a question
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = pipeline.query(
        question=request.question,
        user_role=request.user_role,
        top_k=request.top_k,
    )
    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        retrieval_time_ms=result.get("retrieval_time", 0) * 1000,
        generation_time_ms=result.get("generation_time", 0) * 1000,
    )
```

**Pydantic models** define the shape of requests and responses — like a contract:

```python
# api/main.py:42-52
class QueryRequest(BaseModel):
    question: str               # Required: the question
    user_role: Optional[str] = "public"  # Optional: public/scholar/admin
    top_k: Optional[int] = 5            # Optional: how many chunks to retrieve

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval_time_ms: Optional[float]
    generation_time_ms: Optional[float]
```

**How to call the API:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Arjuna'\''s dilemma?",
    "user_role": "public",
    "top_k": 5
  }'
```

**Automatic docs:** FastAPI automatically creates interactive documentation at `http://localhost:8000/docs` — you can test the API directly from your browser.

**Gradio UI:** [app.py](app.py) creates a browser-based chat interface using **Gradio** — a library for quickly building ML demos. Think of it as a simple ChatGPT-like interface for your system.

---

## 10. Evaluation — How Do We Know the AI is Correct?

**Directory:** [src/evaluation/](src/evaluation/)

**The Problem:** LLMs can generate very confident, fluent, wrong answers. How do you systematically measure quality?

**Ragas** (RAG Assessment) is an open-source evaluation framework designed specifically for RAG systems. It uses AI to evaluate AI.

**The Two Core Metrics:**

**1. Faithfulness (target: ≥ 0.85)**

*"Is the answer actually grounded in the retrieved context, or did the AI make something up?"*

Ragas breaks the answer into individual claims and checks each one against the retrieved passages:
```
Answer: "Arjuna was Kunti's third son and was known for his archery skills."
Claim 1: "Arjuna was Kunti's third son" → Found in context? ✓
Claim 2: "Arjuna was known for archery" → Found in context? ✓
Faithfulness = 2/2 = 1.0 (perfect)

If one claim was invented:
Claim 3: "Arjuna was 25 years old at Kurukshetra" → Found in context? ✗
Faithfulness = 2/3 = 0.67 (below threshold → fail)
```

**2. Answer Relevancy (target: ≥ 0.80)**

*"Does the answer actually address the question that was asked?"*

Ragas generates multiple versions of questions that the given answer would satisfy, then measures cosine similarity between those generated questions and the original question.

**Golden Dataset:** A handcrafted set of question-answer pairs that represent what perfect answers look like. The system's outputs are compared against these gold standards.

**Quality Gates:** These metrics are enforced automatically:
```yaml
# In CI/CD pipeline
- faithfulness >= 0.85    # Fail if AI makes things up too often
- answer_relevancy >= 0.80  # Fail if AI goes off-topic
- overall_score >= 0.80     # Combined gate
```

If these thresholds aren't met, the deployment is **blocked** — the new code cannot go to production.

---

## 11. Deployment — Running in the Real World

### 11.1 Docker

**File:** [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)

**What is Docker?**

Imagine you build an app on your laptop (Mac, Python 3.11, specific libraries). It works perfectly. You send it to a friend who runs Windows with Python 3.9 and different libraries. It breaks.

Docker solves this by packaging your app + all its dependencies + the operating system into a single **container** — a self-contained, portable box that runs identically anywhere.

A **Dockerfile** is the recipe for building the container:
```dockerfile
# Dockerfile (simplified)
FROM python:3.11-slim          # Start from a minimal Python image
WORKDIR /app                   # Working directory inside container
COPY pyproject.toml .          # Copy dependency definitions
RUN pip install -e .           # Install all dependencies
COPY src/ ./src/               # Copy application code
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose** runs multiple containers together:
```yaml
# docker-compose.yml (simplified)
services:
  api:      # The FastAPI server
    build: .
    ports: ["8000:8000"]
  gradio:   # The chat UI
    build: { dockerfile: Dockerfile.gradio }
    ports: ["7860:7860"]
  redis:    # The cache server
    image: redis:alpine
    ports: ["6379:6379"]
```

With one command — `docker compose up` — all three services start together.

### 11.2 Kubernetes

**Directory:** [k8s/](k8s/)

**What is Kubernetes?**

Kubernetes (K8s) is a system for managing containers in production. Think of Docker as running a single restaurant kitchen, and Kubernetes as managing a chain of restaurants:

- **Scaling:** Start more containers when demand spikes ("more chefs during lunch rush")
- **Self-healing:** Restart crashed containers automatically ("if a chef faints, replace immediately")
- **Load balancing:** Distribute traffic across multiple containers ("send customers to the least busy restaurant")
- **Rolling updates:** Update without downtime ("renovate one restaurant at a time while others stay open")

The project deploys to **Rancher Desktop** (a local Kubernetes environment) and is accessible at `http://vyasa.local`.

**Auto-scaling (HPA — Horizontal Pod Autoscaler):**
```yaml
# k8s/ (conceptual)
minReplicas: 2   # Always have at least 2 running copies
maxReplicas: 5   # Scale up to 5 if load increases
# Scale up when CPU usage exceeds 70%
```

### 11.3 CI/CD Pipeline

**Directory:** [.github/workflows/](​.github/workflows/)

**What is CI/CD?**

CI/CD = **Continuous Integration / Continuous Deployment**. It's an automated system that runs every time you push code to GitHub, checking that nothing is broken before allowing the code to go live.

Think of it as a mandatory quality checklist that runs automatically — you can't skip it.

**The 4 Gates:**

```
Push Code to GitHub
        ↓
Gate 1: Code Quality (formatting, linting, security)
  - Black: Is the code formatted consistently?
  - isort: Are imports sorted correctly?
  - Ruff: Are there any code style violations?
  - MyPy: Are all type hints correct?
  - Bandit: Are there any security vulnerabilities?
        ↓ (only if Gate 1 passes)
Gate 2: Unit Tests
  - pytest: Do all individual functions work correctly?
  - Code coverage ≥ 80%: Is 80%+ of the code tested?
        ↓ (only if Gate 2 passes)
Gate 3: Integration Tests
  - Do the API endpoints work end-to-end?
  - Can the system connect to Redis?
  - Are health checks passing?
        ↓ (only if Gate 3 passes)
Gate 4: Quality Gates (Ragas Evaluation)
  - Faithfulness ≥ 0.85?
  - Answer Relevancy ≥ 0.80?
        ↓ (only if Gate 4 passes)
Deploy to Production 🚀
```

If any gate fails, the deployment stops and the developer is notified. This prevents broken code from reaching users.

**Why automated testing matters:**

Without tests, you'd have to manually test every feature after every change — impractical as the codebase grows. With automated tests, you can change one part of the code and instantly know if something else broke.

---

## 12. Technology Glossary — Every Term Explained

| Term | What it is | Analogy |
|------|------------|---------|
| **RAG** | Retrieval-Augmented Generation — find relevant text, give to LLM | Research assistant who finds relevant pages before answering |
| **LLM** | Large Language Model — AI that generates text | A very well-read writer who can answer any question |
| **Embedding** | A list of ~768 numbers representing the meaning of text | GPS coordinates for meaning — similar texts are close on the map |
| **Vector Database** | A database that stores and searches by embeddings | A library organized by meaning, not by title |
| **ChromaDB** | An open-source vector database that runs locally | Your personal vector library |
| **BM25** | Keyword-based ranking algorithm (Best Match 25) | Old-school search engine — counts word frequency |
| **Hybrid Search** | Combining BM25 + dense search | Using both a dictionary AND a thesaurus to search |
| **Cosine Similarity** | Measures the angle between two vectors | How parallel two arrows are pointing |
| **Chunk** | A small piece of text (300-500 words) cut from a larger document | An index card cut from a book |
| **Tokenization** | Splitting text into words or sub-words | Cutting a sentence into individual lego pieces |
| **Temperature** | Controls LLM randomness (0.0=deterministic, 1.0=creative) | The "creativity dial" on an AI |
| **Prompt Engineering** | Crafting instructions to get better AI outputs | Writing a very precise recipe vs a vague one |
| **Guardrails** | Rules preventing harmful inputs/outputs | A bouncer who checks IDs |
| **Caching** | Storing computed results so you don't recompute | Keeping leftovers in the fridge |
| **Redis** | A fast in-memory key-value store (used for shared caching) | A whiteboard shared by multiple people |
| **FastAPI** | A Python framework for building HTTP APIs | A waiter who takes orders (HTTP requests) and brings food (responses) |
| **Docker** | Packages app + dependencies into a portable container | Shipping container for software |
| **Kubernetes** | Manages multiple Docker containers in production | A factory manager for Docker containers |
| **CI/CD** | Automated testing and deployment pipeline | Assembly line quality control |
| **Ragas** | RAG evaluation framework that uses AI to score AI | A teacher grading an AI's homework |
| **Faithfulness** | Does the answer only use facts from the context? | "Did you actually read the book before answering?" |
| **Answer Relevancy** | Does the answer address the actual question? | "Did you actually answer what was asked?" |
| **Cross-Encoder** | A model that jointly scores query + document relevance | A careful editor who re-reads both the question and each candidate answer |
| **HyDE** | Hypothetical Document Embedding — generate a fake answer to search with | If you don't know the answer, imagine what it would look like and search for that |
| **Parva** | A "book" or major section of the Mahabharata (there are 18) | Volume in an encyclopedia series |
| **Adhyaya** | A chapter within a Parva | Chapter within a volume |
| **Pydantic** | Python library for data validation using type hints | A strict form that rejects wrong input types |
| **Pickle** | Python's built-in serialization format (save objects to disk) | Freezing a Python object to thaw later |
| **SHA-256** | A cryptographic hash function | A fingerprint generator for text |
| **HNSW** | Hierarchical Navigable Small World — fast approximate nearest neighbor search | A smart filing system for finding close vectors quickly |
| **LangChain** | Framework for building LLM applications | A toolkit that connects LLMs, databases, and logic |
| **Ollama** | Run open-source LLMs locally | A personal AI model server on your laptop |
| **Groq** | Cloud service for fast LLM inference | A very fast remote AI brain |
| **Sentence Transformers** | Python library for computing embeddings | The tool that translates sentences into numbers |
| **Uvicorn** | ASGI web server that runs FastAPI | The engine that actually listens for HTTP requests |
| **HPA** | Horizontal Pod Autoscaler — Kubernetes auto-scaling | Automatic hiring/firing based on workload |

---

## 13. Full Data Flow — From Question to Answer

Here is the complete journey of a single question through the system:

```
User asks: "What was Arjuna's dilemma before the battle of Kurukshetra?"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (src/api/main.py)                                  │
│  Receives HTTP POST /query                                  │
│  Validates request shape (Pydantic)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  RAGPipeline.query() (src/pipeline.py)                      │
│                                                             │
│  Step 1: Guardrails Input Check                             │
│  ├── Is question about hate speech? No ✓                   │
│  ├── Is it Mahabharata-related? Yes ✓                      │
│  └── Allowed: True                                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Query Classification (src/retrieval/query_classifier.py)   │
│                                                             │
│  Pattern check: "What was" → possible narrative             │
│  Keyword check: "battle", "Arjuna" → entity + narrative     │
│  Semantic check: similar to narrative examples              │
│  Result: QueryType.NARRATIVE, confidence=0.72               │
│  Strategy: bm25_weight=0.5, dense_weight=0.5               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  HybridSearcher.search() (src/retrieval/hybrid_search.py)   │
│                                                             │
│  BM25 Search:                                               │
│  ├── Tokenize: ["what", "was", "arjuna", "dilemma", ...]   │
│  ├── Score 15,000+ chunks against tokens                    │
│  └── Top 10 by keyword frequency                            │
│                                                             │
│  Dense Search:                                              │
│  ├── Encode query → [0.23, -0.87, ..., 0.54] (768 dims)   │
│  ├── ChromaDB HNSW search                                   │
│  └── Top 10 by cosine similarity                            │
│                                                             │
│  Fusion:                                                    │
│  ├── Normalize both score lists to [0,1]                   │
│  ├── combined = 0.5*bm25 + 0.5*dense                       │
│  └── Sort, return Top 5 chunks                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Cache Check  │
                    │  Cache MISS   │
                    │  (first time) │
                    └───────┬───────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  AnswerGenerator.generate_answer() (src/generation/)        │
│                                                             │
│  PromptAssembler builds:                                    │
│  "You are Vyasa, an AI assistant...                         │
│   CRITICAL: Only use provided context...                    │
│   [Bhishma Parva, 25] Then Arjuna saw his kin...           │
│   [Bhishma Parva, 26] Krishna said: Rise above...          │
│   [Adi Parva, 3] Arjuna was born to Kunti...               │
│   Question: What was Arjuna's dilemma?"                     │
│                                                             │
│  LLM (Groq/llama-3.3-70b) generates answer:               │
│  "Arjuna faced a profound moral crisis [Bhishma Parva, 25] │
│   when he saw his teachers, kinsmen, and beloved relatives  │
│   arrayed on both sides of the battlefield..."              │
│                                                             │
│  Extract citations: ["[Bhishma Parva, 25]", ...]           │
│  Validate: All citations match provided context ✓          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Guardrails Output Check                                    │
│  ├── Hate speech in answer? No ✓                           │
│  ├── Has citations? Yes ✓                                  │
│  ├── Length appropriate (20-2000 chars)? Yes ✓             │
│  └── Allowed: True                                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Save to      │
                    │  Cache        │
                    │  (1 hour TTL) │
                    └───────┬───────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  HTTP Response to User                                      │
│  {                                                          │
│    "answer": "Arjuna faced a profound moral crisis...",    │
│    "sources": ["[Bhishma Parva, 25] - Bhagavad Gita"],    │
│    "retrieval_time_ms": 245.5,                              │
│    "generation_time_ms": 1250.3                             │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘

Total time: ~1.5 seconds on first query
           ~50ms on cached query (30x faster!)
```

---

## What You Should Learn from This Project

As a Gen AI student, this project demonstrates the **complete production RAG stack**. Here's a map of concepts to what you learned:

| Concept | Covered In | Key Takeaway |
|---------|-----------|--------------|
| **Embeddings** | ChromaDB indexing, dense search | Text → numbers that capture meaning |
| **Vector Search** | HybridSearcher dense search | Find semantically similar text |
| **BM25** | Keyword index | Traditional but still valuable |
| **Hybrid Search** | HybridSearcher | Combine multiple signals |
| **Prompt Engineering** | PromptAssembler | Instructions shape LLM behavior |
| **LLM Abstraction** | LLM Factory | Switch models without changing code |
| **RAG Evaluation** | Ragas metrics | Measure faithfulness + relevancy |
| **Caching** | ResponseCache / RedisCache | Speed + cost optimization |
| **Guardrails** | ContentGuardrails | Safety in production AI |
| **API Design** | FastAPI main | Expose AI as a service |
| **Containerization** | Docker | Portable, reproducible deployment |
| **CI/CD** | GitHub Actions | Automated quality gates |

The most important insight: **retrieval quality determines answer quality**. A perfect LLM with bad retrieval gives bad answers. A good LLM with good retrieval gives great answers. Most of the engineering effort in this project is in getting retrieval right.

---

*Document generated from codebase walkthrough. All code references are accurate as of the current state of the repository.*
