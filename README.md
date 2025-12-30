# Local Vector RAG Database

**A production-ready, from-scratch implementation of Retrieval-Augmented Generation (RAG) using explicit vector search algorithms.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances language models by retrieving relevant information from a knowledge base before generating responses. Instead of relying solely on pre-trained knowledge, RAG systems:

1. **Convert documents into vector embeddings** (numerical representations)
2. **Store embeddings in a vector database** for efficient similarity search
3. **Retrieve relevant context** by finding semantically similar chunks
4. **Generate informed responses** using retrieved information

This project implements the complete RAG pipeline from scratch, focusing on the **vector database layer** that powers semantic search. No frameworks, no black boxes—just clean, educational implementations of the algorithms that power modern AI systems.

---

## Core Features

- 📄 **Multi-format Document Ingestion** — PDF, DOCX, Markdown, TXT with intelligent chunking
- 🧮 **Local Embedding Generation** — Sentence-transformers (384-dim vectors, no API required)
- 🔍 **Two Search Algorithms:**
  - **Brute-force** — Exact nearest neighbor search (100% recall)
  - **HNSW** — Approximate search based on [Malkov & Yashunin (2018)](https://arxiv.org/abs/1603.09320)
- 🤖 **Local LLM Generation (NEW!)** — Complete RAG with Ollama (100% offline, no API keys)
  - Answer generation with citations
  - Interactive chat mode with conversation history
  - Custom prompt templates
  - Answer caching for repeated queries
  - Confidence scoring
- 💾 **Persistent Collections** — Disk-based storage with incremental updates
- 📊 **Benchmarking Suite** — Compare recall, latency, and scalability
- 🎯 **Production-Ready CLI** — Global `rag` command for seamless workflow

---

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/rag.git
cd rag
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies and CLI
pip install -e .
```

### Using the CLI (Recommended)

**⚠️ Important**: The `rag` command requires the virtual environment to be activated:

```bash
# Activate the virtual environment first
cd /path/to/rag
source venv/bin/activate  # Windows: venv\Scripts\activate

# You should see (venv) in your prompt
```

Once activated, the `rag` command works from anywhere, just like `git` or `docker`:

```bash
# 1. Validate setup
rag

# 2. Create a searchable collection from your documents
rag index ~/Documents/research --name research_papers --algorithm hnsw

# 3. Search your collection
rag search "vector database algorithms" --collection research_papers --top-k 5

# 4. Generate answers with LLM (NEW!)
rag generate "How does HNSW work?" --collection research_papers --stream

# 5. Interactive chat mode (NEW!)
rag chat --collection research_papers

# 6. List all collections
rag list

# 7. View collection details
rag info research_papers
```

That's it! Your documents are now semantically searchable.

### Alternative: Using Python Script (Legacy)

For validation and benchmarking, you can also use `main.py`:

```bash
# Validate system setup
python main.py

# Run benchmarks
python main.py benchmark --dataset-size 1000

# For all other operations, use the 'rag' CLI
```

**Note:** All document indexing, searching, and generation commands have been moved to the `rag` CLI for better user experience.

---

## How It Works

### Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────▶│   Chunking   │────▶│  Embeddings │
│  (PDF/DOCX) │     │   512 chars  │     │   384-dim   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Results   │◀────│    Search    │◀────│ Vector Index│
│  (Ranked)   │     │  (Cosine)    │     │  (HNSW/BF)  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  LLM (Ollama)│
                    │   Citations  │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │    Answer    │
                    │ with Sources │
                    └──────────────┘
```

**Pipeline Steps:**

1. **Ingestion** → Load documents, extract text, split into 512-character overlapping chunks
2. **Embedding** → Convert chunks to vectors using `all-MiniLM-L6-v2` (L2 normalized)
3. **Indexing** → Build searchable index with HNSW graph or brute-force array
4. **Query** → Embed query, find k-nearest neighbors, return ranked results
5. **Generation (Optional)** → Use local LLM to generate answers with retrieved context

### The HNSW Algorithm

This implementation is based on **"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"** ([Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)).

**Key Concepts:**

- **Hierarchical layers** — Like skip lists, higher layers skip across the dataset
- **Navigable small world** — Short paths exist between any two nodes
- **Greedy routing** — Start at top layer, greedily descend to nearest neighbors
- **Probabilistic insertion** — New nodes assigned to layers by `⌊-ln(uniform(0,1)) × mL⌋`

**Performance characteristics:**
- **Time complexity:** ~O(log n) search
- **Recall:** 90-99% depending on `ef_search` parameter
- **Speedup:** 10-100x faster than brute-force on large datasets

**Parameters:**
- `m=16` — Bidirectional links per node (higher = better recall, more memory)
- `ef_construction=200` — Candidate list size during build
- `ef_search=50` — Candidate list size during query (tune for recall/speed tradeoff)

See `src/vectorstore/hnsw.py` for detailed implementation with inline explanations.

---

## LLM Generation (NEW!)

This system now includes **complete RAG capabilities** with local LLM generation using **Ollama**. Generate answers from your documents with citations, confidence scores, and conversation history—all running 100% offline.

### Features

- ✅ **100% Local/Offline** — No API keys, no external services, no internet required after setup
- ✅ **Answer Generation** — Get AI-generated answers with automatic citation markers [1], [2]
- ✅ **Interactive Chat Mode** — Multi-turn conversations with conversation history
- ✅ **Streaming Responses** — Word-by-word output for better UX
- ✅ **Confidence Scoring** — Know how reliable each answer is
- ✅ **Custom Prompt Templates** — Tailor prompts for your use case
- ✅ **Answer Caching** — Speeds up repeated queries

### Setup (One-Time)

```bash
# 1. Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai for other platforms

# 2. Start Ollama server
ollama serve

# 3. Download a model (one-time, ~4GB)
ollama pull llama2:7b

# Done! Model is stored locally and works offline
```

### Quick Examples

```bash
# Generate a single answer
rag generate "How does HNSW improve search performance?" \
  --collection research_papers \
  --stream

# Interactive chat with conversation history
rag chat --collection research_papers

# Use custom prompt template
rag generate "Explain vector databases" \
  --collection my_docs \
  --custom-template templates/expert.j2

# Adjust generation parameters
rag generate "What is semantic search?" \
  --collection docs \
  --model llama2:7b \
  --temperature 0.5 \
  --top-k 10
```

### Available Models

| Model | Size | RAM | Speed | Quality |
|-------|------|-----|-------|---------|
| `phi:2.7b` | 1.6GB | 4GB | Fast | Good |
| `llama2:7b` | 3.8GB | 8GB | Medium | Excellent ⭐ |
| `mistral:7b` | 4.1GB | 8GB | Medium | Excellent |
| `llama2:13b` | 7.4GB | 16GB | Slow | Best |

Download any model with: `ollama pull <model_name>`

⭐ = Default model

### How It Works

1. **Retrieve** → Find top-k relevant chunks from your collection
2. **Prepare Context** → Add citation markers [1], [2] to chunks
3. **Generate** → Local LLM generates answer using context
4. **Mark Citations** → Track which sources were actually used
5. **Calculate Confidence** → Score based on retrieval quality

**Example Output:**
```
Answer:
────────────────────────────────────────────────────────────────
HNSW (Hierarchical Navigable Small World) improves search
performance by using a multi-layer graph structure [1]. Unlike
brute-force search which compares against all vectors, HNSW
navigates through layers to find approximate nearest neighbors
in O(log n) time [2]. This provides 10-100x speedup while
maintaining 90-99% recall [1].

Sources:
[1] hnsw_paper_chunk_5 (score: 0.89) - "HNSW uses hierarchical layers..."
[2] algorithms_doc_chunk_12 (score: 0.85) - "Time complexity analysis..."

Confidence: 0.87 (High)
```

---

## CLI Reference

> **📖 Complete Documentation**: For detailed CLI documentation including all options, workflows, troubleshooting, and advanced usage, see:
> - **[Complete CLI User Guide](src/cli/USER_GUIDE.md)** - Comprehensive 1000+ line guide

### Prerequisites

**Important**: Activate your virtual environment before using `rag` commands:

```bash
cd /path/to/rag
source venv/bin/activate  # You should see (venv) in your prompt
```

### Global RAG Command

After installation with `pip install -e .` and activating the virtual environment, the `rag` command becomes available. Use it from any directory to manage your document collections.

### Quick Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `rag` | Validate system setup | `rag` |
| `rag index` | Create collection from documents | `rag index ~/docs --name my_collection` |
| `rag search` | Search a collection | `rag search "query" --collection my_docs` |
| `rag generate` | Generate answer with LLM (NEW!) | `rag generate "question" --collection my_docs` |
| `rag chat` | Interactive chat mode (NEW!) | `rag chat --collection my_docs` |
| `rag list` | List all collections | `rag list` |
| `rag info` | Show collection details | `rag info my_collection` |
| `rag delete` | Delete a collection | `rag delete old_collection` |
| `rag preview` | Preview document chunks | `rag preview document.pdf` |
| `rag benchmark` | Run performance tests | `rag benchmark --dataset-size 1000` |

### Common Options

| Option | Commands | Description |
|--------|----------|-------------|
| `--name` | `index` | Collection name |
| `--algorithm` | `index` | Algorithm: `hnsw` or `brute_force` |
| `--collection` | `search` | Which collection to search |
| `--top-k` | `search` | Number of results to return |
| `--min-score` | `search` | Minimum similarity threshold (0.0-1.0) |
| `--ef-search` | `search` | HNSW accuracy parameter |
| `--output` | `search`, `benchmark` | Export results to JSON |
| `--force` | `delete` | Skip confirmation prompt |

### Getting Help

```bash
# General help
rag --help

# Command-specific help
rag search --help
rag index --help
rag benchmark --help
```

**📖 For detailed CLI documentation, see [CLI User Guide](src/cli/USER_GUIDE.md)**

---

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
ingestion:
  chunk_size: 512              # Characters per chunk
  chunk_overlap: 50            # Overlap between chunks
  supported_formats: [txt, pdf, docx, md]

embeddings:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu                  # or "cuda" for GPU
  dimension: 384
  normalize: true              # L2 normalization for cosine similarity

vectorstore:
  algorithm: hnsw              # or "brute_force"
  similarity_metric: cosine
  hnsw:
    m: 16                      # Links per node
    ef_construction: 200       # Build-time accuracy
    ef_search: 50              # Query-time accuracy
```

---

## Benchmarking

Compare algorithms and measure performance using either the `rag` CLI or `python main.py`:

```bash
# Using the rag CLI (recommended)
rag benchmark

# Or using python main.py (legacy)
python main.py benchmark

# Quick benchmark (1,000 vectors, 100 queries)
rag benchmark --dataset-size 1000

# Large-scale test
rag benchmark --dataset-size 10000 --n-queries 500

# Compare scalability across sizes
rag benchmark --compare-sizes --dataset-sizes 100 1000 5000 10000
```

**Sample Results (1,000 vectors, k=5):**

| Algorithm    | Query Time | Recall | Memory  | Speedup |
|--------------|------------|--------|---------|---------|
| Brute-force  | 2.5 ms     | 100%   | 1.5 MB  | 1.0x    |
| HNSW (ef=10) | 0.3 ms     | 88%    | 4.2 MB  | 8.3x    |
| HNSW (ef=50) | 0.8 ms     | 96%    | 4.2 MB  | 3.1x    |
| HNSW (ef=100)| 1.2 ms     | 98%    | 4.2 MB  | 2.1x    |

---

## Project Structure

```
rag/
├── main.py                  # Legacy entry point (validation & benchmarks only)
├── config.yaml              # System configuration
├── pyproject.toml           # Package metadata & dependencies
├── src/
│   ├── cli/                 # Modern CLI interface (PRIMARY)
│   │   ├── __init__.py      # CLI entry point ('rag' command)
│   │   ├── commands.py      # All command implementations
│   │   └── USER_GUIDE.md    # Detailed CLI documentation
│   ├── collection.py        # High-level collection API
│   ├── ingestion/           # Document loading & chunking
│   │   ├── loader.py        # Multi-format document loader
│   │   ├── chunker.py       # Fixed-size chunking with overlap
│   │   └── extractors/      # PDF/DOCX/Markdown text extraction
│   ├── embeddings/          # Embedding generation
│   │   ├── model.py         # Sentence-transformers wrapper
│   │   └── pipeline.py      # Batch embedding with L2 norm
│   ├── vectorstore/         # Vector search algorithms
│   │   ├── brute_force.py   # Exact search (O(n) baseline)
│   │   ├── hnsw.py          # HNSW approximate search
│   │   └── similarity.py    # Distance metrics (cosine, L2, dot)
│   ├── generation/          # LLM answer generation (NEW!)
│   │   ├── model.py         # Ollama client management
│   │   ├── prompts.py       # Prompt templates
│   │   ├── citations.py     # Citation extraction
│   │   └── conversation.py  # Multi-turn chat history
│   ├── query/               # Query pipeline
│   └── benchmarks/          # Performance evaluation
├── data/
│   ├── raw/                 # Input documents
│   ├── processed/           # Chunked documents (JSON)
│   ├── embeddings/          # Vector embeddings (NPZ)
│   └── indexes/             # HNSW graphs (pickle)
└── tests/                   # Unit tests (TODO)
```

---

## Supported Document Formats

| Format       | Extensions         | Extraction      | Notes                      |
|--------------|--------------------|-----------------|----------------------------|
| Plain Text   | `.txt`             | Direct read     | UTF-8 encoding             |
| PDF          | `.pdf`             | PyMuPDF (fitz)  | Multi-page with markers    |
| Word         | `.docx`, `.doc`    | python-docx     | Preserves headings/tables  |
| Markdown     | `.md`, `.markdown` | Regex parser    | Preserves headers/links    |

Place documents in `data/raw/` or any directory, then run `python main.py index <directory>`.

---

## Learning Resources

### Understanding Vector Search

Each module is designed for education:
- **Type hints** throughout
- **Google-style docstrings** with examples
- **Inline algorithm explanations** in code

Key files to read:
- `src/vectorstore/hnsw.py` — HNSW implementation with detailed comments
- `src/embeddings/pipeline.py` — Batch embedding generation
- `src/ingestion/chunker.py` — Text chunking strategies

### Foundational Papers

- **HNSW:** [Efficient and robust approximate nearest neighbor search](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2018)
- **Sentence-Transformers:** [Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)

---

## Use Cases

### 1. Complete RAG System with AI Answers (NEW!)
Build a production RAG system with answer generation:

```bash
# Setup Ollama (one-time)
brew install ollama
ollama serve
ollama pull llama2:7b

# Index your documentation
rag index ~/company/docs --name company_knowledge --algorithm hnsw

# Get AI-generated answers with citations
rag generate "How do I configure authentication?" \
  --collection company_knowledge \
  --stream

# Interactive chat mode for follow-up questions
rag chat --collection company_knowledge
```

### 2. Semantic Search Application (CLI)
Build a search engine using the global command:

```bash
# Index your documentation
rag index ~/company/docs --name company_knowledge --algorithm hnsw

# Search from anywhere
rag search "How do I configure authentication?" --collection company_knowledge

# Export results for integration
rag search "API rate limits" --collection company_knowledge --output api_info.json
```

### 3. Programmatic Access (Python API)
For custom applications, use the Python API directly:

```python
from src.collection import load_collection

# Load pre-built collection
collection = load_collection("my_docs")

# Search
results = collection.search("How does HNSW work?", k=5)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['metadata']['text']}\n")

# Generate answer with LLM (NEW!)
answer_result = collection.generate_answer(
    query="How does HNSW work?",
    k=5,
    stream=False,
    template="qa"
)

print(f"Answer: {answer_result['answer']}")
print(f"Confidence: {answer_result['confidence']:.2f}")

# Show cited sources
for source in answer_result['sources']:
    if source['cited']:
        print(f"[{source['citation_num']}] {source['chunk_id']}")
```

### 4. Research & Experimentation
- Compare similarity metrics (cosine, L2, dot product)
- Test different chunking strategies
- Benchmark custom embedding models
- Tune HNSW parameters for your dataset
- Experiment with different LLM models and prompts (NEW!)

```bash
# Test different HNSW parameters
rag search "query" --ef-search 10   # Fast, lower recall
rag search "query" --ef-search 100  # Slower, higher recall

# Compare different LLM models
rag generate "query" --model llama2:7b    # Default, good balance
rag generate "query" --model llama2:13b    # Slower, better quality

# Test custom prompts
rag generate "query" --custom-template my_expert_prompt.j2

# Run comprehensive benchmarks
rag benchmark --compare-sizes --dataset-sizes 1000 5000 10000
```

### 5. Educational Tool
Great for teaching:
- "This is how vector databases work internally"
- "Here's the tradeoff between exact and approximate search"
- "Let's visualize the HNSW graph structure"

```bash
# Show students how documents are chunked
rag preview research_paper.pdf --num-chunks 10

# Compare algorithm performance with detailed stats
rag benchmark --verbose --dataset-size 1000

# Compare scalability across different dataset sizes
rag benchmark --compare-sizes --dataset-sizes 100 1000 5000
```

---

## Contributing

We welcome contributions! This project is designed to be:
- **Educational** — Clear code over clever code
- **Extensible** — Easy to add new algorithms
- **Well-tested** — Comprehensive test suite

**How to contribute:**

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** (add tests if applicable)
4. **Run tests** (`pytest tests/`)
5. **Submit a Pull Request**

**Ideas for contributions:**
- Add new document extractors (HTML, CSV, JSON)
- Implement alternative indexing algorithms (LSH, Product Quantization)
- Add vector compression techniques
- Improve benchmark visualizations
- Write tutorials or example notebooks

**Questions or suggestions?** Open an issue—we're happy to discuss ideas!

---

## License

This project is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2025 Nishit Attrey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**TL;DR:** You can use, modify, and distribute this software freely. Contributions are welcome!

---

**Questions? Found a bug? Want to contribute?**
👉 [Open an issue](https://github.com/yourusername/rag/issues) or start a discussion!
