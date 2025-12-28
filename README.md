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
- 💾 **Persistent Collections** — Disk-based storage with incremental updates
- 📊 **Benchmarking Suite** — Compare recall, latency, and scalability

---

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/rag.git
cd rag
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Create a searchable collection from your documents
python main.py index data/raw/samples --name my_docs --algorithm hnsw

# 2. Search your collection
python main.py search "vector database algorithms" --collection my_docs --top-k 5

# 3. List all collections
python main.py list
```

That's it! Your documents are now semantically searchable.

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
```

**Pipeline Steps:**

1. **Ingestion** → Load documents, extract text, split into 512-character overlapping chunks
2. **Embedding** → Convert chunks to vectors using `all-MiniLM-L6-v2` (L2 normalized)
3. **Indexing** → Build searchable index with HNSW graph or brute-force array
4. **Query** → Embed query, find k-nearest neighbors, return ranked results

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

## Commands

### Index: Create a Collection

```bash
python main.py index <directory> [options]
```

**Options:**
- `--name <name>` — Collection name (default: my_collection)
- `--algorithm <hnsw|brute_force>` — Search algorithm (default: hnsw)
- `--test-query <text>` — Run test query after indexing

**Example:**
```bash
# Index all documents in a directory with HNSW
python main.py index data/raw/samples --name research_papers --algorithm hnsw

# Test immediately after indexing
python main.py index data/raw --name docs --test-query "machine learning"
```

### Search: Query a Collection

```bash
python main.py search "<query>" [options]
```

**Options:**
- `--collection <name>` — Collection to search (default: my_collection)
- `--top-k <n>` — Number of results (default: 5)
- `--min-score <float>` — Minimum similarity threshold (default: 0.0)
- `--ef-search <n>` — HNSW search parameter (higher = better recall)
- `--output <file>` — Save results to JSON file

**Example:**
```bash
# Basic search
python main.py search "neural networks" --collection research_papers

# High-recall search with score filtering
python main.py search "deep learning" --top-k 10 --min-score 0.5 --ef-search 100

# Export results
python main.py search "transformers" --output results.json
```

### List: View All Collections

```bash
python main.py list
```

Shows all collections with metadata (algorithm, document count, chunk count, creation date).

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

Compare algorithms and measure performance:

```bash
# Quick benchmark (1,000 vectors, 100 queries)
python main.py benchmark

# Large-scale test
python main.py benchmark --dataset-size 10000 --n-queries 500

# Compare scalability across sizes
python main.py benchmark --compare-sizes --sizes 100,1000,5000,10000
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
├── main.py                  # CLI entry point
├── config.yaml              # System configuration
├── src/
│   ├── collection.py        # High-level collection API
│   ├── ingestion/           # Document loading & chunking
│   │   ├── loader.py        # Multi-format document loader
│   │   ├── chunker.py       # Fixed-size chunking with overlap
│   │   └── extractors.py    # PDF/DOCX/Markdown text extraction
│   ├── embeddings/          # Embedding generation
│   │   ├── model.py         # Sentence-transformers wrapper
│   │   └── pipeline.py      # Batch embedding with L2 norm
│   ├── vectorstore/         # Vector search algorithms
│   │   ├── brute_force.py   # Exact search (O(n) baseline)
│   │   ├── hnsw.py          # HNSW approximate search
│   │   └── similarity.py    # Distance metrics (cosine, L2, dot)
│   ├── query/               # Query pipeline
│   └── benchmarks/          # Performance evaluation
├── data/
│   ├── raw/                 # Input documents
│   ├── processed/           # Chunked documents (JSON)
│   ├── embeddings/          # Vector embeddings (NPZ)
│   └── indexes/             # HNSW graphs (pickle)
└── tests/                   # Unit tests
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

### 1. Semantic Search Application
Build a production RAG system:
```python
from src.collection import load_collection

# Load pre-built collection
collection = load_collection("my_docs")

# Search
results = collection.search("How does HNSW work?", k=5)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text']}\n")
```

### 2. Research & Experimentation
- Compare similarity metrics (cosine, L2, dot product)
- Test different chunking strategies
- Benchmark custom embedding models
- Tune HNSW parameters for your dataset

### 3. Educational Tool
Great for teaching:
- "This is how vector databases work internally"
- "Here's the tradeoff between exact and approximate search"
- "Let's visualize the HNSW graph structure"

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

## Acknowledgments

Built with 💙 for learning and understanding.

Special thanks to:
- [HNSW authors](https://arxiv.org/abs/1603.09320) for the groundbreaking algorithm
- [Sentence-Transformers](https://www.sbert.net/) team for accessible embeddings
- Open-source community for inspiration

---

**Questions? Found a bug? Want to contribute?**
👉 [Open an issue](https://github.com/yourusername/rag/issues) or start a discussion!
