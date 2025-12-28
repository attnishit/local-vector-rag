# Local Vector RAG Database

> **A from-scratch implementation of a vector database and RAG pipeline using explicit algorithms**
---

## 🎯 What is this?

This project implements a **complete Retrieval-Augmented Generation (RAG) system from the ground up**, including:

- 📄 **Document ingestion** with intelligent chunking
- 🧮 **Local embedding generation** using sentence-transformers
- 🗄️ **Vector databases** with both exact and approximate search (HNSW)
- 🔍 **Query pipeline** for semantic search
- 📊 **Comprehensive benchmarking** tools to compare algorithms

**No frameworks. No abstractions. Just clean, well-tested implementations** that help you understand how modern vector databases and RAG systems actually work under the hood.

---

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py
```

### Basic Usage

```bash
# 1. Preview how documents are chunked
echo "Vector databases enable semantic search by converting text into embeddings." > data/raw/sample.txt
python main.py preview data/raw/sample.txt

# 2. Test embedding generation
python main.py embed-demo "What is a vector database?"

# 3. Test vector search with demo data
python main.py search-demo "semantic search using embeddings" --top-k 3

# 4. Test full query pipeline (brute-force)
python main.py query-demo "How do vector databases work?"

# 5. Test with HNSW approximate search
python main.py query-demo "nearest neighbor algorithms" --algorithm hnsw --top-k 5

# 6. Run performance benchmarks
python main.py benchmark --dataset-size 1000 --n-queries 100
```

---

## 🏗️ Architecture

The system is organized into clean, testable modules:

```
src/
├── ingestion/       # Document loading and chunking
├── embeddings/      # Sentence-transformers integration
├── vectorstore/     # Vector storage and search
│   ├── brute_force.py    # Exact nearest neighbor (baseline)
│   ├── hnsw.py           # Approximate nearest neighbor
│   └── similarity.py     # Distance metrics
├── query/           # End-to-end query pipeline
└── benchmarks/      # Performance evaluation tools
```

### How it works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────▶│   Chunking   │────▶│  Embeddings │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Results   │◀────│    Search    │◀────│ Vector Store│
└─────────────┘     └──────────────┘     └─────────────┘
                                          (Brute-force
                                           or HNSW)
```

1. **Ingestion**: Load `.txt` files, normalize whitespace, split into overlapping chunks
2. **Embedding**: Convert chunks to 384-dim vectors using `all-MiniLM-L6-v2`
3. **Indexing**: Store vectors with either:
   - **Brute-force**: Exact search (100% recall, O(n) time)
   - **HNSW**: Approximate search (90%+ recall, 10-100x faster)
4. **Query**: Embed query, find nearest neighbors, return ranked results

---

## 🔬 What You'll Learn

### 1. Text Chunking Strategies
- Fixed-size chunking with overlap
- Stable chunk IDs for reproducibility
- Whitespace normalization

### 2. Vector Embeddings
- How sentence-transformers work
- L2 normalization for cosine similarity
- Batch processing for efficiency
- Deterministic embedding generation

### 3. Similarity Metrics
- **Cosine similarity**: Angle between vectors (most common)
- **Dot product**: Magnitude + direction
- **Euclidean distance**: Straight-line distance

### 4. Search Algorithms

**Brute-force (Exact Search)**
- Compares query against all vectors
- O(n) time complexity
- 100% recall guaranteed
- Perfect for small datasets (<10k vectors)

**HNSW (Approximate Search)**
- Hierarchical navigable small world graphs
- Logarithmic-like search time
- Configurable accuracy/speed tradeoff
- Production-ready for millions of vectors

### 5. Performance Analysis
- Recall@k evaluation
- Latency benchmarking
- Memory usage comparison
- Scalability testing

---

## 📊 Performance Benchmarks

### Example Results (1,000 vectors, k=5)

| Algorithm    | Latency | Recall | Speedup |
|--------------|---------|--------|---------|
| Brute-force  | 2.5 ms  | 100%   | 1.0x    |
| HNSW (ef=10) | 0.3 ms  | 88%    | 8.3x    |
| HNSW (ef=50) | 0.8 ms  | 96%    | 3.1x    |
| HNSW (ef=100)| 1.2 ms  | 98%    | 2.1x    |

### Running Your Own Benchmarks

```bash
# Quick benchmark (1,000 vectors)
python main.py benchmark

# Large-scale test
python main.py benchmark --dataset-size 10000 --n-queries 500

# Compare across dataset sizes
python main.py benchmark --compare-sizes --dataset-sizes 100 1000 5000 10000

# Export results to JSON
python main.py benchmark --output results.json --verbose
```

---


## 📚 Configuration

All settings are in `config.yaml`:



## 🎓 Educational Resources

### Understanding HNSW
The HNSW implementation in `src/vectorstore/hnsw.py` includes detailed comments explaining:
- Multi-layer graph structure (like skip lists)
- Probabilistic level assignment
- Greedy graph traversal
- Bidirectional neighbor connections

### Key Papers
- [HNSW Algorithm (Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320)
- Sentence-Transformers documentation

### Code Structure
Each module is designed to be readable and educational:
- **Type hints** throughout
- **Google-style docstrings** with examples
- **Clear variable names**
- **Extensive inline comments** explaining algorithms

---

## 🚀 Use Cases

### 1. Learning Tool
Study vector database internals by reading and modifying the code:
```bash
# Experiment with different chunk sizes
# Edit config.yaml -> ingestion.chunk_size

# Try different similarity metrics
# Edit config.yaml -> vectorstore.similarity_metric

# Benchmark the changes
python main.py benchmark
```

### 2. Prototype RAG Systems
Build semantic search applications:
```python
from src.embeddings import load_embedding_model, embed_chunks
from src.vectorstore import create_hnsw_index
from src.query import create_query_pipeline

# Load your documents, create embeddings, build index
model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
# ... (see examples in main.py)
```

### 3. Research Platform
Experiment with custom algorithms:
- Modify similarity metrics in `src/vectorstore/similarity.py`
- Implement new chunking strategies in `src/ingestion/chunker.py`
- Test alternative embedding models
- Benchmark your changes with built-in tools

### 4. Educational Demonstrations
Great for teaching:
- "Here's how cosine similarity actually works"
- "This is what's inside a vector database"
- "Let's compare exact vs approximate search"

---

## 🛠️ Development

### Project Structure

```
rag/
├── main.py              # CLI entry point
├── config.yaml          # Configuration
├── src/                 # Source code
│   ├── config.py
│   ├── logger.py
│   ├── ingestion/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── query/
│   └── benchmarks/
├── tests/               # Test suite (262 tests)
├── data/
│   ├── raw/            # Input documents
│   └── embeddings/     # Stored indexes
└── logs/               # Application logs
```

### Contributing
This is an educational project. Feel free to:
- Fork and experiment
- Submit issues for bugs or questions
- Propose improvements via pull requests

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

---


**Built with 💙 for learning and understanding**

*Questions? Open an issue!*
