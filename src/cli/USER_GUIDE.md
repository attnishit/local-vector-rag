# RAG CLI User Guide

**Version**: 0.2.0
**Last Updated**: 2025-12-29

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Command Reference](#command-reference)
   - [Setup Commands](#setup-commands)
   - [Collection Commands](#collection-commands)
   - [Search Commands](#search-commands)
   - [LLM Generation Commands (NEW!)](#llm-generation-commands-new)
   - [Utility Commands](#utility-commands)
5. [Common Workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Introduction

The RAG CLI is a command-line interface for the Local Vector RAG Database system. It provides a simple, intuitive way to:

- Create and manage document collections
- Perform semantic search on your documents
- **Generate AI-powered answers with local LLMs (NEW!)**
- **Interactive chat mode with conversation history (NEW!)**
- Run performance benchmarks
- Preview and analyze document chunks

The CLI is designed to work globally from any directory, just like familiar tools such as `git`, `docker`, or `npm`.

**What's New in 0.2.0:**
- 🤖 Local LLM answer generation with Ollama
- 💬 Interactive chat mode with multi-turn conversations
- 📑 Automatic citation marking in generated answers
- 🎯 Confidence scoring for answer quality
- 🎨 Custom prompt templates support
- ⚡ Answer caching for repeated queries

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

### Install from Source

```bash
# Clone the repository
cd /path/to/local-vector-rag

# Install in editable mode (recommended for development)
pip install -e .

# Verify installation
rag --version
```

### Install from GitHub (Future)

```bash
pip install git+https://github.com/yourusername/local-vector-rag.git
```

After installation, the `rag` command will be available globally in your terminal.

---

## Quick Start

### 1. Validate Setup

Check that everything is configured correctly:

```bash
rag
```

**Output:**
```
============================================================
              Local Vector RAG v0.1.0
============================================================

Validating setup...

Data directory: /path/to/project/data
  Exists: True
  Raw documents: True
  Processed chunks: True
  Embeddings: True

✓ Setup validated!
```

### 2. Create Your First Collection

Index a directory of documents:

```bash
rag index /path/to/documents --name my_docs
```

**Example:**
```bash
rag index ~/Documents/research_papers --name research
```

### 3. Search Your Collection

```bash
rag search "machine learning" --collection my_docs --top-k 5
```

### 4. List All Collections

```bash
rag list
```

---

## Command Reference

### Global Options

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help message and exit |

### Commands Overview

| Command | Description | Example |
|---------|-------------|---------|
| `rag` | Validate setup | `rag` |
| `rag index` | Create a collection from documents | `rag index docs/ --name my_collection` |
| `rag search` | Search a collection | `rag search "query" --collection my_docs` |
| `rag list` | List all collections | `rag list` |
| `rag info` | Show collection details | `rag info my_collection` |
| `rag delete` | Delete a collection | `rag delete my_collection` |
| `rag preview` | Preview document chunks | `rag preview document.pdf` |
| `rag embed-demo` | Test embedding generation | `rag embed-demo "sample text"` |
| `rag benchmark` | Run performance benchmarks | `rag benchmark --dataset-size 1000` |

---

## Detailed Command Reference

### 1. `rag index` - Create Collection

Create a new searchable collection from documents.

**Syntax:**
```bash
rag index <directory> [OPTIONS]
```

**Arguments:**
- `directory` - Path to directory containing documents (default: `data/raw/samples`)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--name` | string | `my_collection` | Name for the collection |
| `--algorithm` | choice | `hnsw` | Vector search algorithm (`brute_force`, `hnsw`) |
| `--test-query` | string | None | Run a test search after indexing |

**Examples:**

```bash
# Create collection with default settings
rag index ~/Documents

# Create collection with custom name and algorithm
rag index ~/Documents --name research_papers --algorithm hnsw

# Create and test immediately
rag index ~/Documents --name papers --test-query "neural networks"
```

**Supported File Formats:**
- `.txt` - Plain text files
- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents
- `.md` - Markdown files

**Output:**
```
Creating collection 'research_papers' from: /Users/you/Documents
Algorithm: hnsw

Loading documents...
✓ Loaded 15 documents

Processing documents...
✓ Created 128 chunks

Generating embeddings...
✓ Generated 128 embeddings

Building HNSW index...
✓ Index built successfully

✓ Collection 'research_papers' created successfully!
  Documents: 15
  Chunks: 128
  Embeddings: 128
  Algorithm: hnsw
```

---

### 2. `rag search` - Search Collection

Search for semantically similar content in a collection.

**Syntax:**
```bash
rag search <query> [OPTIONS]
```

**Arguments:**
- `query` - Search query text (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--collection` | string | `my_collection` | Collection name to search |
| `--top-k` | integer | `5` | Number of results to return |
| `--min-score` | float | `0.0` | Minimum similarity score (0.0-1.0) |
| `--ef-search` | integer | None | HNSW search parameter (higher = more accurate) |
| `--output` | string | None | Export results to JSON file |

**Examples:**

```bash
# Basic search
rag search "machine learning algorithms" --collection research

# Get top 10 results with minimum score threshold
rag search "neural networks" --collection papers --top-k 10 --min-score 0.3

# Export results to JSON
rag search "deep learning" --collection papers --output results.json

# Fine-tune HNSW search accuracy
rag search "transformer models" --collection papers --ef-search 100
```

**Output:**
```
================================================================================
Search: 'machine learning'
Collection: research_papers
================================================================================

Loading collection 'research_papers'...
✓ Collection loaded
  Documents: 15
  Chunks: 128
  Embeddings: 128
  Algorithm: hnsw

Searching...
✓ Found 5 results

Search Results (Top 5):

1. Score: 0.8524
   Chunk ID: paper_01.pdf_chunk_0_abc123
   Text: Machine learning is a subset of artificial intelligence that
   enables systems to learn and improve from experience...

2. Score: 0.7891
   Chunk ID: paper_03.pdf_chunk_2_def456
   Text: Supervised learning algorithms require labeled training data...

[...]
```

---

### 3. `rag list` - List Collections

Display all available collections with their statistics.

**Syntax:**
```bash
rag list
```

**No options required.**

**Example:**
```bash
rag list
```

**Output:**
```
================================================================================
Collections
================================================================================

Found 3 collection(s):

1. research_papers
   Algorithm: hnsw
   Documents: 15
   Chunks: 128
   Embeddings: 128
   Created: 2025-12-28T10:30:45.123456

2. personal_notes
   Algorithm: brute_force
   Documents: 8
   Chunks: 42
   Embeddings: 42
   Created: 2025-12-27T15:20:10.987654

3. technical_docs
   Algorithm: hnsw
   Documents: 23
   Chunks: 256
   Embeddings: 256
   Created: 2025-12-26T09:15:30.456789

To search a collection:
  rag search "your query" --collection <name>
```

---

### 4. `rag info` - Collection Information

Display detailed information about a specific collection.

**Syntax:**
```bash
rag info <collection>
```

**Arguments:**
- `collection` - Name of the collection (required)

**Example:**
```bash
rag info research_papers
```

**Output:**
```
================================================================================
Collection Info: research_papers
================================================================================

Name: research_papers
Algorithm: hnsw
Documents: 15
Chunks: 128
Embeddings: 128

Index statistics:
  num_nodes: 128
  max_level: 12
  entry_point: 45
  dimension: 384
  m: 16
  m0: 32
  memory_mb: 3.2

Data locations:
  Chunks: data/processed
  Embeddings: data/embeddings
  Index: data/indexes/hnsw
```

---

### 5. `rag delete` - Delete Collection

Permanently delete a collection and all its data.

**Syntax:**
```bash
rag delete <collection> [OPTIONS]
```

**Arguments:**
- `collection` - Name of the collection to delete (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--force`, `-f` | flag | False | Skip confirmation prompt |

**Examples:**

```bash
# Delete with confirmation
rag delete old_collection

# Delete without confirmation
rag delete old_collection --force
```

**Output (with confirmation):**
```
================================================================================
Delete Collection: old_collection
================================================================================

Are you sure you want to delete 'old_collection'? (yes/no): yes
✓ Collection 'old_collection' deleted successfully!
```

**⚠️ WARNING:** This operation is irreversible. All chunks, embeddings, and index data will be permanently deleted.

---

### 6. `rag preview` - Preview Document Chunks

Preview how a document will be chunked before indexing.

**Syntax:**
```bash
rag preview <file> [OPTIONS]
```

**Arguments:**
- `file` - Path to document file (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--num-chunks` | integer | `3` | Number of chunks to preview |
| `--max-preview-length` | integer | `500` | Max characters per chunk preview |

**Examples:**

```bash
# Preview first 3 chunks
rag preview document.pdf

# Preview first 5 chunks with longer text
rag preview document.pdf --num-chunks 5 --max-preview-length 1000
```

**Output:**
```
================================================================================
Document Preview (Stage 2: Document Loading & Chunking)
================================================================================

Loading document: document.pdf
✓ Loaded 5,432 characters

Chunking with size=512, overlap=50...
✓ Created 12 chunks

Statistics:
  Total chunks: 12
  Avg length: 487.3 characters
  Min length: 256 characters
  Max length: 512 characters
  Std dev: 45.8 characters

Preview of first 3 chunk(s):

Chunk 1 (512 chars):
  Machine Learning Fundamentals Machine learning is a subset...

Chunk 2 (498 chars):
  Types of Machine Learning 1. Supervised Learning - Training...

Chunk 3 (506 chars):
  Applications of machine learning include computer vision...
```

---

### 7. `rag embed-demo` - Test Embedding Generation

Generate and inspect an embedding for sample text.

**Syntax:**
```bash
rag embed-demo <text>
```

**Arguments:**
- `text` - Text to generate embedding for (required)

**Example:**
```bash
rag embed-demo "machine learning algorithms"
```

**Output:**
```
================================================================================
Embedding Demo (Stage 3: Local Embedding Pipeline)
================================================================================

Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...
✓ Model loaded

Model dimension: 384

Generating embedding for:
  "machine learning algorithms"

Embedding shape: (384,)
Embedding preview (first 10 values):
  [ 0.1234 -0.5678  0.9012 ... ]

Embedding statistics:
  Min: -0.8234
  Max: 0.9123
  Mean: 0.0012
  Std: 0.3456
```

---

### 8. `rag benchmark` - Performance Benchmarks

Run comprehensive performance benchmarks comparing search algorithms.

**Syntax:**
```bash
rag benchmark [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset-size` | integer | `1000` | Number of vectors to index |
| `--n-queries` | integer | `100` | Number of queries to run |
| `--k-values` | integers | `[1, 5, 10]` | List of k values to test |
| `--ef-search-values` | integers | `[10, 50, 100]` | HNSW ef_search values |
| `--seed` | integer | `42` | Random seed for reproducibility |
| `--verbose` | flag | False | Show detailed statistics |
| `--output` | string | None | Export results to JSON |
| `--compare-sizes` | flag | False | Compare across dataset sizes |
| `--dataset-sizes` | integers | `[100, 1000, 5000]` | Sizes for comparison mode |

**Examples:**

```bash
# Basic benchmark
rag benchmark

# Large-scale benchmark with detailed output
rag benchmark --dataset-size 10000 --n-queries 500 --verbose

# Compare across multiple dataset sizes
rag benchmark --compare-sizes --dataset-sizes 100 1000 5000 10000

# Export results
rag benchmark --output benchmark_results.json
```

**Output:**
```
================================================================================
Benchmark Execution
================================================================================

Running benchmark with:
  Dataset size: 1,000 vectors
  Queries: 100
  k values: [1, 5, 10]
  ef_search values: [10, 50, 100]

This may take a few minutes...

[Benchmark results with tables showing latency, recall, memory usage...]

✓ Benchmark complete!
```

---

## Common Workflows

### Workflow 1: Index and Search Personal Documents

```bash
# 1. Index your documents
rag index ~/Documents/notes --name personal_notes

# 2. List to confirm creation
rag list

# 3. Search for specific topics
rag search "project ideas" --collection personal_notes --top-k 10

# 4. View collection details
rag info personal_notes
```

### Workflow 2: Research Paper Analysis

```bash
# 1. Index research papers
rag index ~/Research/papers --name research_collection --algorithm hnsw

# 2. Search for related work
rag search "neural network architectures" --collection research_collection --top-k 20

# 3. Filter high-quality results
rag search "transformer models" --collection research_collection --min-score 0.5

# 4. Export results for later analysis
rag search "attention mechanisms" --collection research_collection --output results.json
```

### Workflow 3: Document Preview and Optimization

```bash
# 1. Preview how documents will be chunked
rag preview sample_doc.pdf --num-chunks 5

# 2. Adjust chunk settings in config.yaml if needed
# 3. Index with optimized settings
rag index documents/ --name optimized_collection
```

### Workflow 4: Performance Testing

```bash
# 1. Run basic benchmark
rag benchmark --dataset-size 1000

# 2. Compare algorithms at scale
rag benchmark --compare-sizes --dataset-sizes 1000 5000 10000

# 3. Export for analysis
rag benchmark --dataset-size 5000 --verbose --output perf_results.json
```

---

## Troubleshooting

### Issue: Command Not Found

**Problem:**
```bash
rag: command not found
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep local-vector-rag
   ```

2. **Reinstall the package:**
   ```bash
   cd /path/to/project
   pip install -e .
   ```

3. **Check Python environment:**
   ```bash
   which python
   which pip
   ```
   Make sure you're using the correct virtual environment.

4. **Restart terminal** after installation

---

### Issue: Collection Not Found

**Problem:**
```
✗ ERROR: Collection 'my_docs' not found
```

**Solutions:**

1. **List available collections:**
   ```bash
   rag list
   ```

2. **Verify collection name** (case-sensitive)

3. **Check data directory** exists and contains collection metadata:
   ```bash
   cat data/collections.json
   ```

---

### Issue: Out of Memory During Indexing

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce batch size** in `config.yaml`:
   ```yaml
   embeddings:
     batch_size: 8  # Reduce from default 32
   ```

2. **Process fewer documents at once:**
   ```bash
   # Index in smaller batches
   rag index documents/batch1 --name collection_part1
   rag index documents/batch2 --name collection_part2
   ```

3. **Use CPU instead of GPU** if memory is limited (configured in config.yaml)

---

### Issue: Slow Search Performance

**Problem:**
Searches take too long to complete.

**Solutions:**

1. **Use HNSW algorithm** instead of brute-force:
   ```bash
   rag index documents/ --name fast_collection --algorithm hnsw
   ```

2. **Adjust ef_search parameter** for speed/accuracy tradeoff:
   ```bash
   # Faster but less accurate
   rag search "query" --collection my_docs --ef-search 20

   # Slower but more accurate
   rag search "query" --collection my_docs --ef-search 200
   ```

3. **Reduce top-k results:**
   ```bash
   rag search "query" --collection my_docs --top-k 3
   ```

---

### Issue: Poor Search Results

**Problem:**
Search results are not relevant to the query.

**Solutions:**

1. **Use more specific queries:**
   - ❌ "algorithms"
   - ✅ "machine learning classification algorithms"

2. **Adjust minimum score threshold:**
   ```bash
   rag search "query" --collection my_docs --min-score 0.3
   ```

3. **Check chunk size** in config.yaml:
   ```yaml
   ingestion:
     chunk_size: 512  # Experiment with 256, 512, 1024
     chunk_overlap: 50
   ```

4. **Preview chunks** to ensure proper document segmentation:
   ```bash
   rag preview problematic_document.pdf --num-chunks 10
   ```

---

## Advanced Usage

### Custom Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Chunking settings
ingestion:
  chunk_size: 512          # Characters per chunk
  chunk_overlap: 50        # Overlap between chunks

# Embedding settings
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  normalize: true

# HNSW algorithm parameters
vectorstore:
  hnsw:
    m: 16                  # Connectivity (higher = better recall, more memory)
    ef_construction: 200   # Build-time accuracy (higher = slower build)
    ef_search: 50          # Search-time accuracy (higher = slower search)
```

### Using Different Embedding Models

1. Edit `config.yaml`:
   ```yaml
   embeddings:
     model: "sentence-transformers/all-mpnet-base-v2"  # Higher quality
     dimension: 768  # Update dimension accordingly
   ```

2. Re-index collections with the new model

### Batch Processing Multiple Collections

```bash
#!/bin/bash
# batch_index.sh

collections=(
  "research_papers:~/Documents/Research"
  "technical_docs:~/Documents/Technical"
  "meeting_notes:~/Documents/Meetings"
)

for item in "${collections[@]}"; do
  IFS=':' read -r name path <<< "$item"
  echo "Indexing $name from $path..."
  rag index "$path" --name "$name" --algorithm hnsw
done

echo "All collections indexed!"
rag list
```

### Programmatic Access

While the CLI is convenient, you can also use the Python API directly:

```python
from src.collection import create_collection, load_collection
from pathlib import Path

# Create collection
collection = create_collection(
    name="my_collection",
    source_path=Path("documents/"),
    algorithm="hnsw"
)

# Search
results = collection.search("machine learning", k=10)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['metadata']['text'][:100]}...")
```

---

## Tips and Best Practices

### 1. Collection Naming

- Use descriptive names: `research_papers_2024` instead of `collection1`
- Avoid spaces: use `my_documents` not `my documents`
- Use lowercase for consistency

### 2. Indexing Strategy

- **Small collections (<1000 docs)**: Use `brute_force` for simplicity
- **Large collections (>1000 docs)**: Use `hnsw` for speed
- **Test queries**: Always use `--test-query` when creating collections
- **Incremental updates**: Re-index only when documents change significantly

### 3. Search Optimization

- Start with `--top-k 5` and adjust based on needs
- Use `--min-score 0.3` to filter low-quality results
- For HNSW, `--ef-search 50` is a good balance
- Export results to JSON for further analysis

### 4. Data Organization

```
project/
├── data/
│   ├── raw/              # Original documents
│   │   ├── research/
│   │   ├── notes/
│   │   └── technical/
│   ├── processed/        # Generated chunks (auto)
│   ├── embeddings/       # Generated embeddings (auto)
│   └── indexes/          # Vector indexes (auto)
```

### 5. Performance Monitoring

- Run benchmarks before and after configuration changes
- Monitor collection sizes: `rag info <collection>`
- Use `--verbose` flag for detailed performance data

---

## Configuration Reference

### Key Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `config.yaml` | Main configuration | Project root |
| `pyproject.toml` | Package metadata | Project root |
| `data/collections.json` | Collection metadata | Auto-generated |

### Important Paths

| Path | Description |
|------|-------------|
| `data/raw/` | Source documents |
| `data/processed/` | Chunked documents |
| `data/embeddings/` | Vector embeddings |
| `data/indexes/` | Search indexes |
| `logs/` | Application logs |

---

## Getting Help

### Command-Specific Help

```bash
# General help
rag --help

# Command-specific help
rag search --help
rag index --help
rag benchmark --help
```

### Verbose Logging

For debugging, check the logs:

```bash
tail -f logs/rag.log
```

Or set log level in `config.yaml`:

```yaml
logging:
  level: "DEBUG"  # More detailed logs
```

### Community Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check the main README.md
- **Examples**: See `examples/` directory (if available)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.0 | 2025-12-29 | Added LLM generation: `generate` and `chat` commands, custom templates, answer caching |
| 0.1.0 | 2025-12-28 | Initial CLI release with core commands |

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Appendix: Command Quick Reference

```bash
# Setup
rag                                           # Validate setup
rag --version                                 # Show version

# Collections
rag index <dir> --name <name>                 # Create collection
rag list                                      # List collections
rag info <name>                               # Show details
rag delete <name>                             # Delete collection

# Search
rag search "query" --collection <name>        # Basic search
rag search "query" --top-k 10                 # Get more results
rag search "query" --min-score 0.5            # Filter by score
rag search "query" --output results.json      # Export results

# Generation (NEW!)
rag generate "question" --collection <name>   # Generate answer
rag generate "question" --stream              # Stream response
rag generate "question" --model llama2:13b    # Use different model
rag chat --collection <name>                  # Interactive chat mode
rag generate "question" --custom-template <file>  # Custom prompt

# Utilities
rag preview <file>                            # Preview chunks
rag embed-demo "text"                         # Test embeddings
rag benchmark                                 # Run benchmarks

# Advanced
rag index <dir> --algorithm hnsw              # Use HNSW
rag search "query" --ef-search 100            # Tune HNSW
rag benchmark --compare-sizes                 # Compare algorithms
```

---

**End of User Guide**

---

## LLM Generation Commands (NEW!)

The RAG system now includes complete answer generation capabilities using local LLMs via Ollama. Generate AI-powered answers with citations, chat interactively, and customize prompts—all running 100% offline.

### Prerequisites: Ollama Setup

Before using generation commands, you need to set up Ollama:

```bash
# 1. Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai for other platforms

# 2. Start Ollama server (in a separate terminal)
ollama serve

# 3. Download a model (one-time, ~4GB)
ollama pull llama2:7b

# Verify it's working
ollama list
```

**Important**: Keep `ollama serve` running in a terminal while using generation commands.

---

### `rag generate` - Generate Answers with Citations

Generate AI-powered answers from your collection with automatic citations and confidence scoring.

#### Basic Usage

```bash
rag generate "your question" --collection <collection_name>
```

#### Examples

```bash
# Basic answer generation
rag generate "How does HNSW work?" --collection research_papers

# Stream responses word-by-word (recommended)
rag generate "Explain vector databases" --collection my_docs --stream

# Retrieve more context for better answers
rag generate "What is semantic search?" --collection docs --top-k 10

# Use different prompt templates
rag generate "Summarize the key points" --collection notes --template summarize

# Use a different model
rag generate "Explain quantum computing" --collection physics --model llama2:13b

# Adjust creativity (temperature)
rag generate "Write a creative explanation" --collection docs --temperature 1.2

# Use custom prompt template
rag generate "Explain this concept" \
  --collection my_docs \
  --custom-template templates/expert_tutor.j2
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `query` | string | *required* | Question to answer |
| `--collection` | string | *required* | Collection to search |
| `--top-k` | int | 5 | Number of chunks to retrieve |
| `--stream` | flag | false | Stream response word-by-word |
| `--template` | string | qa | Prompt template: `qa`, `summarize`, `chat` |
| `--custom-template` | path | - | Path to custom template file |
| `--model` | string | llama2:7b | LLM model to use |
| `--temperature` | float | 0.7 | Creativity (0.0=factual, 2.0=creative) |

#### Output Format

```
Answer:
────────────────────────────────────────────────────────────────
HNSW (Hierarchical Navigable Small World) is a graph-based
algorithm that provides fast approximate nearest neighbor search
[1]. It uses multiple layers of graphs, where higher layers
contain long-range connections for quick navigation [2]. This
achieves O(log n) search time while maintaining high recall [1].

Sources:
[1] hnsw_paper_chunk_3 (score: 0.92) - "HNSW uses hierarchical..."
[2] algorithms_overview_chunk_7 (score: 0.87) - "Multi-layer graph..."

Confidence: 0.89 (High)
```

#### Tips for Best Results

1. **Use streaming** (`--stream`) for better user experience with long answers
2. **Increase top-k** (10-15) for complex questions requiring more context
3. **Adjust temperature**:
   - 0.0-0.3: Factual, deterministic answers
   - 0.5-0.8: Balanced (default)
   - 1.0-2.0: Creative, varied responses
4. **Choose the right template**:
   - `qa`: Questions requiring specific answers
   - `summarize`: Condensing information
   - `chat`: Conversational, contextual responses

#### Custom Prompt Templates

Create custom templates using Jinja2 syntax:

```jinja2
# templates/expert.j2
You are an expert in {{ domain | default("the field") }}.

Context:
{{ context }}

Question: {{ query }}

Provide a detailed technical explanation with citations [1], [2]:
```

Use it:
```bash
rag generate "Explain HNSW" \
  --collection docs \
  --custom-template templates/expert.j2
```

---

### `rag chat` - Interactive Chat Mode

Start an interactive conversation with your document collection. Maintains conversation history for context-aware responses.

#### Basic Usage

```bash
rag chat --collection <collection_name>
```

#### Examples

```bash
# Start chat session
rag chat --collection research_papers

# Show confidence scores and source statistics
rag chat --collection my_docs --verbose

# Retrieve more context per query
rag chat --collection docs --top-k 10

# Use a larger, higher-quality model
rag chat --collection knowledge_base --model llama2:13b
```

#### Interactive Commands

Once in chat mode, you can use these commands:

| Command | Description |
|---------|-------------|
| Type your question | Get an AI-generated answer |
| `exit` or `quit` | Exit chat mode |
| `clear` | Reset conversation history |
| `stats` | Show session statistics |
| `help` | Show help message |

#### Session Example

```
============================================================
                Chat Mode - Collection: research_papers
============================================================

Commands:
  Type your question to get an answer
  'exit' or 'quit' - Exit chat mode
  'clear' - Reset conversation history
  'stats' - Show session statistics
  'help' - Show this help message

============================================================

You: What is HNSW?
Assistant: HNSW (Hierarchical Navigable Small World) is a graph-based
algorithm for approximate nearest neighbor search [1]. It organizes
data points in a multi-layer graph structure...

You: How does it compare to brute force?
Assistant: Based on our previous discussion about HNSW, it's significantly
faster than brute force search. While brute force has O(n) complexity,
HNSW achieves O(log n) search time [2]...

You: stats

Session Statistics:
  Session ID: 20251229_143022
  Turns: 2 (max: 10)
  Tokens: 487 (max: 4096)
  Created: 2025-12-29T14:30:22
  Updated: 2025-12-29T14:32:15

You: exit

👋 Exiting chat mode...
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--collection` | string | *required* | Collection to search |
| `--top-k` | int | 5 | Chunks to retrieve per query |
| `--model` | string | llama2:7b | LLM model to use |
| `--temperature` | float | 0.7 | Generation temperature |
| `--verbose` | flag | false | Show confidence & source stats |

#### Features

- **Conversation History**: Maintains context from previous turns
- **Auto-Pruning**: Keeps last 10 turns (configurable in config.yaml)
- **Token Management**: Prevents context window overflow
- **Colored Output**: User prompts in blue, assistant in green
- **Session Saving**: Optionally save conversations (enable in config.yaml)

#### Configuration

Edit `config.yaml` to customize chat behavior:

```yaml
generation:
  conversation:
    max_tokens: 4096      # Maximum context window
    max_turns: 10         # Maximum turns to remember
    save_sessions: true   # Save chat logs to data/sessions/
```

#### Use Cases

1. **Research Assistant**: Ask follow-up questions about documents
2. **Documentation Helper**: Interactive Q&A about codebases
3. **Learning Tool**: Conversational exploration of topics
4. **Debugging**: Discuss error messages with context
5. **Content Analysis**: Iterative refinement of queries

#### Tips

1. **Use `clear`** if the conversation goes off-track
2. **Check `stats`** to monitor context usage
3. **Enable `--verbose`** to see confidence scores
4. **Start broad**, then ask specific follow-ups
5. **Reference previous answers** - the model remembers!

---

### Answer Caching

The system automatically caches generated answers to speed up repeated queries.

#### How It Works

- **LRU Cache**: Keeps 100 most recent answers (configurable)
- **TTL**: Cached answers expire after 1 hour (configurable)
- **Automatic**: Works transparently, no action needed
- **Smart Keys**: Considers query, collection, k, model, temperature

#### Benefits

- **Instant responses** for repeated questions
- **Reduced compute** for common queries
- **Lower latency** in interactive sessions

#### Configuration

```yaml
# config.yaml
generation:
  cache:
    enabled: true    # Enable caching
    max_size: 100    # Cache up to 100 answers
    ttl: 3600        # Expire after 1 hour (seconds)
```

#### Cache Statistics

Check cache hit rate in verbose mode or logs:
```
Cache hit for query: "What is HNSW?" (saved 2.3s)
```

---

### Troubleshooting Generation Commands

#### "Ollama not found" Error

```
✗ ERROR: Ollama not found. Install from: https://ollama.ai
```

**Solution**:
```bash
# macOS
brew install ollama

# Other platforms
# Download from https://ollama.ai
```

#### "Ollama server not running" Error

```
✗ ERROR: Ollama server not running. Start with: ollama serve
```

**Solution**:
```bash
# In a separate terminal, run:
ollama serve

# Keep this terminal open while using RAG commands
```

#### "Model not downloaded" Error

```
✗ ERROR: Model 'llama2:7b' not downloaded.
Download with: ollama pull llama2:7b
```

**Solution**:
```bash
ollama pull llama2:7b

# Or use a different model you've downloaded
rag generate "query" --model llama2:7b
```

#### Slow Generation

**Solutions**:
1. **Use a smaller model**: `--model phi:2.7b`
2. **Reduce context**: `--top-k 3`
3. **Lower max_tokens** in config.yaml
4. **Enable streaming**: `--stream` (doesn't speed up, but feels faster)

#### Low Quality Answers

**Solutions**:
1. **Use a larger model**: `--model llama2:13b`
2. **Retrieve more context**: `--top-k 10`
3. **Adjust temperature**: `--temperature 0.3` (more focused)
4. **Try different template**: `--template chat`
5. **Improve your collection**: Index more relevant documents

#### Generation Timeout

```
✗ Error: Request timed out after 120 seconds
```

**Solution**:
```yaml
# Increase timeout in config.yaml
generation:
  timeout: 300  # 5 minutes
```

---

### Advanced: Custom Prompt Engineering

#### Template Variables

Available variables in Jinja2 templates:

| Variable | Type | Description |
|----------|------|-------------|
| `query` | string | User's question |
| `context` | string | Retrieved chunks with [1], [2] markers |
| `history` | list | Previous conversation turns (chat template only) |

#### Example Templates

**Concise Answers**:
```jinja2
# templates/concise.j2
Context: {{ context }}
Question: {{ query }}

Provide a brief 1-2 sentence answer with citations:
```

**Code-Focused**:
```jinja2
# templates/code.j2
You are a senior software engineer.

Documentation:
{{ context }}

Question: {{ query }}

Provide code examples and technical details with citations [1], [2]:
```

**ELI5 (Explain Like I'm 5)**:
```jinja2
# templates/eli5.j2
Context: {{ context }}
Question: {{ query }}

Explain this concept in simple terms that a 5-year-old would understand.
Use everyday analogies and avoid jargon:
```

#### Using Custom Templates

```bash
rag generate "your question" \
  --collection docs \
  --custom-template templates/your_template.j2
```

---

