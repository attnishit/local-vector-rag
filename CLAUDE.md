# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **from-scratch implementation of a vector database and RAG (Retrieval-Augmented Generation) system** built for educational purposes. The project avoids abstractions and frameworks to demonstrate how modern vector databases actually work internally.

**Key principle**: All code is explicit and readable. The implementation prioritizes understanding over performance abstractions.

## Development Commands

### Setup
```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System
```bash
# Validate system setup
python main.py

# Preview document chunking
python main.py preview data/raw/<filename.txt>

# Test embedding generation
python main.py embed-demo "Your text here"

# Test vector search with sample data
python main.py search-demo "your query" --top-k 3

# Test full query pipeline (brute-force)
python main.py query-demo "your question"

# Test with HNSW approximate search
python main.py query-demo "your question" --algorithm hnsw --top-k 5

# Run performance benchmarks
python main.py benchmark
python main.py benchmark --dataset-size 1000 --n-queries 100
python main.py benchmark --compare-sizes --dataset-sizes 100 1000 5000
```

### Code Quality
```bash
# Format code
black src/ main.py

# Lint code
ruff check src/ main.py
```

## Architecture

### Module Organization

```
src/
├── ingestion/          # Document loading and text chunking
│   ├── loader.py       # Load documents, detect format, route to extractors
│   ├── chunker.py      # Fixed-size chunking with overlap
│   └── extractors/     # Format-specific text extraction
│       ├── pdf.py      # PDF text extraction (PyMuPDF)
│       ├── docx.py     # Word document extraction (python-docx)
│       └── markdown.py # Markdown parsing
│
├── embeddings/         # Vector embedding generation
│   ├── model.py        # Load sentence-transformers models
│   └── pipeline.py     # Batch embedding generation with normalization
│
├── vectorstore/        # Vector storage and search
│   ├── similarity.py   # Distance metrics (cosine, euclidean, dot product)
│   ├── brute_force.py  # Exact nearest neighbor (O(n) search)
│   ├── hnsw.py         # Hierarchical Navigable Small World (approximate)
│   └── persistence.py  # Save/load indexes to disk
│
├── query/              # End-to-end query pipeline
│   └── pipeline.py     # Embed query → search → format results
│
└── benchmarks/         # Performance evaluation
    ├── runner.py       # Benchmark execution
    ├── metrics.py      # Recall@k calculation
    ├── data.py         # Synthetic dataset generation
    └── report.py       # Results formatting
```

### Data Flow

1. **Ingestion** (`src/ingestion/`):
   - Load documents in various formats (.txt, .pdf, .docx, .md - see config.yaml)
   - Format detection based on file extension
   - Route to appropriate extractor (text, PDF, DOCX, Markdown)
   - Normalize whitespace → split into overlapping chunks
   - Each chunk has: `chunk_id`, `text`, `doc_id`, `start`, `end` positions, `format`
   - Chunk IDs are stable and reproducible for the same input

2. **Embedding** (`src/embeddings/`):
   - Convert text chunks to 384-dim vectors using `all-MiniLM-L6-v2`
   - Apply L2 normalization (enables cosine similarity via dot product)
   - Batch processing for efficiency
   - Deterministic: same text → same embedding

3. **Indexing** (`src/vectorstore/`):
   - **Brute-force**: Store vectors in numpy array, linear scan for exact search
   - **HNSW**: Multi-layer graph structure for approximate search
     - Layer 0 contains all vectors
     - Higher layers are progressively sparser
     - Greedy graph traversal for logarithmic-like search time

4. **Query** (`src/query/`):
   - Embed query text → search vector store → rank by similarity → return top-k

### HNSW Implementation Details

The HNSW algorithm (`src/vectorstore/hnsw.py`) implements a hierarchical graph:

- **Node assignment**: Each vector is probabilistically assigned to layers (like skip lists)
- **Layer structure**: Layer 0 has all vectors; higher layers have exponentially fewer
- **Insertion**:
  1. Find nearest neighbors at each layer (greedy search)
  2. Connect bidirectionally to M closest neighbors
  3. Prune connections to maintain M max neighbors
- **Search**:
  1. Start at top layer with entry point
  2. Greedily descend through layers
  3. Return k nearest neighbors at layer 0

**Key parameters** (configured in `config.yaml`):
- `m`: Max bidirectional links per node (default: 16) - higher = more accurate but slower
- `ef_construction`: Dynamic candidate list size during insertion (default: 200)
- `ef_search`: Dynamic candidate list size during search (default: 50) - higher = more accurate but slower

### Configuration System

All settings are in `config.yaml`:
- **Project metadata**: Name, version
- **Logging**: Level, format, file rotation
- **Paths**: Data directories for raw docs, embeddings, indexes
- **Ingestion**:
  - Chunk size (512), overlap (50)
  - Supported formats: ["txt", "pdf", "docx", "doc", "md"] (check config for current list)
  - Format-specific options (PDF OCR, DOCX headers, Markdown code blocks)
- **Embeddings**: Model name, device (cpu/cuda), batch size, dimension (384), normalization
- **Vector store**: Algorithm (brute_force/hnsw), similarity metric (cosine/euclidean/dot)
- **HNSW parameters**: m, ef_construction, ef_search
- **Query**: Default top_k, min_score threshold
- **Benchmarks**: Dataset sizes, k values, seeds

Configuration is loaded and validated by `src/config.py` which:
- Validates required sections and keys
- Creates necessary directories
- Returns typed config dictionary

## Key Design Patterns

### 1. Embedding Normalization
All embeddings are L2-normalized (`src/embeddings/pipeline.py`):
```python
# After normalization, cosine similarity = dot product
# This is faster than computing angles
```
**Important**: Query embeddings must use the same normalization as document embeddings.

### 2. Similarity Metrics
Three metrics are implemented in `src/vectorstore/similarity.py`:
- **Cosine similarity**: Measures angle between vectors (normalized: -1 to 1)
- **Euclidean distance**: Straight-line distance (lower = more similar)
- **Dot product**: Magnitude + direction (only works with normalized vectors)

**Configuration note**: `similarity_metric` in `config.yaml` determines which is used throughout the system.

### 3. Vector Store Interface
Both `BruteForceVectorStore` and `HNSWIndex` implement:
- `add(vector, metadata)` or `insert(vector, metadata)`: Add vector to index
- `search(query_vector, k)`: Return top-k similar vectors with scores
- `save(path)` / `load(path)`: Persist to disk
- `statistics()`: Memory usage, size, etc.

### 4. Metadata Propagation
Metadata (chunk_id, doc_id, text) flows through the entire pipeline:
```
Document → Chunks (+ metadata) → Embeddings (+ metadata) → Index (+ metadata) → Results (+ metadata)
```

## Adding Features

### Adding Support for New Document Formats
1. Create extractor in `src/ingestion/extractors/` (e.g., `html.py`):
   ```python
   def extract_text_from_html(filepath: Path) -> str:
       """Extract clean text from HTML file."""
       # Implementation here
       return cleaned_text
   ```
2. Update `src/ingestion/loader.py`:
   - Add format detection for new extension
   - Add extractor import and routing logic
3. Add dependencies to `requirements.txt` if needed
4. Update `config.yaml` supported_formats list
5. Test with: `python main.py preview data/raw/sample.html`

### Adding a New Similarity Metric
1. Implement function in `src/vectorstore/similarity.py`:
   ```python
   def my_metric(vec1: np.ndarray, vec2: np.ndarray) -> float:
       """Compute similarity between vectors."""
       ...
   ```
2. Update `SIMILARITY_FUNCTIONS` dict in both `brute_force.py` and `hnsw.py`
3. Add to `config.yaml` as valid option
4. Test with: `python main.py search-demo "test" --similarity my_metric`

### Adding a New Vector Store Algorithm
1. Create new file in `src/vectorstore/` (e.g., `ivf.py`)
2. Implement interface:
   - `__init__(dimension, similarity_metric, ...)`
   - `add(vector, metadata)` or `insert(vector, metadata)`
   - `search(query_vector, k) -> List[Dict]`
   - `save(path)` / `load(path)`
3. Add to `src/vectorstore/__init__.py` exports
4. Update `main.py` query-demo to support new algorithm
5. Add benchmarking support in `src/benchmarks/runner.py`

### Adding a New Document Format
1. Update `src/ingestion/loader.py` with new file handler
2. Add format to `config.yaml` supported_formats list
3. Ensure output matches existing chunk schema (chunk_id, text, doc_id, start, end, format)
4. Handle format-specific edge cases (tables, images, special characters)
5. Add sample file to `data/raw/samples/` for testing

## Testing Philosophy

The codebase emphasizes **manual validation through CLI demos** rather than automated tests:
- Each stage has a dedicated demo command (`preview`, `embed-demo`, `search-demo`, `query-demo`)
- Benchmarks provide quantitative validation (recall@k, latency)
- Exit criteria are printed after each demo

When making changes:
1. Run relevant demo command to validate behavior
2. Run benchmarks to measure performance impact
3. Check that recall@k remains high for HNSW changes

## Common Pitfalls

### 1. Document Format Issues
**Problem**: PDF extraction produces garbled text or is incomplete
**Solution**:
- Ensure PyMuPDF or pypdf is installed correctly
- For scanned PDFs, OCR may be needed (pytesseract)
- Some PDFs may be encrypted or have extraction restrictions
- Check logs for extraction warnings

### 2. Normalization Mismatch
**Problem**: Query results are poor or rankings are wrong
**Solution**: Ensure `config.yaml` `embeddings.normalize` and `query.normalize` match. If documents were embedded with `normalize: true`, queries must also use `normalize: true`.

### 3. Similarity Metric Confusion
**Problem**: Scores don't make sense
**Solution**:
- Cosine similarity: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
- Euclidean distance: 0.0 = identical, higher = more different
- Dot product: Only meaningful with normalized vectors

### 4. HNSW Parameter Tuning
**Problem**: HNSW search is slow or inaccurate
**Solution**:
- Low recall? Increase `ef_search` (default: 50)
- Slow insertion? Decrease `ef_construction` (default: 200)
- Slow search? Decrease `ef_search` or reduce `m`
- Run benchmarks to measure recall/latency tradeoff

### 5. Memory Issues
**Problem**: System runs out of memory during benchmarking
**Solution**:
- Reduce `dataset_size` in benchmark command
- Use smaller batch sizes in `config.yaml` `embeddings.batch_size`
- For HNSW, reduce `m` to decrease memory per node

### 6. File Path Issues
**Problem**: "File not found" errors
**Solution**:
- The system expects supported files (.txt, .pdf, .docx, .md) in `data/raw/`
- All paths in `config.yaml` are relative to project root
- Ensure you're running `python main.py` from the project root directory
- Check that file extensions match supported formats in config.yaml

## Performance Characteristics

### Brute-Force Vector Store
- **Insertion**: O(1) - append to array
- **Search**: O(n) - scan all vectors
- **Recall**: 100% (exact search)
- **Best for**: <10k vectors, when accuracy is critical

### HNSW Vector Store
- **Insertion**: O(log n) amortized
- **Search**: O(log n) expected
- **Recall**: 90-98% (configurable via ef_search)
- **Best for**: >10k vectors, when speed matters
- **Tradeoff**: 10-100x faster than brute-force with slight recall loss

### Benchmarking
The `benchmark` command measures:
- **Recall@k**: Percentage of correct results in top-k (vs brute-force baseline)
- **Latency**: Average query time (milliseconds)
- **Build time**: Index construction time
- **Memory**: Index size in MB

Example results (1,000 vectors, k=5):
- Brute-force: 2.5ms, 100% recall
- HNSW (ef=50): 0.8ms, 96% recall (3x speedup)
- HNSW (ef=10): 0.3ms, 88% recall (8x speedup)

## Stage Progression

The project was built in stages (documented in `main.py` comments):
1. ✅ Project skeleton & configuration
2. ✅ Document ingestion & chunking
3. ✅ Local embedding pipeline
4. ✅ Brute-force vector store (exact search)
5. ✅ Index persistence (save/load)
6. ✅ HNSW data structures
7. ✅ HNSW insertion logic
8. ✅ HNSW search algorithm
9. ✅ Query pipeline
10. ⏭️ LLM integration (future)
11. ✅ Evaluation & benchmarking

Each stage validates its exit criteria via demo commands.
