# RAG CLI Quick Reference

**Version**: 0.1.0 | **Command**: `rag`

---

## Installation

```bash
cd /path/to/rag
pip install -e .
rag --version
```

---

## Essential Commands

```bash
# Setup validation
rag                                           # Check system status

# Collection management
rag index <directory> --name <collection>     # Create new collection
rag list                                      # List all collections
rag info <collection>                         # Show collection details
rag delete <collection>                       # Delete collection

# Search
rag search "query text" --collection <name>   # Basic search
rag search "query" --top-k 10                 # More results
rag search "query" --min-score 0.5            # Filter by score
```

---

## Common Workflows

### Index and Search Documents

```bash
# 1. Index documents
rag index ~/Documents/notes --name personal_notes

# 2. Search
rag search "machine learning" --collection personal_notes --top-k 5

# 3. View details
rag info personal_notes
```

### Export Search Results

```bash
rag search "neural networks" --collection research --output results.json
```

### Performance Testing

```bash
# Quick test
rag benchmark

# Large-scale comparison
rag benchmark --compare-sizes --dataset-sizes 1000 5000 10000
```

---

## Key Options

| Command | Option | Description |
|---------|--------|-------------|
| `index` | `--name` | Collection name |
| `index` | `--algorithm` | `hnsw` or `brute_force` |
| `search` | `--collection` | Collection to search |
| `search` | `--top-k` | Results to return (default: 5) |
| `search` | `--min-score` | Score threshold (0.0-1.0) |
| `search` | `--ef-search` | HNSW accuracy (higher = slower) |
| `search` | `--output` | Export to JSON file |

---

## Examples

```bash
# Create HNSW collection
rag index ~/Documents --name docs --algorithm hnsw

# High-accuracy search
rag search "transformers" --collection docs --ef-search 100

# Filter low-quality results
rag search "AI" --collection docs --min-score 0.3

# Preview before indexing
rag preview document.pdf --num-chunks 5

# Delete old collection
rag delete old_collection --force
```

---

## Help & Documentation

```bash
rag --help                  # General help
rag search --help           # Command-specific help
```

📖 **Full Documentation**: [USER_GUIDE.md](USER_GUIDE.md)

---

## Supported File Formats

- `.txt` - Plain text
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.md` - Markdown files

---

## Tips

✅ Use `hnsw` for large collections (>1000 docs)
✅ Start with `--top-k 5`, adjust as needed
✅ Use `--min-score 0.3` to filter noise
✅ Preview documents before indexing
✅ Export results with `--output` for analysis

---

**Need help?** See the [full user guide](USER_GUIDE.md) or run `rag --help`
