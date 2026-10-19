# RAG (Retrieval-Augmented Generation) Module

This directory contains the RAG system for healthcare claim rule retrieval, organized by region.

## Directory Structure

```
ai_analyst/rag/
├── chroma_store_uae/           # UAE (CHI) vector database
│   └── (ChromaDB files)
├── chroma_store_ksa/           # KSA (CCHI) vector database [EMPTY]
│   └── (to be populated)
├── embedder.py                 # Sentence embedding service
├── retriever.py                # Semantic search retriever
├── rule_store.py               # ChromaDB operations
├── ingest.py                   # Rule ingestion pipeline
└── README.md                   # This file
```

## Region-Specific RAG Collections

### UAE Collection (Active)
- **ChromaDB Path**: `chroma_store_uae/`
- **Collection Name**: `chi_rules`
- **Rule Count**: 29+ curated CHI rules
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Status**: Production ready

### KSA Collection (Placeholder)
- **ChromaDB Path**: `chroma_store_ksa/`
- **Collection Name**: `cchi_rules`
- **Rule Count**: 0 (to be ingested)
- **Status**: Awaiting CCHI rule curation

## Components

### `embedder.py`
Converts text to semantic embeddings using sentence-transformers.

### `retriever.py`
Searches the vector store for relevant rules given an analyst query.

### `rule_store.py`
Low-level ChromaDB operations (upsert, query, count).

### `ingest.py`
Loads rules from JSON files, embeds them, and stores in ChromaDB.

**Sources**:
- `Thynk/{region}/thynk-claimlookup.json` - Rules from claim responses
- `Thynk/{region}/thynk-rag-rules.json` - Curated rule knowledge base

## Usage

### Ingest Rules (Region-Aware)
```bash
# Ingest UAE rules (default)
python -m ai_analyst.rag.ingest --region uae

# Ingest KSA rules (when ready)
python -m ai_analyst.rag.ingest --region ksa
```

### Search Rules
```python
from ai_analyst.rag.retriever import Retriever
from ai_analyst.region_config import Region

retriever = Retriever(region=Region.UAE)
results = retriever.search("medical necessity for lipid panel", top_k=5)
```

## Migration Note

The existing `chroma_store/` directory contains the original UAE rules. This remains untouched. New region-aware code references `chroma_store_uae/` instead.

## Next Steps for KSA

1. **Collect CCHI Rules**: Gather Nphies validation rules from:
   - CCHI regulatory documentation
   - Nphies API responses
   - Historical denial patterns

2. **Curate KSA RAG File**: Create `Thynk/ksa/nphies-rag-rules.json` with:
   - Rich Arabic + English rule descriptions
   - Common denial scenarios
   - Fix instructions specific to Nphies

3. **Ingest KSA Rules**: Run ingestion pipeline to populate `chroma_store_ksa/`

4. **Test Search**: Validate semantic search with KSA-specific queries
