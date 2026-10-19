# Thynk Rules Knowledge Base

This directory contains healthcare claim validation rules organized by region.

## Directory Structure

```
Thynk/
├── uae/                           # UAE (CHI) rules
│   ├── thynk-claimlookup.json     # Rules extracted from real CHI claim responses
│   └── thynk-rag-rules.json       # Curated CHI rule knowledge base for RAG
├── ksa/                           # KSA (CCHI/Nphies) rules
│   ├── nphies-claimlookup.json    # Rules extracted from Nphies claim responses
│   └── nphies-rag-rules.json      # Curated CCHI/Nphies rule knowledge base for RAG
└── README.md                      # This file
```

## Rule Sources

### UAE (CHI - Council of Health Insurance)
- **Regulator**: CHI Abu Dhabi
- **Claims System**: Thynk
- **Rule Files**: `thynk-*.json`
- **Status**: Active production rules (29+ rules in RAG file)

### KSA (CCHI - Council of Cooperative Health Insurance)
- **Regulator**: CCHI Saudi Arabia
- **Claims System**: Nphies (National Platform for Health Information Exchange Services)
- **Rule Files**: `nphies-*.json`
- **Status**: Placeholder - to be populated during KSA model training

## File Schemas

### Claim Lookup Files
Rules extracted from actual claim API responses, deduplicated by `ruleId` across all claims.

### RAG Rules Files
Standalone rule knowledge base optimized for semantic search and LLM reasoning. These files should be manually curated with:
- Rich `longMessage` content for semantic search
- Common analyst questions embedded in the text
- Clinical examples and fix instructions

## Usage

The RAG ingest pipeline (`ai_analyst/rag/ingest.py`) automatically loads rules from the region-specific directory based on configuration.
