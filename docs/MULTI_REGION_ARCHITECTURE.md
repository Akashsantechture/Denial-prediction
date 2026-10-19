# Multi-Region Architecture Guide

This document describes the multi-region architecture implemented to support both UAE (CHI) and KSA (CCHI/Nphies) healthcare claim denial prediction.

## Architecture Overview

The system uses a **single codebase with region-aware routing** approach:
- Shared prediction engine, RAG pipeline, and frontend framework
- Region-specific models, rule databases, and configurations
- Runtime region selection via frontend toggle or API headers

## Directory Structure

```
denial-prediction/
│
├── ai_analyst/
│   ├── region_config.py           # 🆕 Region configuration registry
│   ├── rag/
│   │   ├── chroma_store_uae/      # 🆕 UAE ChromaDB vector store
│   │   ├── chroma_store_ksa/      # 🆕 KSA ChromaDB vector store
│   │   ├── embedder.py            # Shared embedding service
│   │   ├── retriever.py           # Region-aware retrieval
│   │   └── ingest.py              # Region-aware ingestion
│   └── ...
│
├── models/
│   ├── uae/                       # 🆕 UAE model artifacts
│   │   ├── xgboost/
│   │   ├── denial_reason/
│   │   └── resubmission/
│   ├── ksa/                       # 🆕 KSA model artifacts (placeholders)
│   │   ├── xgboost/
│   │   ├── denial_reason/
│   │   └── resubmission/
│   └── README.md                  # 🆕 Models documentation
│
├── Thynk/
│   ├── uae/                       # 🆕 UAE rule files
│   │   ├── thynk-claimlookup.json
│   │   └── thynk-rag-rules.json   # 29 CHI rules
│   ├── ksa/                       # 🆕 KSA rule files
│   │   ├── nphies-claimlookup.json
│   │   └── nphies-rag-rules.json  # Placeholder
│   └── README.md                  # 🆕 Rules documentation
│
└── frontend/
    └── js/
        └── region-selector.js     # 🔜 Region toggle component (to be added)
```

## Region Configuration

### Supported Regions

| Region | Code | Regulator | Claim System | Currency | Status |
|--------|------|-----------|--------------|----------|--------|
| UAE    | `uae` | CHI Abu Dhabi | Thynk | AED (د.إ) | ✅ Production |
| KSA    | `ksa` | CCHI | Nphies | SAR (ر.س) | 🚧 In Training |

### Configuration API

```python
from ai_analyst.region_config import Region, get_region_config

# Get UAE configuration
uae_config = get_region_config(Region.UAE)
print(uae_config.regulator_name)      # "CHI"
print(uae_config.model_version)       # "v3"
print(uae_config.chroma_collection_name)  # "chi_rules"

# Get KSA configuration
ksa_config = get_region_config(Region.KSA)
print(ksa_config.regulator_name)      # "CCHI"
print(ksa_config.claim_system_name)   # "Nphies"
print(ksa_config.is_production_ready) # False
```

## Region-Specific Components

### 1. Models
Each region has its own trained models:
- **UAE**: Trained on CHI claims from Thynk system (v3, production-ready)
- **KSA**: To be trained on CCHI claims from Nphies system (v1, in development)

### 2. RAG Knowledge Base
Separate ChromaDB collections per region:
- **UAE**: `chroma_store_uae/` with 29+ CHI rules
- **KSA**: `chroma_store_ksa/` (empty, awaiting rule curation)

### 3. Rule Sources
Region-specific validation rule files:
- **UAE**: `Thynk/uae/thynk-*.json` (CHI rules)
- **KSA**: `Thynk/ksa/nphies-*.json` (CCHI rules, to be populated)

## Migration Path

### Phase 1: Structure Setup ✅
- Created region-specific directories
- Copied existing UAE files to `uae/` subdirectories
- Created placeholder KSA directories
- Added `region_config.py` module

### Phase 2: Backend Integration 🔜
- Update RAG retriever to accept region parameter
- Update model loading to use region config
- Add region header to API endpoints
- Update ingest script to support region flag

### Phase 3: Frontend Integration 🔜
- Add region selector dropdown in UI
- Store selected region in localStorage
- Pass region in API headers
- Update UI labels (currency, regulator name)

### Phase 4: KSA Model Training 🔜
- Collect CCHI/Nphies claim data
- Train denial prediction models
- Curate CCHI rule knowledge base
- Ingest KSA rules into ChromaDB

## Backward Compatibility

### Existing Code
All existing code continues to work without changes:
- Legacy model paths (`models/xgboost/`) remain in place
- Original ChromaDB store (`chroma_store/`) unchanged
- Existing Thynk files at root level preserved

### Default Behavior
- Default region: **UAE**
- If no region specified, system uses UAE configuration
- Existing deployments work as-is with UAE data

## Benefits

### For Development
- **90% code reuse** between regions
- Single CI/CD pipeline
- Unified bug fixes and improvements
- Easy A/B testing between regions

### For Users
- **Consistent UX** across regions
- Switch regions with a single click
- Power users can compare rules between markets

### For Data Science
- Transfer learning opportunities
- Unified experiment tracking
- Compare model performance across regions

### For MLOps
- Single monitoring dashboard
- Shared infrastructure
- Lower operational costs

## Next Steps

### Immediate (No Code Changes)
- ✅ Directory structure created
- ✅ UAE files copied to region folders
- ✅ KSA placeholders created
- ✅ Documentation added

### Short Term (Backend Integration)
- [ ] Update `retriever.py` to accept region parameter
- [ ] Update `ingest.py` to support `--region` flag
- [ ] Add region header to API endpoints
- [ ] Update model loading logic

### Medium Term (Frontend Integration)
- [ ] Create region selector component
- [ ] Update API client to pass region
- [ ] Update currency/date formatting
- [ ] Add region-specific help text

### Long Term (KSA Development)
- [ ] Collect CCHI claim dataset
- [ ] Train KSA denial prediction models
- [ ] Curate CCHI validation rules
- [ ] Deploy KSA models to production

## Testing Strategy

### Unit Tests
- Test region config API
- Test model loading per region
- Test RAG retrieval per region

### Integration Tests
- Test API with region headers
- Test frontend region switching
- Test data isolation between regions

### UAT
- Validate UAE predictions unchanged
- Validate KSA predictions when ready
- Validate region switching in UI

## Rollout Plan

### Phase 1: Shadow Mode
- Region infrastructure live
- UAE uses new region-aware code
- KSA hidden (not production-ready)

### Phase 2: Internal Testing
- Enable KSA for internal users
- Validate model predictions
- Gather feedback

### Phase 3: Production Launch
- Enable KSA region toggle in UI
- Monitor performance metrics
- Gradual rollout to customers

## Support

For questions about multi-region architecture:
- Architecture decisions: See this document
- Region config API: See `ai_analyst/region_config.py`
- Model organization: See `models/README.md`
- Rule organization: See `Thynk/README.md`
- RAG architecture: See `ai_analyst/rag/README.md`
