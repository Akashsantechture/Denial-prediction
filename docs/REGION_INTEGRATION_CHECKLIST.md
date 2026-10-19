# Region Integration Checklist

This checklist guides the integration of multi-region support into the existing codebase.

## ✅ Phase 0: Infrastructure Setup (COMPLETE)

- [x] Create region-specific directories
- [x] Create `ai_analyst/region_config.py` module
- [x] Copy UAE files to `uae/` subdirectories
- [x] Create KSA placeholder directories
- [x] Create comprehensive documentation
- [x] Verify existing code still works

**Status**: ✅ Complete - no existing code was modified

---

## 🔄 Phase 1: Backend Integration (TODO)

### 1.1 Update RAG Components

**File**: `ai_analyst/rag/retriever.py`

```python
# Add region parameter to Retriever class
from ai_analyst.region_config import Region, get_region_config

class Retriever:
    def __init__(self, region: Region = Region.UAE):
        self.region = region
        self.config = get_region_config(region)
        # Use config.get_chroma_store_path() instead of hardcoded path
```

**File**: `ai_analyst/rag/rule_store.py`

```python
# Update ChromaDB client initialization to accept region
def get_chroma_client(region: Region = Region.UAE):
    config = get_region_config(region)
    return chromadb.PersistentClient(
        path=str(config.get_chroma_store_path())
    )
```

**File**: `ai_analyst/rag/ingest.py`

```python
# Add CLI argument for region selection
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='uae',
                       choices=['uae', 'ksa'])
    args = parser.parse_args()
    
    region = Region(args.region)
    result = run_ingest(region)
```

### 1.2 Update Analyst Service

**File**: `ai_analyst/analyst_service.py`

```python
# Add region parameter to AnalystService
class AnalystService:
    def __init__(self, region: Region = Region.UAE):
        self.region = region
        self.config = get_region_config(region)
        self.retriever = Retriever(region=region)
```

### 1.3 Update API Endpoints

**File**: `ai_analyst/main.py`

```python
from fastapi import Header
from ai_analyst.region_config import Region

@app.post("/analyst/chat")
async def chat(
    message: ChatMessage,
    session_id: str,
    x_region: str = Header(default="uae", alias="X-Region")
):
    region = Region(x_region)
    service = AnalystService(region=region)
    # ... rest of endpoint logic
```

**Checklist**:
- [ ] Update `Retriever.__init__()` to accept region
- [ ] Update `rule_store.py` to use region config
- [ ] Add `--region` flag to `ingest.py`
- [ ] Update `AnalystService` to accept region
- [ ] Add `X-Region` header to API endpoints
- [ ] Test with both `uae` and `ksa` regions

---

## 🎨 Phase 2: Frontend Integration (TODO)

### 2.1 Add Region Selector Component

**File**: `frontend/js/region-selector.js` (NEW)

```javascript
class RegionManager {
    constructor() {
        this.regions = {
            uae: { name: 'UAE (CHI)', flag: '🇦🇪', currency: 'AED' },
            ksa: { name: 'KSA (CCHI)', flag: '🇸🇦', currency: 'SAR' }
        };
        this.currentRegion = localStorage.getItem('region') || 'uae';
        this.init();
    }
    
    init() {
        this.renderSelector();
        this.attachEventListeners();
    }
    
    renderSelector() {
        const selector = document.getElementById('region-selector');
        if (!selector) return;
        
        Object.entries(this.regions).forEach(([code, info]) => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = `${info.flag} ${info.name}`;
            if (code === this.currentRegion) option.selected = true;
            selector.appendChild(option);
        });
    }
    
    switchRegion(newRegion) {
        if (this.currentRegion === newRegion) return;
        
        this.currentRegion = newRegion;
        localStorage.setItem('region', newRegion);
        
        // Reload page to fetch new region data
        window.location.reload();
    }
    
    getRegion() {
        return this.currentRegion;
    }
}

// Global instance
window.regionManager = new RegionManager();
```

### 2.2 Update API Client

**File**: `frontend/js/api.js` (UPDATE)

```javascript
// Add region header to all API calls
async function fetchAPI(endpoint, options = {}) {
    const region = window.regionManager.getRegion();
    
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'X-Region': region,
            ...options.headers
        }
    };
    
    return fetch(endpoint, { ...options, ...defaultOptions });
}
```

### 2.3 Add Region Selector to UI

**File**: `frontend/index.html` (UPDATE)

```html
<!-- Add to navigation bar -->
<div class="region-selector-container">
    <label for="region-selector">Region:</label>
    <select id="region-selector" class="form-control">
        <!-- Options populated by region-selector.js -->
    </select>
</div>

<!-- Include the script -->
<script src="js/region-selector.js"></script>
```

**Checklist**:
- [ ] Create `region-selector.js` component
- [ ] Update API client to include `X-Region` header
- [ ] Add region selector to UI navigation
- [ ] Store region preference in localStorage
- [ ] Update currency display based on region
- [ ] Add region indicator to report headers

---

## 🧪 Phase 3: Testing (TODO)

### 3.1 Unit Tests

**File**: `tests/test_region_config.py` (NEW)

```python
def test_get_uae_config():
    config = get_region_config(Region.UAE)
    assert config.regulator_name == "CHI"
    assert config.currency == "AED"
    assert config.is_production_ready == True

def test_get_ksa_config():
    config = get_region_config(Region.KSA)
    assert config.regulator_name == "CCHI"
    assert config.currency == "SAR"
    assert config.is_production_ready == False
```

**File**: `tests/test_retriever_region.py` (NEW)

```python
def test_retriever_uae():
    retriever = Retriever(region=Region.UAE)
    assert retriever.config.chroma_collection_name == "chi_rules"

def test_retriever_ksa():
    retriever = Retriever(region=Region.KSA)
    assert retriever.config.chroma_collection_name == "cchi_rules"
```

### 3.2 Integration Tests

- [ ] Test API endpoints with `X-Region: uae` header
- [ ] Test API endpoints with `X-Region: ksa` header
- [ ] Test region switching in frontend
- [ ] Verify UAE predictions unchanged from baseline
- [ ] Verify data isolation between regions

### 3.3 UAT

- [ ] Validate UAE flow works as before
- [ ] Validate KSA flow when models ready
- [ ] Test region toggle in UI
- [ ] Verify currency formatting
- [ ] Verify rule retrieval per region

**Checklist**:
- [ ] Write unit tests for region config
- [ ] Write tests for region-aware components
- [ ] Run integration tests
- [ ] Conduct UAT with sample data

---

## 📦 Phase 4: KSA Model Training (TODO)

### 4.1 Data Collection

- [ ] Collect CCHI claim dataset from Nphies
- [ ] Clean and validate data
- [ ] Map Nphies schema to internal format
- [ ] Create train/validation/test splits

### 4.2 Feature Engineering

- [ ] Adapt UAE feature engineering for KSA
- [ ] Handle KSA-specific fields (approval number, etc.)
- [ ] Handle Arabic text fields
- [ ] Currency conversion (SAR)

### 4.3 Model Training

- [ ] Train XGBoost denial prediction model
- [ ] Train denial reason classification model
- [ ] Train resubmission probability model
- [ ] Generate SHAP visualizations
- [ ] Export feature importance

### 4.4 Rule Curation

- [ ] Collect CCHI validation rules
- [ ] Create `Thynk/ksa/nphies-rag-rules.json`
- [ ] Write rich `longMessage` content
- [ ] Embed common analyst questions
- [ ] Ingest rules: `python -m ai_analyst.rag.ingest --region ksa`

### 4.5 Deployment

- [ ] Copy trained models to `models/ksa/`
- [ ] Verify ChromaDB populated in `chroma_store_ksa/`
- [ ] Update `KSA_CONFIG.is_production_ready = True`
- [ ] Enable KSA in frontend region selector
- [ ] Monitor performance metrics

---

## 🚀 Phase 5: Production Rollout (TODO)

### 5.1 Shadow Mode

- [ ] Deploy multi-region code to staging
- [ ] UAE uses new region-aware code paths
- [ ] KSA hidden from UI (not production-ready yet)
- [ ] Monitor for regressions in UAE

### 5.2 Internal Testing

- [ ] Enable KSA for internal users only
- [ ] Validate model predictions
- [ ] Gather feedback on UI/UX
- [ ] Fix any issues found

### 5.3 Production Launch

- [ ] Enable KSA region toggle in UI
- [ ] Update documentation
- [ ] Monitor performance metrics
- [ ] Announce new region support

### 5.4 Monitoring

- [ ] Track prediction accuracy per region
- [ ] Monitor API latency per region
- [ ] Track region usage distribution
- [ ] Alert on region-specific errors

---

## 📋 Rollback Plan

If issues occur after integration:

1. **Backend Issues**:
   - Revert API changes
   - Use hardcoded `Region.UAE` as default
   - Original code paths still work

2. **Frontend Issues**:
   - Hide region selector
   - Force `localStorage.setItem('region', 'uae')`
   - UI works as before

3. **Data Issues**:
   - Original UAE data untouched
   - Can delete new region directories
   - No impact on production

---

## 🎯 Success Criteria

- [ ] UAE predictions identical to pre-migration baseline
- [ ] Region switching works seamlessly in UI
- [ ] API accepts and respects `X-Region` header
- [ ] KSA models achieve >80% accuracy when trained
- [ ] No performance regression for UAE
- [ ] Documentation complete and up-to-date
- [ ] All tests passing

---

## 📞 Support Contacts

- **Architecture Questions**: See `docs/MULTI_REGION_ARCHITECTURE.md`
- **Backend Integration**: See `ai_analyst/region_config.py`
- **Frontend Integration**: See this checklist, Phase 2
- **KSA Model Training**: See `models/ksa/README.md`

---

**Last Updated**: 2026-10-19
**Status**: Phase 0 Complete, Phase 1-5 Pending
