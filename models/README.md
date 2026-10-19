# Models Directory

This directory contains trained machine learning models organized by healthcare region.

## Directory Structure

```
models/
├── uae/                         # UAE (CHI) models
│   ├── denial_reason/           # Denial reason classification
│   ├── resubmission/            # Resubmission probability
│   └── xgboost/                 # Primary denial prediction
├── ksa/                         # KSA (CCHI/Nphies) models [TO BE TRAINED]
│   ├── denial_reason/           # Placeholder
│   ├── resubmission/            # Placeholder
│   └── xgboost/                 # Placeholder
└── README.md                    # This file
```

## Region-Specific Models

### UAE Models (Production Ready)
- **Status**: Active in production
- **Version**: v3
- **Training Data**: CHI claims from Thynk system
- **Models**:
  - `xgboost/` - Primary denial prediction (XGBoost classifier)
  - `denial_reason/` - Denial reason multi-class classification
  - `resubmission/` - Resubmission probability regression

### KSA Models (In Training)
- **Status**: Placeholder - models to be trained
- **Version**: v1 (planned)
- **Training Data**: CCHI claims from Nphies system (to be collected)
- **Models**: Same architecture as UAE models

## Model Files

Each model directory contains:
- `model.joblib` - Trained model artifact
- `target_encoder.joblib` - Label encoder for predictions
- `label_encoder.joblib` - Feature label encoders
- `metrics.json` - Model performance metrics
- `*.png` - SHAP visualizations and feature importance plots
- `*.csv` - Feature analysis exports

## Usage

Models are loaded via the region-aware configuration:

```python
from ai_analyst.region_config import Region, get_region_config

config = get_region_config(Region.UAE)
model_path = config.get_model_path("xgboost/model.joblib")
```

## Migration Note

The existing top-level model directories (`denial_reason/`, `resubmission/`, `xgboost/`) remain in place as the legacy UAE model location. New region-aware code should reference `models/uae/` instead.

**Next Steps for KSA**:
1. Collect CCHI/Nphies claim data
2. Adapt feature engineering for Nphies schema
3. Train models using KSA-specific data
4. Validate against CCHI regulatory requirements
5. Deploy to `models/ksa/` directory
