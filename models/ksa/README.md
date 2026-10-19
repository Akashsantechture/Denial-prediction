# KSA (CCHI/Nphies) Models - Placeholder

This directory is reserved for KSA-specific healthcare claim denial prediction models.

## Status: In Development

Models for the Kingdom of Saudi Arabia (CCHI/Nphies) are currently in the training and validation phase.

## Planned Model Structure

```
models/ksa/
├── xgboost/                     # Primary denial prediction
│   ├── model.joblib
│   ├── target_encoder.joblib
│   └── metrics.json
├── denial_reason/               # Denial reason classification
│   ├── model.joblib
│   ├── label_encoder.joblib
│   └── metrics.json
└── resubmission/                # Resubmission probability
    ├── model.joblib
    └── metrics.json
```

## Training Data Requirements

- **Source**: Nphies claim submissions and responses
- **Regulator**: CCHI (Council of Cooperative Health Insurance)
- **Required Fields**:
  - Approval number
  - Policy number
  - Member ID
  - Primary diagnosis (ICD-10)
  - Clinician license
  - Activity codes (CPT/local codes)
  - Claim amounts (SAR)

## Feature Engineering Considerations

### Differences from UAE (CHI) Models:
1. **Currency**: SAR instead of AED
2. **Regulatory Framework**: CCHI rules vs CHI rules
3. **Claim System**: Nphies API schema vs Thynk schema
4. **Prior Authorization**: More common in KSA
5. **Network Type**: CCHI network requirements differ
6. **Arabic Text**: More prevalent in KSA claims

### Shared Features:
- Patient demographics
- Diagnosis codes (ICD-10)
- Activity codes (CPT)
- Claim amounts
- Secondary diagnoses
- Clinician specialty

## Deployment Timeline

1. **Phase 1**: Data collection and cleaning
2. **Phase 2**: Feature engineering and EDA
3. **Phase 3**: Model training and validation
4. **Phase 4**: SHAP analysis and interpretability
5. **Phase 5**: Production deployment

## Contact

For questions about KSA model development, contact the data science team.
