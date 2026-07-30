# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

# 2. Load Parquet File from Google Drive
# Adjust path if you uploaded it inside a folder (e.g. '/content/drive/MyDrive/Denial-Prediction/denial_prediction_features.parquet')
parquet_path = '/content/drive/MyDrive/Denial-prediction/denialclaims_features.parquet'

print("Loading Parquet dataset into Colab memory...")
df = pd.read_parquet(parquet_path)
print(f"Dataset Loaded Successfully! Total Rows: {len(df):,}")

# 3. Log Transformation on Skewed Financial Columns
monetary_cols = ['activity_gross', 'gross_amount', 'net_amount']
for col in monetary_cols:
    df[f'{col}_log'] = np.log1p(df[col])

# 4. Separate Features (X) & Target (y)
X = df.drop(columns=['claim_status'] + monetary_cols)
y = df['claim_status']

# 5. Column Setup
high_card_cols = ['primary_diagnosis_code', 'activity_code', 'cpt_icd_pair']
low_card_cols = ['gender', 'nationality', 'payer_id', 'insurance_plan_tier', 'profession', 'category', 'facility_type_id', 'encounter_type']
scaled_num_cols = ['patient_age', 'billing_lag_days', 'length_of_stay', 'activity_quantity']
passed_num_cols = [
    'medical_necessity_score', 'pa_risk_score', 'coverage_score',
    'clinician_success_score', 'facility_success_score', 'pa_exceeds_1000_flag',
    'activity_gross_log', 'gross_amount_log', 'net_amount_log'
]

# 6. Build Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('high_card', TargetEncoder(smooth="auto"), high_card_cols),
        ('low_card', OneHotEncoder(handle_unknown='ignore', max_categories=30, sparse_output=False), low_card_cols),
        ('scaled_num', StandardScaler(), scaled_num_cols),
        ('passed_num', 'passthrough', passed_num_cols)
    ]
)

# 7. Stratified Train-Test Split (80/20)
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Fitting preprocessor and transforming matrices...")
X_train_transformed = preprocessor.fit_transform(X_train, y_train)
X_test_transformed = preprocessor.transform(X_test)

# 8. Train GPU-Accelerated XGBoost Model
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    device='cuda',  # Leverage T4 GPU
    random_state=42
)

print("\nTraining XGBoost Classifier on GPU...")
model.fit(X_train_transformed, y_train, eval_set=[(X_test_transformed, y_test)], verbose=50)

# 9. Model Evaluation
y_pred_prob = model.predict_proba(X_test_transformed)[:, 1]
y_pred = (y_pred_prob >= 0.50).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

print("\n" + "="*45)
print("             MODEL EVALUATION RESULT             ")
print("="*45)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"PR-AUC Score:  {pr_auc:.4f}")
print("="*45 + "\n")
print(classification_report(y_test, y_pred))