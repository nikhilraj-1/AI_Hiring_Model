# 📄 README.md

#  AI Hiring Fit Predictor

This project predicts **candidate-job fit** using a combination of structured and unstructured (textual) features. It applies feature engineering, NLP (TF-IDF), class balancing (SMOTE), and interpretable machine learning (SHAP).

---

## 🔍 Problem Overview

Modern hiring often involves assessing large pools of candidates against job postings. Manual filtering is:
- Time-consuming
- Prone to bias
- Inconsistent

This project builds a pipeline that automates **candidate-job fit prediction** using features like:
- Candidate experience
- Salary expectations
- Skill overlap
- Job description & resume text

---

## 📁 Project Structure
├── data/
│ └── raw/ ← Original CSV (5000 rows)
│ └── processed/ ← Cleaned, split train/test data
├── models/
│ └── xgboost_pipeline.joblib
│ └── xgboost_smote_pipeline.joblib
│ └── *_label_encoder.pkl ← Label encoders for inference
├── notebooks/
│ ├── eda.ipynb
│ ├── data_preprocessing.ipynb
│ ├── gboost_smote_pipeline.py
│ ├── xgboost_smote_pipeline.py
| |-- bert.ipynb
│ └── main.ipynb ← Final inference script
├── results/
│ └── inference_results.csv
├── requirements.txt
└── README.md


---

## ⚙️ Key Concepts

### ✅ 1. **Class Imbalance**
- The `is_fit` label is highly imbalanced (less than 5% are positive).
- Imbalance hurts recall on the minority class (fit candidates).
  
✅ **Solutions Used**:
- `SMOTE`: Oversamples minority class synthetically after preprocessing.
- `scale_pos_weight`: Used in XGBoost to balance class weight during training.
- **Adjusted thresholds** during inference (e.g., 0.2 instead of 0.5) to improve recall.

---

### 🎯 2. **Stratified Sampling**
- We used `stratify=y` in `train_test_split()` to ensure that both training and test datasets maintain the same class distribution.
- Especially important in imbalanced classification tasks to prevent skewed evaluation.

---

### 🏗️ 3. **Feature Engineering**

Structured:
- `experience_gap`, `salary_diff`, `skill_overlap`, `skill_match_ratio`
- Text lengths: `job_desc_len`, `past_titles_len`

Textual:
- Combined columns: `candidate_skills`, `job_description`, etc.
- Processed with TF-IDF vectorizer (`max_features=300`)

Categorical:
- `OneHotEncoder` for `education_level`, `job_location`, etc.
- Encoders saved and reused at inference time.

---

## 🤖 Models Trained (Results 1-Fit, 0- Not fit)


| Model               | SMOTE | Precision (1) | Recall (1) | F1 (1) | Precision (0) | Recall (0) | F1 (0) | PR AUC |
|--------------------|-------|---------------|------------|--------|----------------|------------|--------|--------|
| XGBoost            | No    | 0.67          | 0.36       | 0.47   | 0.97           | 0.99       | 0.98   | 0.54   |
| XGBoost + SMOTE    | Yes   | 0.73          | 0.28       | 0.41   | 0.98           | 0.99       | 0.98   | 0.52   |
| DistilBERT         | No    | 0.00          | 0.00       | 0.00   | 0.96           | 1.00       | 0.98   | —      |


## 🚀 Running Inference

1. Prepare your new data in `data/processed/new_data.csv` with **same columns** as training data.
2. Open `main.ipynb`
3. Uncomment the model you want to use:
```python
# model = joblib.load("../models/xgboost_pipeline.joblib")

## Conclusion
The project successfully demonstrates a structured approach to predicting candidate-job fit using a mix of structured features and NLP-based textual features. Key outcomes:

XGBoost + SMOTE achieved balanced performance with:

PR AUC ~ 0.52

F1-score ~ 0.44 on minority class

Logistic Regression showed similar or slightly better generalization without SMOTE.

DistilBERT underperformed due to:

Lack of fine-tuning

Imbalanced data

Possibly limited training examples for the text classifier

Insight: For a highly imbalanced, small dataset (~5,000 rows), classical models with engineered features and TF-IDF still perform robustly. Oversampling (SMOTE) slightly improves recall but may lower precision.


