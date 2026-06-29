# ML Train/Test Split Project

A complete machine learning pipeline demonstrating the **train/test split** concept using scikit-learn.

## 📋 What It Does

| Step | Description |
|------|-------------|
| 1 | Loads the Iris dataset (150 samples, 4 features, 3 classes) |
| 2 | Splits data → 90% train / 10% test using `train_test_split` |
| 3 | Scales features with `StandardScaler` (fit on train only) |
| 4 | Trains a `LogisticRegression` model on training data |
| 5 | Evaluates on test data: Accuracy, Precision, Recall, F1-Score |
| 6 | Saves metrics to `outputs/metrics.csv` |
| 7 | Saves confusion matrix + bar chart to `outputs/results.png` |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python main.py
```

## 📁 Project Structure

```
ml_train_test_project/
├── main.py            ← Full ML pipeline
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
└── outputs/           ← Auto-created on first run
    ├── metrics.csv    ← Accuracy, Precision, Recall, F1
    └── results.png    ← Confusion matrix & metrics chart
```

## 🔑 Key Concepts

### Why Split Data?
- **Training set** — model learns patterns here (90% of data)
- **Testing set** — evaluates generalization on *unseen* data (10%)

### train_test_split Parameters
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,    # 10% goes to test set
    random_state=42   # reproducible shuffle
)
```

### Evaluation Metrics
| Metric | What It Measures |
|--------|-----------------|
| Accuracy | % of all predictions that are correct |
| Precision | Of predicted positives, how many are actually positive |
| Recall | Of actual positives, how many were correctly identified |
| F1-Score | Harmonic mean of Precision & Recall |

## ⚠️ Important Notes
- `StandardScaler` is **fit only on training data** — never on test data — to prevent data leakage.
- `random_state=42` ensures the same split is produced every run (reproducibility).
