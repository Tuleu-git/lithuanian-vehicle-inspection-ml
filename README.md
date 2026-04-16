# Lithuanian Vehicle Inspection — Pass/Fail Predictor

A binary classification project predicting whether a vehicle will pass its
technical inspection (*techninė apžiūra*) in Lithuania, based on real open
government data covering millions of inspections since 2015.

---

## Project Overview

**Goal:** Train a model that predicts `inspection_passed` (pass/fail) for
M1-class passenger cars undergoing a standard technical inspection.

**Dataset:** Real inspection records from Lithuania (11M+ rows total).
After filtering to M1-class technical inspections from 2019-05-01 onwards,
the working dataset contains ~4.6M rows with a near-balanced target
(51.4% pass / 48.6% fail).

**Best model:** Logistic Regression (L1 regularization, C=1)
**Test AUC:** ~0.68 | **Test Accuracy:** ~63%

---

## Approach

### 1. EDA & Data Cleaning
- Dropped identifier columns and data-leakage columns (inspection conclusion)
- Engineered `vehicle_age` from manufacture year and inspection date
- Extracted `inspection_month` as a seasonal signal
- Filtered outliers: vehicle age [0–60 years], mileage [0–1,000,000 km]
- Normalised make variants (e.g. `VOLKSWAGEN, VW` → `VW`)
- Handled `CENZŪRUOTA` (censored) values as missing

### 2. Feature Selection

| Feature | Type | Notes |
|---|---|---|
| `make` | Categorical | Car brand |
| `model` | Categorical | Car model |
| `body_type` | Categorical | 35% missing, kept & imputed |
| `fuel_type` | Categorical | Diesel, petrol, hybrid, electric |
| `inspection_station_municipality` | Categorical | Location of inspection |
| `mileage_km` | Numeric | Odometer reading |
| `vehicle_age` | Numeric | Engineered from manufacture year |
| `inspection_month` | Numeric | Month of inspection (seasonal signal) |
| `inspection_periodicity_months` | Numeric | How often vehicle must be inspected |

### 3. Preprocessing Pipeline (scikit-learn)
- **Numeric:** median imputation → StandardScaler
- **Categorical:** constant imputation (`'missing'`) → OneHotEncoder
  with `max_categories=100` to control dimensionality on high-cardinality
  columns like `model`

### 4. Model Training & Hyperparameter Tuning

Both models were tuned using `RandomizedSearchCV` (chosen over `GridSearchCV`
for speed on this dataset size):

| Model | Tuned On | CV Folds | Best Params | CV AUC |
|---|---|---|---|---|
| Logistic Regression | 1M rows | 5 | C=1 | 0.6837 |
| Random Forest | 250k rows | 3 | n_estimators=100, max_depth=20 | 0.6871 |

### 5. Model Selection
Random Forest scored marginally higher (0.003 AUC), but Logistic Regression
was selected as the final model because:
- The performance gap is within noise across CV folds
- LR trains significantly faster on millions of rows
- LR is more interpretable (inspectable coefficients)

### 6. Final Evaluation (Test Set)
The final LR model was retrained on 1M training rows and evaluated on the
held-out test set (~927k rows):

```
              precision    recall  f1-score   support
           0       0.61      0.65      0.63    450538
           1       0.65      0.61      0.63    476515
    accuracy                           0.63    927053
```

**ROC AUC: 0.68** — the model is meaningfully better than random guessing,
but the modest ceiling reflects that key drivers (maintenance history,
specific defect type, individual inspector) are not available in the dataset.

---

> **Note:** The dataset (`data.duckdb`) is not included in this repository
> due to its size. It is based on publicly available Lithuanian vehicle
> inspection open data.

---

## Requirements

```
python >= 3.10
pandas
numpy
duckdb
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy duckdb scikit-learn matplotlib seaborn
```

---

## Key Visualisations

The notebook includes:
- Correlation heatmap of numeric features vs. the target
- Distribution plots for mileage and vehicle age
- Precision / Recall vs. threshold curve
- ROC curve (AUC = 0.68)

---

## Sample Prediction

```python
my_car = {
    'make': 'VW',
    'model': 'GOLF',
    'body_type': 'AB',
    'fuel_type': 'dyzelinas',
    'mileage_km': 220_000,
    'inspection_station_municipality': 'Vilniaus m. sav.',
    'vehicle_age': 12,
    'inspection_month': 5,
    'inspection_periodicity_months': 24
}
# → Predicted probability of passing: 60.2% → PASS
```

---

## Data Source

Lithuanian vehicle inspection open data — publicly available via the
[Lithuanian Road Administration](https://www.registrucentras.lt/).
