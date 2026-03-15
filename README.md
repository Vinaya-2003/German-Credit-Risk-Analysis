# 💳 German Credit Risk Analysis & Classification

A Machine Learning project that predicts credit risk (Good/Bad) for loan applicants using multiple classifiers, hyperparameter tuning via GridSearchCV, and a deployed Tkinter GUI for real-time prediction.

---

## 📌 Project Overview

| Property | Details |
|---|---|
| **Problem Type** | Binary Classification |
| **Best Model** | Random Forest (GridSearchCV tuned) |
| **Dataset** | German Credit Dataset — 1,000 records, 11 features |
| **Target** | Risk: Good (1) / Bad (0) |
| **Best Test Accuracy** | **75.5%** |
| **Cross-Val Score** | **74.8%** |
| **Deployment** | Tkinter GUI |

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Applicant age |
| Sex | Categorical | male / female |
| Job | Numeric | 0 (unskilled) to 3 (highly skilled) |
| Housing | Categorical | own / rent / free |
| Saving accounts | Categorical | little / moderate / quite rich / rich |
| Checking account | Categorical | little / moderate / rich |
| Credit amount | Numeric | Loan amount in DM |
| Duration | Numeric | Loan duration in months |
| Purpose | Categorical | car / education / furniture / radio/TV / business etc. |
| Risk | Target | good / bad |

- **Missing values:** Saving accounts (183), Checking account (394) — filled with 'Unknown'

---

## 🔄 Pipeline

```
Raw CSV (1,000 records)
        │
        ▼
  EDA: Countplots, Pairplot, Correlation Heatmap
  GroupBy analysis by Sex, Risk, Purpose
        │
        ▼
  Preprocessing
  ├── Fill nulls with 'Unknown'
  ├── One-Hot Encoding → Purpose column (get_dummies)
  ├── Label Encoding → Sex, Housing, Saving accounts,
  │                    Checking account, Risk
  └── StandardScaler → all features
        │
        ▼
  Train/Test Split (80/20 | random_state=100)
        │
        ├── Random Forest → GridSearchCV tuning
        ├── SVM → GridSearchCV tuning
        ├── Logistic Regression
        └── AdaBoost
        │
        ▼
  PCA (5 components) → applied to Logistic Regression
        │
        ▼
  Evaluate all models:
  Accuracy · Confusion Matrix · Classification Report
  Precision · Recall · F1-Score · Cross-Validation Score
        │
        ▼
  Tkinter GUI Deployment
```

---

## 📈 Results

### Model Comparison

| Model | Test Accuracy | Cross-Val Score |
|---|---|---|
| **Random Forest** | **75.5%** | **74.8%** |
| AdaBoost | 75.0% | 71.7% |
| Logistic Regression | 70.0% | 70.1% |
| SVM (poly kernel) | 67.5% | 70.8% |

### Best Model — Random Forest (GridSearchCV)
**Best Parameters:** `max_depth=10`, `n_estimators=100`, `n_jobs=1`, `random_state=10`

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Bad (0) | 0.59 | 0.41 | 0.48 | 56 |
| Good (1) | 0.80 | 0.89 | **0.84** | 144 |
| **Weighted Avg** | 0.74 | 0.76 | **0.74** | 200 |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python | Core language |
| Pandas, NumPy | Data handling |
| Scikit-learn | ML models, GridSearchCV, PCA, metrics |
| Matplotlib, Seaborn | EDA and visualisation |
| Tkinter | GUI deployment |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Vinaya-2003/German-Credit-Risk-Analysis.git
cd German-Credit-Risk-Analysis
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Add dataset
Place `german_credit_data.csv` in the project folder.

### 4. Run the notebook
```bash
jupyter notebook CAPSTONE_PROJECT__German_Credit_Risk_Analysis_.ipynb
```

### 5. Launch Tkinter GUI
Run the final cells of the notebook to open the prediction GUI window.

---

## 💡 Key Learnings

- **GridSearchCV** automates hyperparameter tuning across a parameter grid using cross-validated scoring — more reliable than manual tuning.
- **Random Forest outperforms** SVM and Logistic Regression here because it handles mixed feature types (numeric + encoded categorical) well and is robust to outliers.
- **Class imbalance** (700 Good vs 300 Bad) skews all models toward predicting "Good" — the low Recall for Bad (0.41) is a direct consequence.
- **PCA** reduced 17 features to 5 components with no accuracy drop for Logistic Regression — useful for speeding up training on larger datasets.
- **False Negatives** (predicting Good when Bad) are riskier than False Positives in credit risk — a business-aware threshold adjustment would improve real-world utility.

---

## 🔮 Future Improvements

- [ ] Address class imbalance with SMOTE or class_weight='balanced'
- [ ] Add ROC-AUC curve for threshold analysis
- [ ] Try XGBoost or LightGBM for potentially higher accuracy
- [ ] Deploy as a Flask or Streamlit web app instead of Tkinter
- [ ] Add SHAP values for model explainability

---

## 📁 Project Structure

```
German-Credit-Risk-Analysis/
│
├── CAPSTONE_PROJECT__German_Credit_Risk_Analysis_.ipynb
├── german_credit_data.csv
└── README.md
```

---

## 👩‍💻 Author

**Vinaya K**
AI & Data Science Enthusiast | Edure Learning, Cochin
[LinkedIn](https://www.linkedin.com/in/vinaya-jayadas-jun25) · [GitHub](https://github.com/Vinaya-2003)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
