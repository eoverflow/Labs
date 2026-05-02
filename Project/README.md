# Clinical Outcome Prediction — Myocardial Infarction Complications

**Group:** Ahmad Balubaid, Faisal Khalili, Mohammed Tazi, Hamza Hamza

**Dataset:** [UCI Myocardial Infarction Complications](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) — 1,700 patients, 124 features, 12 complication targets

---

## Project Overview

This project applies machine learning to predict in-hospital mortality (`LET_IS`) following myocardial infarction (MI). The pipeline spans literature review, exploratory data analysis, preprocessing, adversarial robustness testing, multi-model evaluation, hyperparameter tuning, and final held-out test evaluation.

The work is structured across five phases, each in its own notebook, plus a literature review document.

---

## Repository Structure

```
├── lit_review_ML.docx                  # Literature review
├── Phase_2_EDA.ipynb                   # Phase 2: EDA + Preprocessing + Adversarial Robustness + Baseline Modeling
├── Phase_3_Modeling.ipynb              # Phase 3: 7-classifier comparison + cross-validation + model selection
├── Phase_4.ipynb                       # Phase 4: Hyperparameter tuning + feature selection + ensemble methods
├── Phase_5.ipynb                       # Phase 5: Final test evaluation + limitations analysis
└── README.md
```

> **Note:** The dataset CSV (`Myocardial infarction complications Database.csv`) is not included in this repository. Download it from [Kaggle](https://www.kaggle.com/datasets/rafatashrafjoy/myocardial-infarction-complications/data) or [UCI](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) and place it in the working directory. Each notebook falls back to a synthetic 1,700 × 124 stand-in if the file is not found.

---

## Phase Summaries

### Literature Review (`lit_review_ML.docx`)
Reviews five key studies on ML for MI complication prediction (Golovenkin et al. 2020, Soleimani et al. 2025, Diakou et al. 2022, Zhang et al. 2025, Lee et al. 2021). Synthesizes common preprocessing strategies, model choices, and persistent challenges (missing data, class imbalance, interpretability, generalizability). Concludes that tree-based ensembles (GBDT/Random Forest) consistently perform best on tabular clinical data.

### Phase 2 — EDA, Preprocessing & Baseline Modeling (`Phase_2_EDA.ipynb`)
- **EDA:** Target distribution (16% mortality), missing value heatmap, correlation matrix, IQR outlier detection, feature distributions by outcome
- **Preprocessing:** Stratified 70/15/15 train/val/test split; type-aware pipeline (median imputation + scaling for continuous; mode imputation for binary/ordinal)
- **Adversarial Robustness:** Label noise injection (15% flip rate), SMOTE oversampling, simulated concept drift
- **Baseline Models:** Logistic Regression and Random Forest evaluated on clean vs. noisy vs. drifted data

### Phase 3 — Multi-Model Comparison (`Phase_3_Modeling.ipynb`)
- Trains 7 classifiers on the SMOTE-balanced training set: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting
- Reports accuracy, precision, recall, F1, ROC-AUC on the validation set
- 5-fold stratified cross-validation on the combined train+val set (pre-SMOTE to avoid leakage)
- Composite scoring selects the top 3 models based on recall, F1, AUC-ROC, and CV stability
- **Clinical priority:** recall is weighted highest — false negatives (missed deaths) are the costliest error

### Phase 4 — Tuning, Feature Selection & Ensembles (`Phase_4.ipynb`)
- **Phase 4.1:** `RandomizedSearchCV` hyperparameter tuning for top models
- **Phase 4.2:** Additional class imbalance handling
- **Phase 4.3:** Feature selection and engineering
- **Phase 4.4:** Voting classifier and stacking ensemble

### Phase 5 — Final Evaluation (`Phase_5.ipynb`)
- **Phase 5.1:** Final held-out test set evaluation of the best model(s)
- **Phase 5.2:** Honest limitations analysis (single-centre data, synthetic fallback, concept drift assumptions)

---

## Dataset Issues (Quick Reference)

| Issue | Details |
|---|---|
| **Missing values** | ~7.6% overall; lab values (K_BLOOD, NA_BLOOD, ALT/AST/KFK_BLOOD) up to 30–40% missing |
| **Label leakage risk** | ICU treatment columns (LID_S_n, B_BLOK_S_n, GEPAR_S_n, etc.) recorded post-admission — excluded from features |
| **Class imbalance** | Mortality ~16%; rare complications (RAZRIV, DRESSLER, A_V_BLOK) are 2–5% positive |
| **Multicollinearity** | ECG rhythm flags are mutually exclusive encodings; blood pressure pairs are correlated |
| **Ordinal mishandling** | FK_STENOK, DLIT_AG, ZSN_A are clinical severity ordinals coded as integers |
| **Mixed feature types** | 111 input features span binary flags, ordinal scales, and continuous lab values |

---

## Setup

```bash
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

Run notebooks in order (2 → 3 → 4 → 5). Each notebook is self-contained and re-runs the shared setup/preprocessing block at the top.

---

## Key Results Summary

| Phase | Highlight |
|---|---|
| Phase 2 | Random Forest more robust to label noise than Logistic Regression; both degrade under concept drift |
| Phase 3 | Top 3 models selected via composite recall/F1/AUC/CV-stability score |
| Phase 4 | Ensemble (voting/stacking) improves over individual tuned models |
| Phase 5 | Final test ROC-AUC and F1 reported; limitations documented |

---

## References

Golovenkin et al. (2020). Predicting complications of myocardial infarction using machine learning.  
Soleimani et al. (2025). *Clinical Cardiology* (Wiley).  
Diakou et al. (2022). *IEEE Conference.*  
Zhang et al. (2025). *Reviews in Cardiovascular Medicine.*  
Lee et al. (2021). *Scientific Reports / PMC.*  
Full reference list in `lit_review_ML.docx`.
