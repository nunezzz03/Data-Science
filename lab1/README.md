# Lab 1: Baseline Models for Classification

**Course:** Data Science  
**Date:** November 2025  
**Goal:** Train baseline models on raw data to establish performance benchmarks

---

## 📁 Project Structure

```
lab1/
├── README.md                    # This file
├── run_all.sh                   # Run everything with one command
│
├── datasets/                    # Original raw datasets
│   
│
├── data/
│   ├── raw/
    │   ├── TrafficTwoMonth.csv
    │   ├── traffic_accidents.csv
    │   ├── economic_indicators_dataset_2010_2023.csv
    │   └── Combined_Flights_2022.csv
│   └── processed/              # Train/test splits (generated)
│       ├── traffic_train.csv
│       ├── traffic_test.csv
│       ├── accidents_train.csv
│       ├── accidents_test.csv
│       ├── economic_train.csv
│       ├── economic_test.csv
│       ├── flights_train.csv
│       └── flights_test.csv
│
├── models/                      # All Python scripts
│   ├── prepare_data.py         # Data preprocessing (fixes leakage)
│   ├── naive_bayes.py          # Naive Bayes models
│   ├── knn.py                  # K-Nearest Neighbors
│   ├── decision_tree.py        # Decision Trees
│   ├── logistic_regression.py  # Logistic Regression
│   ├── mlp.py                  # Multi-Layer Perceptron
│   └── summary.py              # Generate final results CSV
│
├── images/                      # Generated charts (30 files)
│   ├── traffic_nb_study.png
│   ├── traffic_knn_study.png
│   ├── traffic_dt_study.png
│   ├── ...
│   └── flights_final_comparison.png
│
├── results/                     # Final metrics
│   └── baseline_results_summary.csv
│
└── docs/                        # Documentation & analysis
    ├── FINAL_PERFORMANCE_RESULTS.md    # ⭐ Main report for evaluation
    ├── FIXES_SUMMARY.md                # What was fixed
    ├── rigorous_analysis.md            # Detailed technical analysis
    └── notes.md                        # Working notes
```

---

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)

```bash
cd /Users/goncalofrutuoso/Developer/DataScience/labs/lab1
./run_all.sh
```

### Option 2: Step-by-Step

```bash
# 1. Prepare data (clean, split, fix leakage)
python models/prepare_data.py

# 2. Run each model
python models/naive_bayes.py
python models/knn.py
python models/decision_tree.py
python models/logistic_regression.py
python models/mlp.py

# 3. Generate summary
python models/summary.py
```

**⏱ Total runtime:** ~3-5 minutes

---

## 📊 Datasets

| Dataset       | Size    | Task                         | Classes                      | Fixed Issues                        |
| ------------- | ------- | ---------------------------- | ---------------------------- | ----------------------------------- |
| **Traffic**   | 5,953   | Predict traffic situation    | 4 (low/normal/heavy/high)    | Removed `Total` column (derived)    |
| **Accidents** | 209,306 | Predict crash severity       | 2 (injury vs no-injury)      | None (clean)                        |
| **Economic**  | 501     | Predict GDP category         | 4 (negative/low/medium/high) | **Changed from predicting Country** |
| **Flights**   | 394,700 | Predict arrival delay >15min | 2 (on-time vs delayed)       | **Removed 7 post-arrival features** |

### Key Fixes Applied:

1. **Economic Dataset:** Changed task from "predict Country" → "predict GDP Growth Category"

   - Original task was backwards (country determines economics, not vice versa)
   - New task: Classify GDP growth into meaningful categories

2. **Flights Dataset:** Removed data leakage
   - Excluded: `ArrTime`, `ArrDelayMinutes`, `ArrDelay`, `ActualElapsedTime`, `WheelsOn`, `TaxiIn`, `ArrivalDelayGroups`
   - Now uses only pre-departure information for realistic prediction

---

## 🎯 Results Summary

| Dataset   | Best Model          | Accuracy | Notes                                 |
| --------- | ------------------- | -------- | ------------------------------------- |
| Traffic   | Decision Tree       | 95.0%    | Excellent performance, clear patterns |
| Accidents | Decision Tree       | 82.8%    | **Most realistic baseline**           |
| Economic  | Logistic Regression | 37.3%    | Hard task, needs feature engineering  |
| Flights   | Decision Tree       | 94.1%    | Large dataset, no data leakage        |

**Key Findings:**

- ✅ Decision Trees dominate on raw, unscaled data
- ✅ KNN/LogReg/MLP skip large datasets (>50k rows) due to computational cost
- ✅ All models beat simple majority-class baselines
- ⚠️ Economic dataset shows need for better features
- ⚠️ Preprocessing (scaling) will unlock MLP/KNN potential in Lab 2

---

## 📈 Model Performance by Dataset

### Models Tested:

1. **Naive Bayes** - Fast, probabilistic baseline
2. **K-Nearest Neighbors (KNN)** - Distance-based, needs scaling
3. **Decision Tree** - Non-parametric, robust to raw data
4. **Logistic Regression** - Linear model, needs preprocessing
5. **Multi-Layer Perceptron (MLP)** - Neural network, needs scaling

### Computational Notes:

- **Small datasets (≤50k rows):** All 5 models complete
- **Large datasets (>50k rows):** Only Decision Tree + Naive Bayes
- **Why skip?** KNN is O(n²), LogReg/MLP converge slowly on unscaled features

---

## 📝 Key Learnings

### What This Lab Demonstrates:

1. **Data Quality Matters More Than Algorithms**

   - Fixed data leakage in Flights (was giving fake 100% accuracy)
   - Redesigned Economic task (was predicting wrong thing)

2. **Model Selection Depends on Data Characteristics**

   - Decision Trees excel on raw data (no preprocessing needed)
   - MLP/KNN need feature scaling to be competitive
   - Computational cost varies dramatically (seconds vs hours)

3. **Baselines Are Essential**

   - Always compare to "dummy classifier" (predict majority class)
   - Our models beat baselines by 12-34%

4. **Class Imbalance Affects Metrics**
   - High accuracy can be misleading on imbalanced data
   - Precision/Recall provide fuller picture

---

## 🔧 Technical Details

### Data Preprocessing Steps:

1. Drop all rows with missing values
2. Remove non-numeric columns (except target)
3. Exclude data leakage features (flights)
4. Remove derived features (traffic)
5. Create meaningful target variable (economic)
6. 70/30 train-test split with stratification

### Hyperparameter Search:

- **Naive Bayes:** Tested Gaussian, Multinomial, Bernoulli
- **KNN:** k ∈ [1,3,5,7,9,11,13,15,17,19,21,23,25], distances = [manhattan, euclidean, chebyshev]
- **Decision Tree:** criterion ∈ [entropy, gini], max_depth ∈ [2,4,6,8,10,15,20,25]
- **Logistic Regression:** penalty ∈ [l1, l2], max_iter ∈ [10,50,100,500,1000,2500,5000]
- **MLP:** learning_rate ∈ [constant, invscaling, adaptive], init_lr ∈ [0.5, 0.05, 0.005, 0.0005]

### Evaluation Metrics:

- **Accuracy:** Overall correctness
- **Precision (macro):** Quality of positive predictions
- **Recall (macro):** Coverage of actual positives

---

## 📚 Documentation

- **`docs/FINAL_PERFORMANCE_RESULTS.md`** - Complete analysis with explanations (USE THIS!)
- **`docs/FIXES_SUMMARY.md`** - Summary of what was fixed and why
- **`docs/rigorous_analysis.md`** - Detailed technical review
- **`results/baseline_results_summary.csv`** - All metrics in table format

---

## 🎓 For Evaluation Tomorrow

### Main Points to Emphasize:

1. **Critical Thinking:** Identified and fixed data leakage + wrong prediction task
2. **Realistic Results:** Accidents dataset provides honest 82.8% baseline
3. **Computational Awareness:** Understood when to skip models due to scale
4. **Clear Roadmap:** Identified preprocessing as key next step

### Don't Say:

- ❌ "We got 100% accuracy" (that was the leaky Flights data)
- ❌ "All models worked perfectly" (be honest about skips)
- ❌ "Economic performed great" (it's supposed to be hard)

### Do Say:

- ✅ "We established honest baselines with fixed data quality issues"
- ✅ "Decision Trees proved most scalable on raw data"
- ✅ "Results guide our preprocessing strategy for Lab 2"

---

## 🔄 Next Steps (Lab 2)

1. **Feature Scaling:** StandardScaler for MLP/KNN/LogReg
2. **Feature Engineering:** Economic dataset needs temporal features
3. **Class Balancing:** SMOTE or class weights for imbalanced data
4. **Hyperparameter Optimization:** Grid search with cross-validation
5. **Ensemble Methods:** Random Forest, XGBoost

**Expected improvements:** +5-15% accuracy on MLP/KNN with proper preprocessing

---

## 👥 Team

Ready to share with teammates! All code is organized, documented, and reproducible.

## 📧 Questions?

Check `docs/FINAL_PERFORMANCE_RESULTS.md` for detailed explanations of all results and decisions.

---

**Status:** ✅ Complete and ready for evaluation
**Last Updated:** November 13, 2025
