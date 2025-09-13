# Diabetes Detection & Prediction

A **Machine Learning project** to analyze health indicators and predict the likelihood of diabetes.  
The project involves **data preprocessing, exploratory data analysis (EDA), and multiple classification models**.

---

## Summary

- **Dataset:** 768 records, 9 features (PIMA Indians Diabetes dataset).  
- **Preprocessing:**
  - Fixed shifted column names.
  - Replaced impossible insulin values (`0`) with median.
  - Removed outliers using Z-Score.
  - Verified no missing or duplicate values.
- **EDA:**
  - Pregnancies, Insulin, BMI, and Family History (DPF) are most correlated with diabetes.
  - No strong multicollinearity issue (all correlations < 0.7).
  - Insulin values showed heavy outliers.
- **Modeling:**
  - Logistic Regression → **79–80% accuracy**.
  - SVC, KNN → similar performance (~80%).
  - Decision Tree, Random Forest, XGBoost → tendency to overfit.
  - Ensemble methods (Voting, Stacking) → ~80% stable accuracy.
- **Overfitting/Underfitting:**
  - Training accuracy ≈ Testing accuracy → no severe overfitting.
  - Overfitting flagged when train ≫ test accuracy.
  - Underfitting flagged when both train and test scores are low.

---

## Exploratory Data Analysis (EDA)

- **Key Insights:**
  - Pregnancies, Insulin, BMI, and Family History (DPF) are strong predictors.
  - Heavy outliers in Insulin → handled with Z-Score removal.
- **Visualizations:**
  - Pair plots, scatter plots, box plots, and bar plots.
  - Showed feature distributions and highlighted outliers.
- **Data Cleaning:**
  - Replaced invalid medical values (e.g., BMI = 0) with median.
  - Ensured dataset balance and integrity.

---

## Modeling & Results

### Without Hyperparameter Tuning
- Logistic Regression → **79%**
- SVC → **78%**
- Random Forest → **77%** (overfitting)
- KNN → **75%**
- Decision Tree → **73%** (overfitting)
- XGBoost → **75%** (overfitting)
- Voting Classifier → **79%**
- Stacking Classifier → **79%**

### With Hyperparameter Tuning
- Logistic Regression, SVC, KNN → improved to **~80%**
- Decision Tree → improved, reduced overfitting
- Voting & Stacking Classifiers → stable at **~80%**
- XGBoost → still overfitting

---

## Final Insights

- **Best Models:** Logistic Regression, SVC, KNN → **79–80% accuracy**  
- **Stable Ensembles:** Voting and Stacking → ~80%  
- **Overfitting Risks:** Random Forest, Decision Tree, XGBoost  
- **Improvement Ideas:**
  - Feature Engineering (e.g., One-Hot Encoding, StandardScaler)
  - Better handling of outliers
  - Category-wise data splits

**In short:** Logistic Regression, SVC, and KNN generalized well and serve as reliable baselines.  
Ensemble methods gave stable accuracy, while tree-based models need careful tuning to avoid overfitting.

---

## Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost  
- **Modeling Approaches:** Logistic Regression, SVC, KNN, Decision Tree, Random Forest, XGBoost, Voting, Stacking  

---
