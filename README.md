#  Diabetes Prediction Model Performance Analysis

This project analyzes multiple classification algorithms on the Pima Indians Diabetes Dataset. The core focus is on **hyperparameter tuning** and implementing techniques to effectively handle **class imbalance**, prioritizing the identification of the diabetic (minority) class.

## 1. Project Goal & Metrics

The objective is to maximize the model's ability to correctly identify positive cases (Diabetes). Due to the class imbalance observed in the dataset (Outcome: 65% Non-Diabetic, 35% Diabetic), the primary evaluation metrics are:

* **Recall:** Maximizing the True Positive Rate (minimizing missed diabetic cases).
* **F1-Score:** Providing a balanced measure of Precision and Recall.

## 2. Data Preparation Pipeline

The data underwent comprehensive preprocessing:

1.  **Outlier Handling:** Outliers in all features were addressed using the **Interquartile Range (IQR) method** (clipping to the bounds).
2.  **Missing Value Imputation:** Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were treated as missing data and replaced with the **median** of their respective columns.
3.  **Feature Transformation:** Skewed features (`Pregnancies`, `DiabetesPedigreeFunction`) were transformed using the **Yeo-Johnson Power Transformer**.
4.  **Scaling:** Data was standardized using **StandardScaler** for distance-based models (KNN, SVC, Logistic Regression).
5.  **Data Split:** Data was split into 80% Training and 20% Testing using **stratified sampling** to maintain the original class ratio in both sets.

 ## 3 .Context and Composition of the Dataset

| Attribute | Detail | Value |
| :--- | :--- | :--- |
| **Source Population** | Females of Pima Indian heritage. | Age $\geq 21$ years old. |
| **Total Size** | Number of Instances (Rows) | 768 |
| **Features** | Number of Attributes (Columns) | 9 (8 predictors + 1 target) |
| **Class Imbalance** | **Non-Diabetic (Outcome = 0)** | 500 instances ($\approx 65\%$) |
| **Class Imbalance** | **Diabetic (Outcome = 1)** | 268 instances ($\approx 35\%$) |


## 4. Model Performance Summary (After Tuning)

All models were tuned using either **GridSearchCV** or **RandomizedSearchCV**, optimizing primarily for **F1-Score** or **Balanced Accuracy**.

| Model | Best Hyperparameters Found | Accuracy | Precision | **Recall** | **F1 Score** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVC (Best Balanced)** | `{'C': 50, 'gamma': 0.05, 'class_weight': 'balanced', 'kernel': 'rbf'}` | 0.662 | 0.513 | **0.759** | **0.612** |
| **XGBoost (Highest Recall)** | `{'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 484, 'scale_pos_weight': 5, 'subsample': 0.8}` | 0.448 | 0.376 | **0.870** | 0.525 |
| **Decision Tree** | `{'max_depth': 3, 'min_samples_leaf': 5, 'class_weight': 'balanced', 'criterion': 'gini'}` | 0.610 | 0.464 | 0.722 | 0.565 |
| **Random Forest** | `{'n_estimators': 312, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 8, ...}` | 0.623 | 0.462 | 0.444 | 0.453 |
| **Logistic Regression** | (Base model with `class_weight='balanced'`) | 0.597 | 0.438 | 0.519 | 0.475 |
| **KNN** | `{'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'uniform'}` | 0.617 | 0.414 | 0.222 | 0.289 |


## 5. Key Findings & Recommendations

### A. Recommended Production Model: Tuned SVC

The **Tuned SVC model** achieved the highest overall balanced performance:

* **F1-Score: 0.612**
* **Recall: 0.759** (Correctly identifying $\approx 76\%$ of diabetic patients).
* The use of `class_weight='balanced'` was highly effective in pushing the decision boundary toward the minority class.

### B. Trade-Off Analysis (XGBoost)

The XGBoost model, while achieving a Recall of **0.870** (finding almost all positive cases), suffered from severe overcompensation.

* **Confusion Matrix:** `[[ 22  78 ] [ 7  47]]`
* The model generated 78 False Positives (predicting diabetes when it wasn't present) against only 22 True Negatives, making its Precision (0.376) too low for reliable positive predictions.

### C. Future Work

1.  **Threshold Optimization:** Implement **threshold tuning** for the high-Recall XGBoost model to find a sweet spot that maximizes Precision without sacrificing too much Recall (e.g., test thresholds between 0.6 and 0.8).
2.  **Ensemble Methods:** Explore **Stacking** or **Voting Classifiers** using the top performers (SVC and Decision Tree) to potentially boost the F1-Score further.
