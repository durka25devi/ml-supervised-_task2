# ======================================
# Diabetes Classification with MLflow
# ======================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import randint

# ==============================
# 1️⃣ Load and preprocess data
# ==============================
df = pd.read_csv("diabetes.csv")

# Replace invalid zeros with median values
zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_as_missing:
    df[col] = df[col].replace(0, df[col].median())

# Fix skewness
cols_to_transform = ["Pregnancies", "DiabetesPedigreeFunction"]
pt = PowerTransformer(method='yeo-johnson')
df[cols_to_transform] = pt.fit_transform(df[cols_to_transform])

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 2️⃣ Setup MLflow Experiment
# ==============================
mlflow_dir = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri("file://" + mlflow_dir)

experiment_name = "Diabetes_Models_Comparison"
exp = mlflow.get_experiment_by_name(experiment_name)

if exp is None:
    exp_id = mlflow.create_experiment(experiment_name)
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(experiment_name)

print(" Active MLflow Experiment:")
print(f"Name: {experiment_name}")
print(f"ID: {exp_id}")
print(f"Location: {mlflow_dir}")

# ==============================
# 3️⃣ Helper Function: Train + Log
# ==============================
def evaluate_and_log(model, X_test, y_test, model_name, params=None):
    """Train, evaluate, and log model to MLflow."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with mlflow.start_run(run_name=model_name, experiment_id=exp_id):
        # Log params if provided
        if params:
            mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log confusion matrix directly to MLflow
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{model_name} - Confusion Matrix")
        mlflow.log_figure(plt.gcf(), f"{model_name}_confusion_matrix.png")
        plt.close()

    print(f"\n {model_name} logged successfully!")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# ==============================
# 4️⃣ Train & Log All Models
# ==============================
models_with_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "params": {"max_iter":1000, "class_weight":"balanced"}
    },
    "Decision Tree (Tuned)": {
        "model": GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid={
                'max_depth': [3,5,7,10],
                'min_samples_leaf': [5,10,15,20],
                'class_weight':['balanced'],
                'criterion':['gini','entropy']
            },
            scoring='balanced_accuracy', cv=5, n_jobs=-1
        ),
        "params": None
    },
    "Random Forest (Tuned)": {
        "model": RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions={
                'n_estimators': randint(100, 500),
                'max_depth': [5,10,15,20],
                'min_samples_leaf': randint(1,10),
                'max_features': [1.0,'sqrt','log2'],
                'criterion': ['gini','entropy'],
                'class_weight':['balanced']
            },
            n_iter=20, scoring='balanced_accuracy', cv=5, n_jobs=-1, random_state=42
        ),
        "params": None
    },
    "KNN (Tuned)": {
        "model": GridSearchCV(
            KNeighborsClassifier(),
            param_grid={
                'n_neighbors':[3,5,7,9,11,15],
                'weights':['uniform','distance'],
                'metric':['euclidean','manhattan']
            },
            scoring='precision', cv=5, n_jobs=-1
        ),
        "params": None
    },
    "SVC (Tuned)": {
        "model": GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid={
                'C':[50,75,100,125,150],
                'gamma':[0.05,0.1,0.15,0.2,1],
                'class_weight':['balanced'],
                'kernel':['rbf']
            },
            scoring='f1', cv=5, n_jobs=-1
        ),
        "params": None
    },
    "XGBoost (Tuned)": {
        "model": RandomizedSearchCV(
            XGBClassifier(eval_metric='logloss', random_state=42),
            param_distributions={
                'max_depth': randint(3,10),
                'n_estimators': randint(100,400),
                'learning_rate':[0.01,0.05,0.1,0.2],
                'gamma':[0,0.1,0.5,1],
                'subsample':[0.6,0.8,1.0],
                'scale_pos_weight':[1,2,5,10]
            },
            n_iter=20, scoring='f1', cv=5, n_jobs=-1, random_state=42
        ),
        "params": None
    }
}

for name, info in models_with_params.items():
    model = info["model"]
    # Fit GridSearch/RandomizedSearch first
    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        model.fit(X_train_scaled, y_train)
        model_to_log = model.best_estimator_
        params = model.best_params_
    else:
        model.fit(X_train_scaled, y_train)
        model_to_log = model
        params = info["params"]

    evaluate_and_log(model_to_log, X_test_scaled, y_test, name, params)

print("\n All models have been trained and logged to MLflow successfully!")
print(f" Open MLflow UI with:\n   mlflow ui --backend-store-uri file://{mlflow_dir}")