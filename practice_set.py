"""
Week 4 Practice Set - Model Evaluation & Optimization
Runs standalone (uses scikit-learn's built-in breast-cancer data, no download).

1. cross_val_score to test accuracy
2. GridSearchCV for SVM
3. Save trained model with joblib
4. Load model and predict on new data
"""
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Q1: cross-validation accuracy
base = make_pipeline(StandardScaler(), SVC())
scores = cross_val_score(base, X_train, y_train, cv=5, scoring="accuracy")
print("Q1  CV accuracies:", scores.round(3), "| mean =", round(scores.mean(), 3))

# Q2: GridSearchCV for SVM
grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": [0.1, 1, 10, 100],
     "svc__gamma": ["scale", 0.1, 0.01, 0.001],
     "svc__kernel": ["rbf", "linear"]},
    cv=5, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
print("Q2  Best params:", grid.best_params_)
print("    Best CV accuracy:", round(grid.best_score_, 3))
print("    Test accuracy:", round(grid.score(X_test, y_test), 3))

# Q3: save model
joblib.dump(grid.best_estimator_, "svm_model.joblib")
print("Q3  Saved -> svm_model.joblib")

# Q4: load model and predict on new data
model = joblib.load("svm_model.joblib")
print("Q4  Predictions for 5 new samples:", model.predict(X_test[:5]),
      "| actual:", y_test[:5])
