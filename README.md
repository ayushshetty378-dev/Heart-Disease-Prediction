# Heart Disease Prediction (Week 4 Capstone)

Predicts whether a patient is likely to have heart disease from clinical features.
Course: Machine Learning & AI. Everything is included except the dataset.

## 1. Get the dataset
Download **Heart Disease UCI** from Kaggle and put the CSV in a folder named `data/`
(e.g. `data/heart.csv`). Both common versions work: the one with a `target` column and the
one with a `num` column (`heart_disease_uci.csv`).

## 2. Run
```
pip install -r requirements.txt
python heart_disease_pipeline.py          # train, tune, evaluate, save model, make plots
python make_presentation.py               # builds Heart_Disease_Presentation.pptx from the results
python practice_set.py                    # Week 4 practice questions (needs no dataset)
```

## What the pipeline does
| Step | Details |
|---|---|
| Cleaning | binary target, drop id/duplicate rows, median / most-frequent imputation |
| Scaling | StandardScaler on numeric features, one-hot encoding on categorical ones, all inside a Pipeline |
| Models | Logistic Regression, Decision Tree |
| Validation | stratified 80/20 split, 5-fold `cross_val_score`, `GridSearchCV` (ROC AUC) |
| Metrics | accuracy, precision, recall, F1, ROC AUC |
| Plots | confusion matrices, ROC curve, class balance, correlation heatmap (`figures/`) |
| Saving | best model saved to `models/heart_model.joblib`, reloaded and used for predictions |

The final model is chosen by **ROC AUC** (F1 breaks ties). Recall is reported too, since missing a sick patient is the costly error.

## Outputs
`results.json`, `metrics.csv`, `figures/*.png`, `models/heart_model.joblib`, `Heart_Disease_Presentation.pptx`

## Practice set (`practice_set.py`)
1. `cross_val_score` for accuracy  2. `GridSearchCV` for SVM  3. save with `joblib`  4. load and predict.

Note: educational project only, not a medical diagnostic tool.
