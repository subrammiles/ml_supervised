# Supervised learning

Runnable examples of the main supervised learning algorithms: how each one works, its key settings, and when to use it. Each script prints its results and explains them in comments. Data is generated in the script or built into scikit-learn, so nothing needs to be downloaded.

New to machine learning? Start with the `ml_intro` repository (types of ML, train/test split, overfitting, metrics), then come back here.

## Setup

```bash
python -m pip install numpy pandas scikit-learn scipy matplotlib
```

Run a script from the repository root, for example:

```bash
python 03-logistic-regression/logistic_regression.py
```

## Learning path

### Linear models

- `01-linear-regression/houseprice.py` — a first regression model on a tiny house-price table (see `doc.md`).
- `01-linear-regression/linear_regression_explained.py` — reading coefficients, one-hot encoded categories, R², residual checks.
- `02-regularization-ridge-lasso/ridge_lasso.py` — Ridge and Lasso, choosing `alpha`, Lasso for feature selection.
- `03-logistic-regression/logistic_regression.py` — sigmoid, probabilities, odds ratios, decision boundary, multiclass, `C`.

### Distance- and rule-based models

- `04-k-nearest-neighbors/knn.py` — KNN by hand, why scaling matters, choosing `k`, KNN regression.
- `05-decision-trees/decision_tree.py` — Gini impurity, readable rules, depth and overfitting, regression trees.

### Ensembles

- `06-random-forest/random_forest.py` — bagging, number of trees, out-of-bag score, two kinds of feature importance.
- `07-gradient-boosting/gradient_boosting.py` — boosting by hand, `HistGradientBoosting`, early stopping, learning rate.

### Other important algorithms

- `08-support-vector-machines/svm.py` — margins, linear vs. RBF kernel, `C` and `gamma`, calibrated probabilities.
- `09-naive-bayes-text-classification/spam_classifier.py` — bag of words, Naive Bayes spam filter, TF-IDF + logistic regression.

### Practical problems

- `10-imbalanced-data/imbalanced_classes.py` — class weights, oversampling, choosing the decision threshold, PR-AUC.
- `11-model-selection-and-tuning/compare_and_tune.py` — compare 8 algorithms fairly, then tune with `RandomizedSearchCV`.

## Choosing an algorithm (rules of thumb)

| Situation | Good first choices |
|---|---|
| Need an explainable model | Linear / logistic regression, a shallow decision tree |
| Tabular data, best accuracy | Gradient boosting, random forest |
| Many features, few rows | Regularized linear models (Ridge, Lasso, logistic with small `C`) |
| Text classification | Naive Bayes or TF-IDF + logistic regression as a baseline |
| Small dataset with complex boundary | SVM with RBF kernel, KNN |

Always compare against a simple baseline, and judge models with cross-validation rather than a single split.
