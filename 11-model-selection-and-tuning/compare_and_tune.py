"""Compare several algorithms fairly, then tune the best one.

1. Put preprocessing + model in a Pipeline (no data leakage).
2. Compare candidates with the same cross-validation folds.
3. Tune the most promising model's hyperparameters.
4. Report the final score once on the held-out test set.
"""
import time

from scipy.stats import loguniform, randint
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

candidates = {
    "Baseline (most frequent)": DummyClassifier(),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Naive Bayes": GaussianNB(),
    "Decision tree": DecisionTreeClassifier(random_state=0),
    "Random forest": RandomForestClassifier(random_state=0, n_jobs=-1),
    "Gradient boosting": HistGradientBoostingClassifier(random_state=0),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # same folds for all
print(f"{'model':<26} {'accuracy':>14} {'f1':>6} {'fit time':>9}")
for name, model in candidates.items():
    cv = cross_validate(model, X_train, y_train, cv=folds, scoring=["accuracy", "f1"])
    print(f"{name:<26} {cv['test_accuracy'].mean():.3f} +/- {cv['test_accuracy'].std():.3f} "
          f"{cv['test_f1'].mean():6.3f} {cv['fit_time'].mean():8.3f}s")
# Simple, scaled linear models are often as good as complex ones on small
# tabular datasets. Prefer the simpler model when scores are within the noise.

# Tune: RandomizedSearchCV samples settings at random. It is usually more
# efficient than trying every combination (GridSearchCV) when there are many.
search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=0),
    param_distributions={
        "learning_rate": loguniform(0.01, 0.3),
        "max_leaf_nodes": randint(8, 64),
        "min_samples_leaf": randint(5, 50),
        "l2_regularization": loguniform(1e-4, 10),
    },
    n_iter=25, cv=folds, scoring="f1", random_state=0, n_jobs=-1,
)
start = time.perf_counter()
search.fit(X_train, y_train)
print(f"\nRandom search: {time.perf_counter() - start:.1f}s, best CV f1 {search.best_score_:.3f}")
print("Best settings:", {k: round(float(v), 4) for k, v in search.best_params_.items()})

# Tune the logistic regression too (its main knob is C)
lr_search = RandomizedSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    {"logisticregression__C": loguniform(1e-3, 1e2)},
    n_iter=20, cv=folds, scoring="f1", random_state=0,
).fit(X_train, y_train)
print(f"Tuned logistic regression: best CV f1 {lr_search.best_score_:.3f}, "
      f"C={lr_search.best_params_['logisticregression__C']:.3f}")

# Final check ONCE on the test set
for name, s in [("Gradient boosting", search), ("Logistic regression", lr_search)]:
    # best_estimator_.score uses accuracy; s.score would use the search's metric (f1)
    print(f"Test accuracy, tuned {name}: {s.best_estimator_.score(X_test, y_test):.3f}")
