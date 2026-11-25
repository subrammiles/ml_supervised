"""Gradient boosting: build trees one after another, each fixing the previous errors.

Random forest: many deep trees in PARALLEL, averaged.
Boosting:      many small trees in SEQUENCE; each new tree predicts the
               remaining errors (residuals) of the model so far.
Often the most accurate choice for tables of data (XGBoost, LightGBM, CatBoost).
"""
import numpy as np
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1) Boosting by hand for regression: fit residuals step by step
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

learning_rate = 0.1
pred_train = np.full(len(y_train), y_train.mean())  # start with the average
pred_test = np.full(len(y_test), y_train.mean())
print("trees | train RMSE | test RMSE")
for i in range(1, 201):
    residuals = y_train - pred_train
    small_tree = DecisionTreeRegressor(max_depth=2).fit(X_train, residuals)
    pred_train += learning_rate * small_tree.predict(X_train)  # take a small step
    pred_test += learning_rate * small_tree.predict(X_test)
    if i in (1, 10, 50, 100, 200):
        print(f"{i:<5} | {mean_squared_error(y_train, pred_train) ** 0.5:10.1f} | "
              f"{mean_squared_error(y_test, pred_test) ** 0.5:9.1f}")
# Train error keeps falling; test error stops improving -> boosting CAN overfit,
# so the number of trees must be chosen (early stopping, below).

# 2) The library version: HistGradientBoosting (fast, handles missing values)
Xc, yc = make_classification(n_samples=5_000, n_features=20, n_informative=8,
                             random_state=0)
Xc[np.random.default_rng(0).random(Xc.shape) < 0.05] = np.nan  # 5% missing values
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.25, random_state=0)

gb = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=500,              # maximum number of trees
    early_stopping=True,       # stop when the validation score stops improving
    validation_fraction=0.1,
    random_state=0,
)
gb.fit(Xc_train, yc_train)  # no need to fill the missing values first
print(f"\nHistGradientBoosting: test accuracy {gb.score(Xc_test, yc_test):.3f}, "
      f"stopped after {gb.n_iter_} trees (of max 500)")

# Random forest cannot take NaN in older versions, so fill values for a fair comparison
filled_train = np.nan_to_num(Xc_train)
filled_test = np.nan_to_num(Xc_test)
rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(filled_train, yc_train)
print(f"Random forest:        test accuracy {rf.score(filled_test, yc_test):.3f}")

# 3) Learning rate vs number of trees: smaller steps need more trees
print("\nlearning_rate | trees used | test accuracy")
for lr in [0.3, 0.1, 0.03]:
    m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=1000, early_stopping=True,
                                       random_state=0).fit(Xc_train, yc_train)
    print(f"{lr:<13} | {m.n_iter_:>10} | {m.score(Xc_test, yc_test):.3f}")

# Main settings: learning_rate, number of trees (with early stopping),
# max_depth / max_leaf_nodes (tree size), l2_regularization.
# XGBoost / LightGBM / CatBoost are popular libraries built on the same idea.
