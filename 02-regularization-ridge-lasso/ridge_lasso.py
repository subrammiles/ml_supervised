"""Regularization: penalise large coefficients to reduce overfitting.

Ridge (L2): shrinks all coefficients toward 0; good when many features matter a little.
Lasso (L1): can shrink coefficients to EXACTLY 0, so it also selects features.
alpha controls the strength: 0 = plain linear regression; larger = simpler model.
"""
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(1)
n_samples, n_features = 60, 40          # few rows, many columns: easy to overfit
X = rng.normal(size=(n_samples, n_features))
true_coefs = np.zeros(n_features)
true_coefs[:5] = [4, -3, 2, 1.5, -1]     # only the first 5 features matter
y = X @ true_coefs + rng.normal(0, 1.5, n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Scale first: the penalty treats all coefficients equally, so features must be
# on the same scale for regularization to be fair.
models = {
    "Linear (no penalty)": LinearRegression(),
    "Ridge alpha=10": Ridge(alpha=10),
    "Lasso alpha=0.2": Lasso(alpha=0.2),
}
print(f"{'model':<22} {'train R^2':>9} {'test R^2':>9} {'non-zero coefs':>15}")
for name, reg in models.items():
    pipe = make_pipeline(StandardScaler(), reg).fit(X_train, y_train)
    coefs = pipe[-1].coef_
    print(f"{name:<22} {pipe.score(X_train, y_train):9.3f} "
          f"{pipe.score(X_test, y_test):9.3f} {np.sum(np.abs(coefs) > 1e-6):15d}")
# Plain regression: almost perfect on train, poor on test (overfitting).
# Regularized models trade a little training fit for much better test scores.

# How alpha changes Lasso's coefficients
print("\nLasso: number of features kept as alpha grows")
for alpha in [0.01, 0.1, 0.3, 1.0, 3.0]:
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=alpha)).fit(X_train, y_train)
    kept = np.flatnonzero(np.abs(lasso[-1].coef_) > 1e-6)
    print(f"  alpha={alpha:<5} keeps {len(kept):>2} features "
          f"test R^2={r2_score(y_test, lasso.predict(X_test)):.3f}")

# Choose alpha automatically with cross-validation
cv_model = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)).fit(X_train, y_train)
coefs = cv_model[-1].coef_
kept = np.flatnonzero(np.abs(coefs) > 1e-6)
print(f"\nLassoCV chose alpha={cv_model[-1].alpha_:.3f}, keeping {len(kept)} features")
print(f"Test R^2: {cv_model.score(X_test, y_test):.3f}")
largest = np.argsort(np.abs(coefs))[::-1][:5]
print(f"5 largest coefficients: features {sorted(largest.tolist())}  (truly useful: [0, 1, 2, 3, 4])")
print(f"Size of the largest 5: {np.round(np.abs(coefs[largest]), 2).tolist()}")
print(f"Largest of the rest:   {np.sort(np.abs(np.delete(coefs, largest)))[-1]:.2f}")
# CV picks the alpha that PREDICTS best, which often keeps some noise features
# with tiny coefficients. The big coefficients still point to the real ones.
# For stricter selection, use a larger alpha (fewer features, slightly worse fit).

# Related: ElasticNet mixes L1 and L2. For classification, LogisticRegression
# uses the same idea through its C parameter (C = 1 / alpha: smaller C, stronger penalty).
