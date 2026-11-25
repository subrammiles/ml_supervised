"""Support vector machines: find the boundary with the widest margin between classes.

Linear SVM: the widest "street" between two classes.
Kernel trick (RBF): curved boundaries, by implicitly mapping data to a higher dimension.
C:     penalty for points inside the margin (large C = strict, may overfit).
gamma: (RBF) how far one example's influence reaches (large = wiggly boundary).
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Two interleaving half-moons: a straight line cannot separate them
X, y = make_moons(n_samples=500, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ["linear", "rbf"]:
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel)).fit(X_train, y_train)
    n_sv = model[-1].n_support_.sum()
    print(f"{kernel:<6} kernel: test accuracy {model.score(X_test, y_test):.3f}, "
          f"{n_sv} support vectors")
# Support vectors are the training points on or inside the margin. Only they
# define the boundary; the other points could be removed without changing it.

# Effect of C and gamma
print("\n   C  | gamma | train acc | test acc")
for C in [0.1, 1, 100]:
    for gamma in [0.1, 1, 50]:
        m = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma)).fit(X_train, y_train)
        print(f"{C:>5} | {gamma:>5} | {m.score(X_train, y_train):9.3f} | {m.score(X_test, y_test):.3f}")
# Large C and large gamma -> near-perfect train accuracy but worse test: overfitting.

# Tune both with cross-validation
search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]},
    cv=5,
).fit(X_train, y_train)
print(f"\nBest: {search.best_params_}, test accuracy {search.score(X_test, y_test):.3f}")

# SVMs give scores, not probabilities. Calibration learns how to turn the
# scores into probabilities. (Older code uses SVC(probability=True), which
# is deprecated since scikit-learn 1.9.)
prob_model = make_pipeline(
    StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False)
).fit(X_train, y_train)
print("Probabilities for 3 test points:", prob_model.predict_proba(X_test[:3]).round(2).tolist())

# Use SVMs for small-to-medium datasets with clear margins (text, images features).
# Always scale features. Training time grows quickly beyond ~100k rows.
print(f"\nDecision function (signed distance from boundary) for 3 points: "
      f"{np.round(search.decision_function(X_test[:3]), 2).tolist()}")
