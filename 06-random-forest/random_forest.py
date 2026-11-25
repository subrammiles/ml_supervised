"""Random forest: many decision trees that vote (bagging).

Each tree is trained on a random sample of the rows (bootstrap) and considers a
random subset of features at each split. The trees make different mistakes,
so their average is more accurate and much more stable than one tree.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# One tree vs a forest, with cross-validation to show the spread
tree_scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=10)
forest_scores = cross_val_score(RandomForestClassifier(random_state=0), X, y, cv=10)
print(f"Single tree:   {tree_scores.mean():.3f} +/- {tree_scores.std():.3f}")
print(f"Random forest: {forest_scores.mean():.3f} +/- {forest_scores.std():.3f}")

# More trees: better up to a point, then it levels off (it does not overfit with more trees)
print("\nn_estimators | CV accuracy")
for n in [1, 5, 25, 100, 300]:
    rf = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    print(f"{n:<12} | {cross_val_score(rf, X, y, cv=10).mean():.3f}")

# Out-of-bag (OOB) score: each tree is tested on the rows it did not see.
# A free validation estimate without a separate split.
forest = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
print(f"\nOOB score: {forest.oob_score_:.3f}, test score: {forest.score(X_test, y_test):.3f}")

# Feature importance, two ways
impurity_top = np.argsort(forest.feature_importances_)[::-1][:5]
print("\nTop 5 features (built-in impurity importance):")
for i in impurity_top:
    print(f"  {data.feature_names[i]:<25} {forest.feature_importances_[i]:.3f}")

# Permutation importance: how much the test score drops when a feature is shuffled.
# More reliable, especially when features are correlated or have many values.
perm = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
perm_top = np.argsort(perm.importances_mean)[::-1][:5]
print("Top 5 features (permutation importance):")
for i in perm_top:
    print(f"  {data.feature_names[i]:<25} {perm.importances_mean[i]:.3f}")
# Permutation values are small here because many features are strongly
# correlated (radius, perimeter, area): shuffle one and the forest uses the others.
# Low importance can mean "redundant", not "useless".

# Key settings: n_estimators (more is safer, slower), max_depth / min_samples_leaf
# (limit each tree), max_features (randomness per split; "sqrt" is a good default).
# RandomForestRegressor works the same way for numbers (averages instead of votes).
