"""Decision trees: learn a flowchart of yes/no questions.

At each step the tree picks the question that best separates the classes
(lowest Gini impurity), then repeats on each side until it stops.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

# Gini impurity: 0 = all one class (pure), higher = mixed
def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

print("Gini of [A, A, A, A]:", gini(["A"] * 4))
print("Gini of [A, A, B, B]:", gini(["A", "A", "B", "B"]))
print("Gini of [A, A, A, B]:", round(gini(["A", "A", "A", "B"]), 3))

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=0, stratify=iris.target)

# A shallow tree is readable: print its rules
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
print("\nLearned rules:")
print(export_text(tree, feature_names=list(iris.feature_names)))
print(f"Test accuracy (depth 2): {tree.score(X_test, y_test):.3f}")
print("Feature importances:",
      {name: round(float(imp), 2) for name, imp in zip(iris.feature_names, tree.feature_importances_)})

# Depth controls complexity. An unlimited tree memorises the training data.
# Data: class depends on whether features 0 and 1 have the same sign (needs
# several splits to learn), plus 15% of labels flipped at random (noise).
rng = np.random.default_rng(0)
X_noisy = rng.normal(size=(400, 5))
pattern = (X_noisy[:, 0] > 0) ^ (X_noisy[:, 1] > 0)
y_noisy = (pattern ^ (rng.random(400) < 0.15)).astype(int)
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noisy, y_noisy, random_state=0)
print("\nmax_depth | train acc | test acc | leaves")
for depth in [1, 3, 5, 10, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xn_train, yn_train)
    print(f"{str(depth):<9} | {t.score(Xn_train, yn_train):9.3f} | "
          f"{t.score(Xn_test, yn_test):8.3f} | {t.get_n_leaves()}")
# Too shallow (1-2) cannot learn the pattern; around 3 is best; deeper trees
# learn the noisy labels too: train accuracy rises, test accuracy falls.
# Other ways to limit a tree: min_samples_leaf, min_samples_split, ccp_alpha (pruning).

# Regression trees predict the average value in each leaf: a step function
hours = np.arange(0, 24).reshape(-1, 1)
traffic = np.array([5, 3, 2, 2, 3, 10, 40, 90, 95, 60, 45, 50,
                    55, 50, 45, 50, 70, 95, 90, 60, 35, 20, 12, 8])
reg_tree = DecisionTreeRegressor(max_depth=2).fit(hours, traffic)
print("\nTraffic predicted by a depth-2 regression tree:")
print(dict(zip(range(0, 24, 3), reg_tree.predict(hours[::3]).round(0).tolist())))

# Pros: easy to explain, no scaling needed, handles non-linear patterns.
# Cons: a single tree overfits easily and changes a lot with small data changes.
# Fix: combine many trees (06-random-forest, 07-gradient-boosting).
