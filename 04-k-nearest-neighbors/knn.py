"""k-nearest neighbours: predict from the k most similar training examples.

No real training step: the model memorises the data, and all the work happens
at prediction time (find neighbours, then vote or average).
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# KNN by hand on a tiny example
train_points = np.array([[1, 1], [2, 1], [1, 2], [6, 6], [7, 6], [6, 7]])
train_labels = np.array(["A", "A", "A", "B", "B", "B"])
query = np.array([5, 5])
distances = np.sqrt(((train_points - query) ** 2).sum(axis=1))
nearest = np.argsort(distances)[:3]
print("Distances:", distances.round(2))
print("3 nearest labels:", train_labels[nearest], "-> vote:",
      max(set(train_labels[nearest]), key=list(train_labels[nearest]).count))

# Scaling matters: KNN uses distances, so big-number features dominate.
wine = load_wine()
X, y = wine.data, wine.target
print("\nFeature ranges differ a lot, e.g.",
      f"proline {X[:, -1].min():.0f}-{X[:, -1].max():.0f} vs hue {X[:, 10].min():.1f}-{X[:, 10].max():.1f}")

unscaled = cross_val_score(KNeighborsClassifier(), X, y, cv=5).mean()
scaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier()), X, y, cv=5).mean()
print(f"Accuracy without scaling: {unscaled:.3f}")
print(f"Accuracy with scaling:    {scaled:.3f}")

# Choosing k: small k = follows noise (overfit), large k = too smooth (underfit)
print("\nk  | CV accuracy")
for k in [1, 3, 5, 11, 25, 51]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    print(f"{k:<3}| {cross_val_score(pipe, X, y, cv=5).mean():.3f}")

# KNN regression: average the neighbours' values
sizes = np.array([[50], [60], [80], [100], [120], [150]])
rents = np.array([800, 950, 1200, 1500, 1700, 2100])
knn_reg = KNeighborsRegressor(n_neighbors=2).fit(sizes, rents)
print(f"\nRent for 90 m^2 (average of 2 nearest): {knn_reg.predict([[90]])[0]:.0f}")

# Pros: simple, no assumptions about the shape of the data.
# Cons: slow prediction on big datasets; struggles with many features
# ("curse of dimensionality": in high dimensions everything is far apart).
