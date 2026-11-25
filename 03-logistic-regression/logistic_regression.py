"""Logistic regression: a linear model for CLASSIFICATION that outputs probabilities.

score = b0 + b1*x1 + b2*x2 + ...      (same weighted sum as linear regression)
probability = sigmoid(score) = 1 / (1 + e^-score)   (squashes the score into 0..1)
predict class 1 if probability >= 0.5
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print("sigmoid turns any score into a probability:")
for z in [-5, -1, 0, 1, 5]:
    print(f"  score {z:>2} -> {sigmoid(z):.3f}")

# Binary example: will a student pass, from hours studied and hours slept?
rng = np.random.default_rng(0)
n = 200
hours_study = rng.uniform(0, 10, n)
hours_sleep = rng.uniform(4, 9, n)
true_score = -6 + 0.9 * hours_study + 0.5 * hours_sleep
passed = (rng.random(n) < sigmoid(true_score)).astype(int)
X = np.column_stack([hours_study, hours_sleep])

X_train, X_test, y_train, y_test = train_test_split(X, passed, test_size=0.25, random_state=0)
model = LogisticRegression().fit(X_train, y_train)
print(f"\nTest accuracy: {model.score(X_test, y_test):.3f}")

# Coefficients are in "log-odds". exp(coef) = how the ODDS multiply per +1 unit.
for name, coef in zip(["hours studied", "hours slept"], model.coef_[0]):
    print(f"{name:<14} coef={coef:+.2f}  -> each extra hour multiplies the odds by {np.exp(coef):.2f}")

students = np.array([[2, 6], [5, 7], [8, 8]])
for s, p in zip(students, model.predict_proba(students)[:, 1]):
    print(f"Study {s[0]}h, sleep {s[1]}h -> P(pass) = {p:.2f}")

# Decision boundary: where the probability is exactly 0.5 (score = 0).
# For 7 hours of sleep, how much study is needed?
b0, (b_study, b_sleep) = model.intercept_[0], model.coef_[0]
print(f"With 7h sleep, need about {-(b0 + b_sleep * 7) / b_study:.1f}h of study for a 50% chance")

# Multiclass: the same model handles 3+ classes (softmax over one score per class)
iris = load_iris()
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=0, stratify=iris.target)
multi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
multi.fit(Xi_train, yi_train)
print(f"\nIris (3 classes) test accuracy: {multi.score(Xi_test, yi_test):.3f}")
print("Class probabilities for one flower:", multi.predict_proba(Xi_test[:1]).round(3))

# Regularization strength: C (smaller C = stronger penalty, simpler model)
for C in [0.01, 1, 100]:
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000)).fit(Xi_train, yi_train)
    print(f"C={C:<5} test accuracy {m.score(Xi_test, yi_test):.3f}")
