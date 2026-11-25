"""Handle rare classes (fraud, disease, churn): what helps and how to measure it.

With 3% positives, a model can ignore the rare class and still score 97% accuracy.
Remedies: class weights, resampling, and choosing the decision threshold.
Measure with recall, precision, F1, and PR AUC, not accuracy.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

X, y = make_classification(n_samples=10_000, n_features=12, n_informative=5,
                           weights=[0.97], class_sep=2.0, flip_y=0, random_state=1)
# stratify keeps the same 3% rate in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=0)
print(f"Positive rate: train {y_train.mean():.2%}, test {y_test.mean():.2%}\n")


def report(name, model, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(f"{name:<32} precision={precision_score(y_test, pred, zero_division=0):.2f}  "
          f"recall={recall_score(y_test, pred):.2f}  f1={f1_score(y_test, pred):.2f}  "
          f"PR-AUC={average_precision_score(y_test, proba):.2f}")


# 1) Plain model
plain = LogisticRegression(max_iter=1000).fit(X_train, y_train)
report("Plain", plain)

# 2) Class weights: mistakes on the rare class cost more during training
weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
report("class_weight='balanced'", weighted)

# 3) Random oversampling: repeat rare-class rows in the TRAINING data only
X_min, X_maj = X_train[y_train == 1], X_train[y_train == 0]
X_min_up = resample(X_min, n_samples=len(X_maj), replace=True, random_state=0)
X_bal = np.vstack([X_maj, X_min_up])
y_bal = np.array([0] * len(X_maj) + [1] * len(X_min_up))
oversampled = LogisticRegression(max_iter=1000).fit(X_bal, y_bal)
report("Oversampled training data", oversampled)
# Never resample before splitting: copies of the same row would end up in both
# train and test, making scores look better than they are. (The imbalanced-learn
# library offers SMOTE and more, and handles this correctly in pipelines.)

# 4) Keep the plain model, but move the threshold
report("Plain, threshold 0.2", plain, threshold=0.2)

# Choose the threshold that gives the best F1 (or meets a recall target)
precision, recall, thresholds = precision_recall_curve(y_test, plain.predict_proba(X_test)[:, 1])
f1 = 2 * precision * recall / (precision + recall + 1e-12)
best = np.argmax(f1[:-1])
print(f"\nBest F1 threshold: {thresholds[best]:.2f} "
      f"(precision {precision[best]:.2f}, recall {recall[best]:.2f})")
target_idx = np.flatnonzero(recall[:-1] >= 0.8)[-1]
print(f"Highest threshold with recall >= 0.80: {thresholds[target_idx]:.2f} "
      f"(precision {precision[target_idx]:.2f})")
# In practice, choose the threshold on a validation set, not the test set.

# Notice: weighting, oversampling and lowering the threshold all trade precision
# for recall. PR-AUC (ranking quality) barely changes: they move the cut-off,
# they do not make the model better at telling the classes apart.
