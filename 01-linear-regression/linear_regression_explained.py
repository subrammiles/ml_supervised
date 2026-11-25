"""Linear regression in more depth: coefficients, R^2, residuals, and categorical inputs.

Model: price = b0 + b1*area + b2*rooms + ...  (a weighted sum of the features)
Training finds the weights that minimise the squared errors (least squares).
"""
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(0)
n = 300
houses = pd.DataFrame({
    "area": rng.uniform(800, 4000, n).round(),
    "rooms": rng.integers(1, 7, n),
    "age": rng.integers(0, 50, n),
    "location": rng.choice(["city", "suburb", "rural"], n, p=[0.3, 0.5, 0.2]),
})
location_bonus = houses["location"].map({"city": 80_000, "suburb": 30_000, "rural": 0})
houses["price"] = (
    50_000 + 150 * houses["area"] + 10_000 * houses["rooms"]
    - 1_500 * houses["age"] + location_bonus + rng.normal(0, 25_000, n)
).round()

X = houses.drop(columns="price")
y = houses["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# One-hot encode the text column; drop="first" avoids redundant columns
model = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(drop="first"), ["location"]),
        remainder="passthrough",
    ),
    LinearRegression(),
)
model.fit(X_train, y_train)

# Coefficients: the change in price for +1 of that feature, others held constant
names = model[0].get_feature_names_out()
coefs = model[-1].coef_
print(f"Intercept: {model[-1].intercept_:,.0f}")
for name, coef in zip(names, coefs):
    print(f"  {name:<28} {coef:>+12,.0f}")
print("(True values: area +150, rooms +10,000, age -1,500, city vs rural +80,000)")
# The location coefficients are compared with the dropped category ("city"),
# so rural should be about -80,000 and suburb about -50,000.

pred = model.predict(X_test)
print(f"\nTest R^2: {r2_score(y_test, pred):.3f}")
print(f"Test MAE: {mean_absolute_error(y_test, pred):,.0f}")

# Residual check: errors should look like random noise around 0
residuals = y_test - pred
print(f"Residual mean: {residuals.mean():,.0f} (should be near 0)")
print(f"Residual spread for cheap vs expensive houses: "
      f"{residuals[pred < np.median(pred)].std():,.0f} vs "
      f"{residuals[pred >= np.median(pred)].std():,.0f} (should be similar)")

# Predict new houses
new = pd.DataFrame([
    {"area": 2000, "rooms": 3, "age": 10, "location": "suburb"},
    {"area": 2000, "rooms": 3, "age": 10, "location": "city"},
])
for (_, row), p in zip(new.iterrows(), model.predict(new)):
    print(f"{row['location']:<7} 2000 sq ft -> {p:,.0f}")

# Limits of linear regression:
#  - it can only draw straight-line (or flat-plane) relationships
#    (add polynomial features or use trees for curves)
#  - strongly correlated features make individual coefficients unstable (see 02)
#  - outliers pull the line strongly, because errors are squared
