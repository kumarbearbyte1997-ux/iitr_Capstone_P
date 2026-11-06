"""Part B - delivery-hours regression: split, leakage-safe scaling, model comparison, CV, one test score."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]

# 1. Split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. Scaling without leakage: fit on training rows only
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
checks = {
    "train_rows": len(X_train), "val_rows": len(X_val), "test_rows": len(X_test),
    "train_scaled_max_abs_mean": np.abs(train_scaled.mean(axis=0)).max(),
    "test_scaled_max_abs_mean": np.abs(test_scaled.mean(axis=0)).max(),
}
pd.DataFrame({"metric": list(checks), "value": list(checks.values())}).round(4).to_csv(
    "outputs/part_b_checks.csv", index=False)


def score(model_name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": model_name, "split": split, "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}


rows = []

# 3. Baseline: mean of y_train
rows.append(score("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean())))

# 4. Tree-depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({
        "max_depth": "unlimited" if d is None else d,
        "train_rmse": np.sqrt(mean_squared_error(y_train, t.predict(X_train))),
        "val_rmse": np.sqrt(mean_squared_error(y_val, t.predict(X_val))),
        "_d": d})
depth_df = pd.DataFrame(depth_rows)
# lowest val_rmse; ties -> smaller depth (unlimited counts as the largest)
depth_df["_sort"] = depth_df["_d"].map(lambda v: 10**9 if pd.isna(v) else v)
best = depth_df.sort_values(["val_rmse", "_sort"]).iloc[0]
best_depth = None if pd.isna(best["_d"]) else int(best["_d"])
depth_df[["max_depth", "train_rmse", "val_rmse"]].round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. Four candidates
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    r = score(name, "val", y_val, model.predict(X_val))
    rows.append(r)
    val_rmse[name] = r["RMSE"]

# 6. Learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, p)),
                    "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. Cross-validation of the two best (by validation RMSE) + final model
ranking = sorted(val_rmse, key=val_rmse.get)
top2 = ranking[:2]
X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)   # one shared object
cv_rows = []
for name in top2:
    s = cross_val_score(clone(candidates[name]), X_trval, y_trval, cv=kf)
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranking[0]
rows.append(score(final_name, "test", y_test, candidates[final_name].predict(X_test)))  # scored once
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)

print("Validation ranking:", ranking, "| final:", final_name, "| best tree depth:", best_depth)
