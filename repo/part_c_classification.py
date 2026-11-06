"""Part C - delay classifier, feature signal check, tuning and the operations alert threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]

# 1. Stratified split (stratify in both calls)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test),
      "delayed", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))


def cm_row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# 2. Blind baseline + three classifiers (default 0.5 rule)
rows = [cm_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]

logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append(cm_row("logistic_regression", y_val, logit.predict(X_val)))

tree_f1 = {}
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    tree_f1[d] = (f1_score(y_val, t.predict(X_val)), t)
best_d = sorted(tree_f1, key=lambda d: (-tree_f1[d][0], d))[0]   # highest F1, ties -> smaller d
rows.append(cm_row(f"decision_tree_depth_{best_d}", y_val, tree_f1[best_d][1].predict(X_val)))

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append(cm_row("random_forest_default", y_val, rf_default.predict(X_val)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. Feature signal: forest importance + ANOVA F-score on training rows
rf_all = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
skb_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": skb_all.scores_, "rf_importance": rf_all.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

k_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    k_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
pd.DataFrame(k_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. Tuning with k = 7 (selector fitted on training rows only; test rows untouched)
selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(Xtr7, y_train)
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5,
                          random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)

tuning = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": str(grid.best_params_), "best_cv_accuracy": grid.best_score_,
     "val_f1": f1_score(y_val, grid.best_estimator_.predict(Xva7))},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": str(rand.best_params_), "best_cv_accuracy": rand.best_score_,
     "val_f1": f1_score(y_val, rand.best_estimator_.predict(Xva7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "{'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1}", "best_cv_accuracy": np.nan,
     "val_f1": f1_score(y_val, default7.predict(Xva7))},
]
tuning_df = pd.DataFrame(tuning)
tuning_df.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

# 5. Final configuration: highest val_f1, ties -> default, grid, randomized
prio = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
tuning_df["_p"] = tuning_df["search"].map(prio)
tuning_df["_f1r"] = tuning_df["val_f1"].round(10)
winner = tuning_df.sort_values(["_f1r", "_p"], ascending=[False, True]).iloc[0]
if winner["search"] == "grid_search":
    params = dict(grid.best_params_)
elif winner["search"] == "randomized_search":
    params = dict(rand.best_params_)
else:
    params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
final_model = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                        ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. Alert threshold from validation probabilities
p_val = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t_rows = []
for t in thresholds:
    r = cm_row("x", y_val, (p_val >= t).astype(int))
    r.pop("model")
    t_rows.append({"threshold": t, **r})
t_df = pd.DataFrame(t_rows)
t_df.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
ok = t_df[t_df["recall"] >= 0.80]
chosen_t = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. One test evaluation at the chosen threshold
p_test = final_model.predict_proba(X_test)[:, 1]
r = cm_row("x", y_test, (p_test >= chosen_t).astype(int))
r.pop("model")
pd.DataFrame([{"threshold": chosen_t, **r}]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)

md = params["max_depth"]
pd.DataFrame([{
    "selector_k": 7, "n_estimators": params["n_estimators"],
    "max_depth": "unlimited" if md is None else md,
    "min_samples_leaf": params["min_samples_leaf"], "threshold": chosen_t,
    "chosen_from": winner["search"]}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("final:", winner["search"], params, "threshold", chosen_t)
