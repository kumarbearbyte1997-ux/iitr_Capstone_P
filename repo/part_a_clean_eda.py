"""Part A - clean the raw export, build features, write the modelling table and EDA evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv", encoding="utf-8")
hubs = pd.read_csv("data/hubs.csv")

# "Before" counts: after removing duplicates, before any filling
dedup = raw.drop_duplicates()
spell_before = dedup["service_type"].nunique()
sentinel_rows = int((dedup["parcel_weight_kg"] > 100).sum())
missing_before = {
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
}

clean, medians = clean_shipments(raw)
spell_after = clean["service_type"].nunique()
feat = build_features(clean, hubs)
rows_after_hub_merge = len(feat)

feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)
feat.to_csv("data/modelling_table.csv", index=False)

# IQR rule on unrounded quartiles
q1, q3 = feat["delivery_hours"].quantile(0.25), feat["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = int(((feat["delivery_hours"] < lo) | (feat["delivery_hours"] > hi)).sum())

dv = feat["declared_value_inr"]
checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    **missing_before,
    "sentinel_weight_rows": sentinel_rows,
    "service_type_spellings_before": spell_before,
    "service_type_spellings_after": spell_after,
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "declared_value_mean_inr": dv.mean(),
    "declared_value_median_inr": dv.median(),
    "log_declared_value_mean": feat["log_declared_value"].mean(),
    "log_declared_value_median": feat["log_declared_value"].median(),
    "rows_after_hub_merge": rows_after_hub_merge,
    "nulls_in_features_and_target": int(feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(feat["is_delayed"].sum()),
    "delay_rate": feat["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": outliers,
}
pd.DataFrame({"metric": list(checks.keys()), "value": list(checks.values())}).round(4).to_csv(
    "outputs/part_a_checks.csv", index=False)

# Delay evidence by group
feat["distance_band"] = pd.cut(
    feat["distance_km"], bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = feat.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean")).reset_index()
    g = g.rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    g["group"] = g["group"].astype(str)
    rows.append(g)
pd.concat(rows, ignore_index=True).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# Correlations of numeric columns of the modelling table
num = feat.drop(columns=["distance_band"]).select_dtypes(include=[np.number])
num.corr().round(4).to_csv("outputs/part_a_correlations.csv")

print("Part A done:", len(feat), "rows; delay rate", round(feat["is_delayed"].mean(), 4))
