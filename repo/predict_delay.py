"""Part D (3/3) - turn data/new_shipments.csv into an action list (delay probability x hub segment)."""
import joblib
import pandas as pd
from src.shipment_prep import build_features, clean_shipments

new_raw = pd.read_csv("data/new_shipments.csv", encoding="utf-8")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = dict(zip(imp["column"], imp["median"]))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new_raw, medians=medians)   # same cleaning as training, stored medians
feat = build_features(clean, hubs)
X = feat[feature_order]

feat["delay_probability"] = model.predict_proba(X)[:, 1].round(4)
feat["flagged"] = (feat["delay_probability"] >= threshold).astype(int)
feat = feat.merge(segments, on="hub_id", how="left")


def action(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    if row["flagged"] == 1:
        return "Proactive customer call"
    return "No action"


feat["action"] = feat.apply(action, axis=1)
out = feat[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(len(out), "parcels;", int(out["flagged"].sum()), "flagged;",
      int((out["action"] == "Escalate to hub manager").sum()), "escalations")
