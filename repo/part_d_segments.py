"""Part D (1/3) - hub profile, K-Means segments, separation evidence and tier-vs-segment check."""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. Hub profile (groupby sorts by ascending hub_id)
profile = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index().sort_values("hub_id").reset_index(drop=True)
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means on the four scaled profile columns, hubs in ascending hub_id order
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[cols])


def fit(k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)


# 3. Separation evidence: mean across-cluster / mean within-cluster pairwise distance (276 pairs)
def across_over_within(labels):
    across, within = [], []
    for i, j in combinations(range(len(Z)), 2):
        d = np.linalg.norm(Z[i] - Z[j])
        (within if labels[i] == labels[j] else across).append(d)
    return np.mean(across) / np.mean(within)


sep = []
for k in [2, 3, 4, 5]:
    lab = fit(k).labels_
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": k, "across_over_within": across_over_within(lab),
                "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# 4. Segment names by mean delay rate of member hubs: Standard < Watch < Escalate
labels = fit(3).labels_
profile["cluster"] = labels
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

seg_profile = profile.groupby("segment").agg(
    hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")).reset_index()
seg_profile["_o"] = seg_profile["segment"].map({"Standard": 0, "Watch": 1, "Escalate": 2})
seg_profile = seg_profile.sort_values("_o").drop(columns="_o")
seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

tier_implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = seg[seg["segment"] != seg["hub_tier"].map(tier_implied)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("segments:", seg["segment"].value_counts().to_dict(), "| mismatches:", mismatch["hub_id"].tolist())
