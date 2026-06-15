import random
from math import floor, ceil
from pathlib import Path

import hdbscan
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from streamlit_app.data_processing import haversine_m

DATA_FILE = Path("../data/2024_11_17_17_00_processed.csv")
SEGMENTS_FILE = Path("../data/data_2024-11-17.csv")

EARTH_R = 6_371_000  # promień Ziemi [m]
PARAMS = {
    "time_gap_new_trip_s": 5 * 60,
    "max_plausible_speed_kmh": 180,
    "stop_speed_kmh": 3,
    "long_stop_s": 3 * 60,
    "endpoint_cluster_eps_m": 93,
    "endpoint_cluster_min_samples": 12,
    "min_trip_points": 4,
    "min_trip_duration_s": 30,
    "min_trip_distance_m": 200,
    "car_min_p90_speed_kmh": 45,
    "car_min_max_speed_kmh": 60,
    "voronoi_clip_rotation_deg": 45,
    "voronoi_clip_padding_m": 600,
}


def load_and_preprocess_data():
    df_raw = pd.read_csv(DATA_FILE, parse_dates=["timestamp", "time"])
    df_raw = df_raw.sort_values(["vehicle_id", "timestamp"]).copy()
    df = df_raw.copy()
    df["prev_timestamp"] = df.groupby("vehicle_id")["timestamp"].shift(1)
    df["prev_lat"] = df.groupby("vehicle_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("vehicle_id")["lon"].shift(1)
    df["time_diff_s"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    df["distance_m"] = haversine_m(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df.loc[df["time_diff_s"].isna(), "distance_m"] = np.nan
    df["gps_speed_kmh"] = (df["distance_m"] / df["time_diff_s"]) * 3.6

    df["is_stationary_point"] = df["speed"].le(PARAMS["stop_speed_kmh"])
    df["is_implausible_jump"] = df["gps_speed_kmh"].gt(PARAMS["max_plausible_speed_kmh"])
    df["stationary_block"] = df.groupby("vehicle_id")["is_stationary_point"].transform(
        lambda s: s.ne(s.shift()).cumsum())

    stop_blocks = df[df["is_stationary_point"]].groupby(["vehicle_id", "stationary_block"]).agg(
        stop_start=("timestamp", "min"), stop_end=("timestamp", "max")).reset_index()
    stop_blocks["stop_duration_s"] = (stop_blocks["stop_end"] - stop_blocks["stop_start"]).dt.total_seconds()
    long_stops = stop_blocks[stop_blocks["stop_duration_s"] >= PARAMS["long_stop_s"]].copy()

    long_stop_keys = set(zip(long_stops["vehicle_id"], long_stops["stationary_block"]))
    df["is_long_stop_block"] = list(zip(df["vehicle_id"], df["stationary_block"]))
    df["is_long_stop_block"] = df["is_long_stop_block"].isin(long_stop_keys)
    df["prev_is_long_stop_block"] = df.groupby("vehicle_id")["is_long_stop_block"].shift(1, fill_value=False)

    df["new_trip_reason"] = "continue"
    df.loc[df["time_diff_s"].isna(), "new_trip_reason"] = "first_point"
    df.loc[df["time_diff_s"].gt(PARAMS["time_gap_new_trip_s"]), "new_trip_reason"] = "time_gap"
    df.loc[df["is_implausible_jump"], "new_trip_reason"] = "gps_jump"
    df.loc[df["prev_is_long_stop_block"] & ~df["is_long_stop_block"], "new_trip_reason"] = "after_long_stop"
    df["new_trip"] = df["new_trip_reason"].ne("continue")
    df["trip_seq"] = df.groupby("vehicle_id")["new_trip"].cumsum()
    df["trip_uid"] = df["vehicle_id"].astype(str) + "_" + df["trip_seq"].astype(str)

    trip_stats = df.groupby("trip_uid").agg(
        vehicle_id=("vehicle_id", "first"), start_time=("timestamp", "min"), end_time=("timestamp", "max"),
        points=("timestamp", "size"), start_lat=("lat", "first"), start_lon=("lon", "first"),
        end_lat=("lat", "last"), end_lon=("lon", "last"), reported_speed_p90=("speed", lambda s: s.quantile(0.90)),
        reported_speed_max=("speed", "max"), distance_m=("distance_m", "sum"),
        implausible_jumps=("is_implausible_jump", "sum"),
    ).reset_index()

    trip_stats["duration_s"] = (trip_stats["end_time"] - trip_stats["start_time"]).dt.total_seconds()
    trip_stats["valid_basic"] = ~(trip_stats["points"].lt(PARAMS["min_trip_points"]) | trip_stats["duration_s"].lt(
        PARAMS["min_trip_duration_s"]) | trip_stats["distance_m"].lt(PARAMS["min_trip_distance_m"]) | trip_stats[
                                      "implausible_jumps"].gt(0))

    car_like = trip_stats["reported_speed_p90"].ge(PARAMS["car_min_p90_speed_kmh"]) | trip_stats[
        "reported_speed_max"].ge(PARAMS["car_min_max_speed_kmh"])
    trip_stats["mode_class"] = np.select([car_like], ["car_like"], default="other")
    trip_stats_filtered = trip_stats[trip_stats["valid_basic"] & trip_stats["mode_class"].eq("car_like")].copy()

    start_points = trip_stats_filtered[["trip_uid", "start_lat", "start_lon"]].rename(
        columns={"start_lat": "lat", "start_lon": "lon"})
    start_points["endpoint_type"] = "origin"
    end_points = trip_stats_filtered[["trip_uid", "end_lat", "end_lon"]].rename(
        columns={"end_lat": "lat", "end_lon": "lon"})
    end_points["endpoint_type"] = "destination"
    endpoints = pd.concat([start_points, end_points], ignore_index=True).dropna(subset=["lat", "lon"])

    return endpoints


def haversine_dist_rad(a, b):
    """Distance between two points in radians, returns meters."""
    return np.arccos(
        np.sin(a[0]) * np.sin(b[0]) +
        np.cos(a[0]) * np.cos(b[0]) * np.cos(a[1] - b[1])
    ) * EARTH_R


def compute_cluster_medoids(coords_rad, labels):
    """Return dict: cluster_id -> medoid coordinate (in radians)."""
    centers = {}
    for c in set(labels):
        if c == -1:
            continue
        pts = coords_rad[labels == c]
        D = pairwise_distances(pts, metric="haversine")
        medoid_idx = np.argmin(D.mean(axis=1))
        centers[c] = pts[medoid_idx]
    return centers


def enforce_min_center_distance(coords_rad, labels, min_center_dist_m, max_iter=10):
    """
    Iteratively merge clusters whose medoid centers are closer than min_center_dist_m.
    Always keeps the larger cluster and relabels the smaller one.
    Recomputes medoids after each merge.
    """

    for _ in range(max_iter):
        centers = compute_cluster_medoids(coords_rad, labels)
        cluster_ids = list(centers.keys())

        merged_any = False

        # Compare all cluster pairs
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                ci = cluster_ids[i]
                cj = cluster_ids[j]

                # Skip if one of them was merged earlier in this iteration
                if ci not in centers or cj not in centers:
                    continue

                d = haversine_dist_rad(centers[ci], centers[cj])

                if d < min_center_dist_m:
                    # Determine which cluster to keep
                    size_i = (labels == ci).sum()
                    size_j = (labels == cj).sum()

                    if size_i >= size_j:
                        keep, drop = ci, cj
                    else:
                        keep, drop = cj, ci

                    # Merge: relabel all points of the smaller cluster
                    labels[labels == drop] = keep

                    # Remove dropped cluster center
                    centers.pop(drop, None)

                    merged_any = True

        if not merged_any:
            break  # convergence

    # ---- Renumber labels to 0..K-1 ----
    unique = sorted(set(labels[labels != -1]))
    mapping = {old: new for new, old in enumerate(unique)}
    mapping[-1] = -1
    labels = np.array([mapping[l] for l in labels])

    return labels


def run_hdbscan(coords_rad, min_cluster_size, min_center_dist_m=None, persist=False):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="haversine",
        prediction_data=persist,
    )
    labels = clusterer.fit_predict(coords_rad)

    # post-processing: enforce minimum center spacing
    labels = enforce_min_center_distance(coords_rad, labels, min_center_dist_m)

    # metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()

    sil = np.nan
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = silhouette_score(coords_rad[mask], labels[mask], metric="haversine")

    return labels, n_clusters, noise_frac, sil, clusterer


def search_hdbscan_params(coords_rad, ms_list, min_center_dist_m):
    results = []

    for ms in ms_list:
        for dist in min_center_dist_m:
            _, nc, noise, sil, _ = run_hdbscan(coords_rad, ms, dist)
            results.append({
                "min_cluster_size": ms,
                "min_center_dist": dist,
                "n_clusters": nc,
                "noise_frac": noise,
                "silhouette": sil
            })

    return pd.DataFrame(results)


# -----------------------------
# Utility functions
# -----------------------------
def k_distance(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine").fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, k]) * EARTH_R


def knee(y):
    x = np.arange(len(y), dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    i = int(np.argmax(np.abs(yn - xn)))
    return i, float(y[i])


def run_dbscan(coords_rad, eps_m, min_samples):
    eps_rad = eps_m / EARTH_R
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(coords_rad)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()

    sil = np.nan
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = silhouette_score(coords_rad[mask], labels[mask], metric="haversine")

    return labels, n_clusters, noise_frac, sil


# -----------------------------
# Parameter search
# -----------------------------
def search_dbscan_params(coords_rad, ms_list, eps_expand_factor=0.15):
    """
    eps_expand_factor = search eps in ±15% around elbow
    """
    results = []

    for ms in ms_list:
        kd = k_distance(coords_rad, ms)
        _, eps_elbow = knee(kd)

        # search a small range around elbow
        eps_low = float(floor(eps_elbow * (1 - eps_expand_factor)))
        eps_high = float(ceil(eps_elbow * (1 + eps_expand_factor)))

        eps_candidates = np.linspace(eps_low, eps_high, 5).astype(float)

        for eps_m in eps_candidates:
            _, nc, noise, sil = run_dbscan(coords_rad, eps_m, ms)
            results.append({
                "min_samples": ms,
                "eps_m": eps_m,
                "n_clusters": nc,
                "noise_frac": noise,
                "silhouette": sil
            })

    df = pd.DataFrame(results)
    return df


def visualize_clusters_folium(coords_rad, labels, map_path="clusters_map.html"):
    """
    coords_rad : array of shape (N, 2) in radians
    labels     : cluster labels from HDBSCAN/DBSCAN
    map_path   : output HTML file
    """

    # convert back to degrees
    coords_deg = np.degrees(coords_rad)

    # center map on median location
    center_lat = np.median(coords_deg[:, 0])
    center_lon = np.median(coords_deg[:, 1])

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # unique clusters (excluding noise)
    unique_labels = sorted(set(labels))
    unique_clusters = [c for c in unique_labels if c != -1]

    # color map for clusters
    colormap = plt.get_cmap("tab20", len(unique_clusters)) #gist_ncar
    norm = colors.Normalize(vmin=0, vmax=len(unique_clusters) - 1)

    random.shuffle(unique_clusters)

    # assign colors
    cluster_colors = {
        c: colors.to_hex(colormap(norm(i)))
        for i, c in enumerate(unique_clusters)
    }

    # noise color
    noise_color = "#000000"

    # add points
    for (lat, lon), lab in zip(coords_deg, labels):
        if lab == -1:
            color = noise_color
            radius = 2
        else:
            color = cluster_colors[lab]
            radius = 4

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            weight=0,
        ).add_to(fmap)

    # add legend
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 200px; 
        background-color: white; 
        border:2px solid grey; z-index:9999; 
        font-size:14px;
        padding: 10px;
    ">
    <b>Clusters</b><br>
    """
    unique_clusters.sort()
    for c in unique_clusters:
        legend_html += f'<i style="background:{cluster_colors[c]};width:12px;height:12px;float:left;margin-right:8px;"></i> Cluster {c}<br>'
    legend_html += '<i style="background:#000;width:12px;height:12px;float:left;margin-right:8px;"></i> Noise<br>'
    legend_html += "</div>"

    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(map_path)
    print(f"Saved map to: {map_path}")

    return fmap


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    endpoints = load_and_preprocess_data()

    coords = endpoints[["lat", "lon"]].to_numpy()
    coords_rad = np.radians(coords)
    print("# points: {}".format(len(coords)))

    # ---- Train/test split ----
    X_train, X_test = train_test_split(
        coords_rad, test_size=0.25, random_state=42, shuffle=True
    )

    ms_list = [10, 12, 15, 20, 30]
    min_center_dist_m = [100, 150, 200, 300, 400, 500]
    # ---- Parameter search on TRAIN ----
    search_df = search_hdbscan_params(X_train, ms_list, min_center_dist_m)
    print(search_df.sort_values("silhouette", ascending=False).head(10))
    analyze_hdbscan_results(search_df)

    # evaluate_hdbscan(X_test, X_train, search_df, n_clust_max=60, noise_frac_max=0.3)
    evaluate_hdbscan(X_test, X_train, search_df, n_clust_max=50, n_clust_min=40)
    evaluate_hdbscan(X_test, X_train, search_df, n_clust_max=30, n_clust_min=20)
    evaluate_hdbscan(X_test, X_train, search_df, n_clust_max=40, n_clust_min=30)

    # ---- Parameter search on TRAIN only ----
    search_df_db = search_dbscan_params(X_train, ms_list)
    print("\n\n ----- DBSCAN -----\n")
    print(search_df_db.sort_values("silhouette", ascending=False).head(10))
    analyze_dbscan_results(search_df_db)

    print("---- Best unrestricted ----")
    get_best_and_test(X_test, search_df_db, X_train, "unrestricted")

    print("\n\n---- Best with noise frac below 15% ----")
    # get_best_and_test(X_test, search_df.loc[search_df["noise_frac"] <= 0.15], X_train, "noise15")
    get_best_and_test(X_test, search_df_db.loc[(search_df_db["n_clusters"] <= 40) & (search_df_db["n_clusters"] > 30)],
                      X_train, "clust30-40")  # & (search_df["noise_frac"] <= 0.2)

    print("\n\n---- Best with n_clusters below 40 and noise frac below 20%----")
    get_best_and_test(X_test, search_df_db.loc[(search_df_db["n_clusters"] <= 30) & (search_df_db["n_clusters"] > 20)], X_train, "clust20-30") #& (search_df["noise_frac"] <= 0.2)

    compare_hdbscan_dbscan(search_df,search_df_db)
    return


def evaluate_hdbscan(X_test, X_train, search_df, n_clust_max=40, n_clust_min=15):
    # filter reasonable models
    filtered = search_df.loc[
        (search_df["n_clusters"] <= n_clust_max) &
        (search_df["n_clusters"] > n_clust_min) &
        # (search_df["noise_frac"] <= noise_frac_max) &
        (search_df["silhouette"].notna())
        ]
    if filtered.empty:
        print(f"No HDBSCAN clustering satisfies: n_clusters={n_clust_min}-{n_clust_max}")
        return
    # pick best
    best_row = filtered.loc[filtered["silhouette"].idxmax()]
    best_ms = int(best_row["min_cluster_size"])
    best_dist = int(best_row["min_center_dist"])
    print("\n\n--- hdbscan ---")
    print("Best parameters (train):")
    print(best_row)
    labels_train, _, _, _, clusterer = run_hdbscan(X_train, best_ms, best_dist, True)
    visualize_clusters_folium(X_train, labels_train,
                              map_path=f"hdbscan_train_clusters_nc{n_clust_min}-{n_clust_max}.html")

    labels_test, strengths = hdbscan.approximate_predict(clusterer, X_test)

    # post-processing: enforce minimum center spacing
    labels_test = enforce_min_center_distance(X_test, labels_test, best_dist)

    # metrics
    nc_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
    noise_test = (labels_test == -1).mean()

    sil_test = np.nan
    mask = labels_test != -1
    if nc_test >= 2 and mask.sum() > nc_test:
        sil_test = silhouette_score(X_test[mask], labels_test[mask], metric="haversine")
    print("\nEvaluation on test set:")
    print("Clusters:", nc_test)
    print("Noise fraction:", round(noise_test, 3))
    print("Silhouette:", round(sil_test, 3))
    # visualize clusters on TEST set
    # visualize_clusters_folium(X_test, labels_test,
    #                           map_path=f"hdbscan_test_clusters_nc{n_clust_max}_nf{noise_frac_max}.html")

def analyze_hdbscan_results(search_df):
    """
    Produce comparative visualisations for HDBSCAN parameter search.
    """

    # Convert to pivot tables for heatmaps
    piv_sil = search_df.pivot_table(
        index="min_cluster_size",
        columns="min_center_dist",
        values="silhouette"
    )

    piv_noise = search_df.pivot_table(
        index="min_cluster_size",
        columns="min_center_dist",
        values="noise_frac"
    )

    piv_nc = search_df.pivot_table(
        index="min_cluster_size",
        columns="min_center_dist",
        values="n_clusters"
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.heatmap(piv_sil, annot=True, cmap="viridis", ax=axes[0])
    axes[0].set_title("Silhouette Score")

    sns.heatmap(piv_noise, annot=True, cmap="magma_r", ax=axes[1])
    axes[1].set_title("Noise Fraction")

    sns.heatmap(piv_nc, annot=True, cmap="coolwarm", ax=axes[2])
    axes[2].set_title("Number of Clusters")

    plt.tight_layout()
    plt.show()

    # ---- Scatter plots ----
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].scatter(search_df["n_clusters"], search_df["silhouette"], c=search_df["noise_frac"], cmap="viridis")
    ax[0].set_xlabel("Number of clusters")
    ax[0].set_ylabel("Silhouette")
    ax[0].set_title("Silhouette vs Number of Clusters")
    cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
    cbar.set_label("Noise fraction")

    ax[1].scatter(search_df["min_center_dist"], search_df["silhouette"], c=search_df["min_cluster_size"], cmap="plasma")
    ax[1].set_xlabel("Min center distance [m]")
    ax[1].set_ylabel("Silhouette")
    ax[1].set_title("Silhouette vs Min Center Distance")
    cbar = plt.colorbar(ax[1].collections[0], ax=ax[1])
    cbar.set_label("Min cluster size")

    plt.tight_layout()
    plt.show()


def analyze_dbscan_results(search_df):
    """
    Produce comparative visualisations for DBSCAN parameter search.
    """

    # Pivot tables for heatmaps
    piv_sil = search_df.pivot_table(
        index="min_samples",
        columns="eps_m",
        values="silhouette"
    )

    piv_noise = search_df.pivot_table(
        index="min_samples",
        columns="eps_m",
        values="noise_frac"
    )

    piv_nc = search_df.pivot_table(
        index="min_samples",
        columns="eps_m",
        values="n_clusters"
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.heatmap(piv_sil, annot=True, cmap="viridis", ax=axes[0])
    axes[0].set_title("Silhouette Score")

    sns.heatmap(piv_noise, annot=True, cmap="magma_r", ax=axes[1])
    axes[1].set_title("Noise Fraction")

    sns.heatmap(piv_nc, annot=True, cmap="coolwarm", ax=axes[2])
    axes[2].set_title("Number of Clusters")

    plt.tight_layout()
    plt.show()

    # ---- Scatter plots ----
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Silhouette vs number of clusters
    ax[0].scatter(
        search_df["n_clusters"],
        search_df["silhouette"],
        c=search_df["noise_frac"],
        cmap="viridis"
    )
    ax[0].set_xlabel("Number of clusters")
    ax[0].set_ylabel("Silhouette")
    ax[0].set_title("Silhouette vs Number of Clusters")
    cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
    cbar.set_label("Noise fraction")

    # Silhouette vs eps
    ax[1].scatter(
        search_df["eps_m"],
        search_df["silhouette"],
        c=search_df["min_samples"],
        cmap="plasma"
    )
    ax[1].set_xlabel("eps [m]")
    ax[1].set_ylabel("Silhouette")
    ax[1].set_title("Silhouette vs eps")
    cbar = plt.colorbar(ax[1].collections[0], ax=ax[1])
    cbar.set_label("min_samples")

    plt.tight_layout()
    plt.show()


def get_best_and_test(X_test, search_df, X_train, version=""):
    # ---- Select best parameters ----
    best_row = search_df.loc[search_df["silhouette"].idxmax()]
    best_ms = int(best_row["min_samples"])
    best_eps = float(best_row["eps_m"])
    print("Best parameters (train set):")
    print(best_row)
    labels_train, nc_train, noise_train, sil_train = run_dbscan(X_train, best_eps, best_ms)
    visualize_clusters_folium(X_train, labels_train, map_path=f"dbscan_train_clusters_{version}.html")
    print("\nEvaluation on train set:")
    print(f"Clusters: {nc_train}")
    print(f"Noise fraction: {noise_train:.3f}")
    print(f"Silhouette: {sil_train:.3f}")

    # ---- Final evaluation on TEST ----
    labels_test, nc_test, noise_test, sil_test = run_dbscan(X_test, best_eps, best_ms)
    print("\nEvaluation on held-out test set:")
    print(f"Clusters: {nc_test}")
    print(f"Noise fraction: {noise_test:.3f}")
    print(f"Silhouette: {sil_test:.3f}")
    print(search_df.sort_values("silhouette", ascending=False).head())
    # visualize_clusters_folium(X_test, labels_test, map_path=f"dbscan_test_clusters_{version}.html")
    return best_row, nc_test, noise_test, sil_test

def compare_hdbscan_dbscan(hdb_df, db_df):
    """
    Combined comparative analysis of HDBSCAN vs DBSCAN search results.
    Produces:
      - Silhouette comparison
      - Noise comparison
      - Cluster count comparison
      - Pareto-style scatter plot
      - Summary of best models
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Add method labels
    hdb = hdb_df.copy()
    hdb["method"] = "HDBSCAN"

    db = db_df.copy()
    db["method"] = "DBSCAN"

    combined = pd.concat([hdb, db], ignore_index=True)

    # -----------------------------
    # 1. Silhouette comparison
    # -----------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined, x="method", y="silhouette", palette="viridis")
    sns.stripplot(data=combined, x="method", y="silhouette", color="black", alpha=0.4)
    plt.title("Silhouette Score Comparison: HDBSCAN vs DBSCAN")
    plt.show()

    # -----------------------------
    # 2. Noise fraction comparison
    # -----------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined, x="method", y="noise_frac", palette="magma")
    sns.stripplot(data=combined, x="method", y="noise_frac", color="black", alpha=0.4)
    plt.title("Noise Fraction Comparison: HDBSCAN vs DBSCAN")
    plt.show()

    # -----------------------------
    # 3. Number of clusters comparison
    # -----------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined, x="method", y="n_clusters", palette="coolwarm")
    sns.stripplot(data=combined, x="method", y="n_clusters", color="black", alpha=0.4)
    plt.title("Number of Clusters Comparison: HDBSCAN vs DBSCAN")
    plt.show()

    # -----------------------------
    # 4. Pareto-style scatter plot
    # -----------------------------
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=combined,
        x="noise_frac",
        y="silhouette",
        hue="method",
        size="n_clusters",
        sizes=(40, 300),
        alpha=0.8,
        palette="Set2"
    )
    plt.xlabel("Noise Fraction")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette vs Noise Fraction (Pareto Comparison)")
    plt.legend()
    plt.show()

    # -----------------------------
    # 5. Best models summary
    # -----------------------------
    best_hdb = hdb.loc[hdb["silhouette"].idxmax()]
    best_db = db.loc[db["silhouette"].idxmax()]

    summary = pd.DataFrame([
        {
            "method": "HDBSCAN",
            "silhouette": best_hdb["silhouette"],
            "noise_frac": best_hdb["noise_frac"],
            "n_clusters": best_hdb["n_clusters"],
            "params": {
                "min_cluster_size": best_hdb.get("min_cluster_size"),
                "min_center_dist": best_hdb.get("min_center_dist")
            }
        },
        {
            "method": "DBSCAN",
            "silhouette": best_db["silhouette"],
            "noise_frac": best_db["noise_frac"],
            "n_clusters": best_db["n_clusters"],
            "params": {
                "min_samples": best_db.get("min_samples"),
                "eps_m": best_db.get("eps_m")
            }
        }
    ])

    print("\n===== BEST MODELS SUMMARY =====")
    print(summary)

    return summary


if __name__ == '__main__':
    main()
    # print(search_df.sort_values("silhouette", ascending=False).head())
