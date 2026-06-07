import hdbscan
import pandas as pd
import gpd as gpd
import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from pathlib import Path

# from codes.dbscan_analysis import enforce_min_center_distance

DATA_FILE = Path("data/2024_11_17_17_00_processed.csv")
SEGMENTS_FILE = Path("data/data_2024-11-17.csv")

PARAMS = {
    "time_gap_new_trip_s": 5 * 60,
    "max_plausible_speed_kmh": 180,
    "stop_speed_kmh": 3,
    "long_stop_s": 3 * 60,
    "endpoint_cluster_eps_m": 150,
    "endpoint_cluster_min_samples": 8,
    "min_trip_points": 4,
    "min_trip_duration_s": 30,
    "min_trip_distance_m": 200,
    "car_min_p90_speed_kmh": 45,
    "car_min_max_speed_kmh": 60,
    "voronoi_clip_rotation_deg": 45,
    "voronoi_clip_padding_m": 600,
    "min_cluster_size": 15,
    "min_center_dist": 500
}
EARTH_R = 6_371_000  # promień Ziemi [m]

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

def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_m * np.arcsin(np.sqrt(a))

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi requires 2D points")
    new_regions, new_vertices = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None: radius = np.ptp(vor.points, axis=0).max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0: v1, v2 = v2, v1
            if v1 >= 0: continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            new_region.append(len(new_vertices))
            new_vertices.append((vor.vertices[v2] + direction * radius).tolist())
        vertices_arr = np.asarray([new_vertices[v] for v in new_region])
        centroid = vertices_arr.mean(axis=0)
        angles = np.arctan2(vertices_arr[:, 1] - centroid[1], vertices_arr[:, 0] - centroid[0])
        new_regions.append([v for _, v in sorted(zip(angles, new_region))])
    return new_regions, np.asarray(new_vertices)


def find_clusters(coords_rad):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=PARAMS["min_cluster_size"],
        metric="haversine",
    )
    labels = clusterer.fit_predict(coords_rad)

    # post-processing: enforce minimum center spacing
    labels = enforce_min_center_distance(coords_rad, labels, PARAMS["min_center_dist"])
    return labels


def load_and_preprocess_data():
    df_raw = pd.read_csv(DATA_FILE, parse_dates=["timestamp", "time"])
    df_raw = df_raw.sort_values(["vehicle_id", "timestamp"]).copy()

    segments_raw = pd.read_csv(SEGMENTS_FILE, parse_dates=["time"])
    segments_raw = segments_raw.rename(columns={"segmnet_id": "segment_id"})
    segments_daily = segments_raw.groupby(["segment_id", "wkt"], as_index=False).agg(avg_speed_day=("avg_speed", "mean"))
    segments_daily["geometry"] = segments_daily["wkt"].apply(wkt.loads)
    segments_gdf = gpd.GeoDataFrame(segments_daily, geometry="geometry", crs="EPSG:4326")

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
    df["stationary_block"] = df.groupby("vehicle_id")["is_stationary_point"].transform(lambda s: s.ne(s.shift()).cumsum())
    
    stop_blocks = df[df["is_stationary_point"]].groupby(["vehicle_id", "stationary_block"]).agg(stop_start=("timestamp", "min"), stop_end=("timestamp", "max")).reset_index()
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
        reported_speed_max=("speed", "max"), distance_m=("distance_m", "sum"), implausible_jumps=("is_implausible_jump", "sum"),
    ).reset_index()
    
    trip_stats["duration_s"] = (trip_stats["end_time"] - trip_stats["start_time"]).dt.total_seconds()
    trip_stats["valid_basic"] = ~(trip_stats["points"].lt(PARAMS["min_trip_points"]) | trip_stats["duration_s"].lt(PARAMS["min_trip_duration_s"]) | trip_stats["distance_m"].lt(PARAMS["min_trip_distance_m"]) | trip_stats["implausible_jumps"].gt(0))
    
    car_like = trip_stats["reported_speed_p90"].ge(PARAMS["car_min_p90_speed_kmh"]) | trip_stats["reported_speed_max"].ge(PARAMS["car_min_max_speed_kmh"])
    trip_stats["mode_class"] = np.select([car_like], ["car_like"], default="other")
    trip_stats_filtered = trip_stats[trip_stats["valid_basic"] & trip_stats["mode_class"].eq("car_like")].copy()

    start_points = trip_stats_filtered[["trip_uid", "start_lat", "start_lon"]].rename(columns={"start_lat": "lat", "start_lon": "lon"})
    start_points["endpoint_type"] = "origin"
    end_points = trip_stats_filtered[["trip_uid", "end_lat", "end_lon"]].rename(columns={"end_lat": "lat", "end_lon": "lon"})
    end_points["endpoint_type"] = "destination"
    endpoints = pd.concat([start_points, end_points], ignore_index=True).dropna(subset=["lat", "lon"])

    # eps_rad = PARAMS["endpoint_cluster_eps_m"] / 6_371_000
    coords_rad = np.radians(endpoints[["lat", "lon"]].to_numpy())
    # clusterer = DBSCAN(eps=eps_rad, min_samples=PARAMS["endpoint_cluster_min_samples"], metric="haversine")
    endpoints["endpoint_cluster"] = find_clusters(coords_rad)

    cluster_stats = endpoints[endpoints["endpoint_cluster"] >= 0].groupby("endpoint_cluster").agg(
        lat=("lat", "mean"), lon=("lon", "mean"), events=("trip_uid", "size"),
        origins=("endpoint_type", lambda s: (s == "origin").sum()), destinations=("endpoint_type", lambda s: (s == "destination").sum())
    ).reset_index()
    
    cluster_stats = cluster_stats.sort_values("events", ascending=False).reset_index(drop=True)
    cluster_stats["terminal_id"] = "T" + cluster_stats["endpoint_cluster"].astype(str)

    base_terminals_2180 = gpd.GeoDataFrame(cluster_stats, geometry=gpd.points_from_xy(cluster_stats["lon"], cluster_stats["lat"]), crs="EPSG:4326").to_crs(epsg=2180)
    segments_2180 = segments_gdf.to_crs(epsg=2180)
    clip_source_geom = base_terminals_2180.geometry.union_all().union(segments_2180.geometry.union_all())
    center = clip_source_geom.centroid
    rotated_source = rotate(clip_source_geom, -PARAMS["voronoi_clip_rotation_deg"], origin=center)
    minx, miny, maxx, maxy = rotated_source.bounds
    pad = PARAMS["voronoi_clip_padding_m"]
    rotated_clip_rect = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
    clip_geom = rotate(rotated_clip_rect, PARAMS["voronoi_clip_rotation_deg"], origin=center)

    return trip_stats_filtered, cluster_stats, clip_geom

def rebuild_voronoi(active_clusters, cluster_stats, clip_geom):
    active_terminals = cluster_stats[cluster_stats["endpoint_cluster"].isin(active_clusters)].copy()
    terminals_gdf = gpd.GeoDataFrame(active_terminals, geometry=gpd.points_from_xy(active_terminals.lon, active_terminals.lat), crs="EPSG:4326")
    terminals_2180 = terminals_gdf.to_crs(epsg=2180)
    points_xy = np.column_stack([terminals_2180.geometry.x, terminals_2180.geometry.y])
    
    vor = Voronoi(points_xy)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=100_000)
    
    polygons = []
    for region in regions:
        polygon = Polygon(vertices[region])
        if not polygon.is_valid: polygon = polygon.buffer(0)
        polygons.append(polygon.intersection(clip_geom))
        
    vor_dyn = terminals_2180.copy()
    vor_dyn["geometry"] = polygons
    vor_dyn = gpd.GeoDataFrame(vor_dyn, geometry="geometry", crs="EPSG:2180")
    vor_dyn = vor_dyn[~vor_dyn.geometry.is_empty].copy()
    vor_dyn["area_km2"] = vor_dyn.geometry.area / 1_000_000
    return vor_dyn.to_crs(epsg=4326)

def get_od_flows(voronoi_dynamic, trips):
    orig_gdf = gpd.GeoDataFrame(trips, geometry=gpd.points_from_xy(trips.start_lon, trips.start_lat), crs="EPSG:4326")
    dest_gdf = gpd.GeoDataFrame(trips, geometry=gpd.points_from_xy(trips.end_lon, trips.end_lat), crs="EPSG:4326")
    orig_assigned = gpd.sjoin(orig_gdf, voronoi_dynamic[["terminal_id", "geometry"]], how="inner", predicate="within")
    dest_assigned = gpd.sjoin(dest_gdf, voronoi_dynamic[["terminal_id", "geometry"]], how="inner", predicate="within")
    od_pairs = orig_assigned[["trip_uid", "terminal_id"]].rename(columns={"terminal_id": "origin_terminal"}).merge(dest_assigned[["trip_uid", "terminal_id"]].rename(columns={"terminal_id": "destination_terminal"}), on="trip_uid")
    return od_pairs.groupby(["origin_terminal", "destination_terminal"]).size().reset_index(name="trips")

def extract_top_routes(flows_df, top_n):
    if flows_df.empty: return pd.DataFrame()
    f = flows_df.copy()
    f['term1'] = np.minimum(f['origin_terminal'], f['destination_terminal'])
    f['term2'] = np.maximum(f['origin_terminal'], f['destination_terminal'])
    bi_flows = f.groupby(['term1', 'term2'])['trips'].sum().reset_index()
    bi_flows = bi_flows[bi_flows['term1'] != bi_flows['term2']]
    return bi_flows.nlargest(top_n, 'trips').reset_index(drop=True)

def calculate_od_matrix(flows_df, active_terminal_ids):
    all_terms = sorted(active_terminal_ids, key=lambda x: int(str(x).replace('T', '')))
    if flows_df.empty:
        return pd.DataFrame(index=all_terms, columns=all_terms).fillna(0)
    
    od_matrix = flows_df.pivot_table(index="origin_terminal", columns="destination_terminal", values="trips", aggfunc='sum').fillna(0)
    od_matrix = od_matrix.reindex(index=all_terms, columns=all_terms, fill_value=0)
    return od_matrix

def normalize_matrix(od_matrix, method="Row"):
    """Normalizes the OD matrix based on the selected method."""
    if method == "Row":
        row_sums = od_matrix.sum(axis=1)
        return od_matrix.div(row_sums.replace(0, 1), axis=0)
    elif method == "Column":
        col_sums = od_matrix.sum(axis=0)
        return od_matrix.div(col_sums.replace(0, 1), axis=1)
    else: # "None"
        return od_matrix