import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import branca.colormap as cm

# Funkcja plot_dynamic_voronoi() pozostaje bez zmian
def plot_dynamic_voronoi(voronoi_dynamic, top_flows, term_coords, active_clusters, segments_gdf=None):
    colormap = cm.LinearColormap(
        ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"], 
        vmin=voronoi_dynamic["events"].min(), 
        vmax=voronoi_dynamic["events"].max()
    )
    colormap.caption = "Number of Trips per Terminal Zone"
    
    m = folium.Map(location=[52.25, 20.98], zoom_start=11, tiles="CartoDB positron")
    
    folium.GeoJson(
        voronoi_dynamic, 
        style_function=lambda f: {"fillColor": colormap(f["properties"]["events"]), "color": "#333333", "weight": 1.2, "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(fields=["terminal_id", "events", "area_km2"])
    ).add_to(m)

    colormap.add_to(m)

    if not top_flows.empty:
        max_trips = top_flows['trips'].max()
        for _, row in top_flows.iterrows():
            t1, t2, w = row['term1'], row['term2'], row['trips']
            c1, c2 = (term_coords[t1]['lat'], term_coords[t1]['lon']), (term_coords[t2]['lat'], term_coords[t2]['lon'])
            folium.PolyLine(locations=[c1, c2], color="#ff2a00", weight=2 + 10 * (w / max_trips), opacity=0.6, tooltip=f"{t1} &harr; {t2}: {w} trips").add_to(m)

    # -------------------------------------------------
    # Draw road segments (black)
    # -------------------------------------------------
    if segments_gdf is not None:
        for _, row in segments_gdf.iterrows():
            geom = row.geometry

            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]

                folium.PolyLine(
                    locations=coords,
                    color="#555555",
                    weight=1,
                    opacity=0.4,
                ).add_to(m)

            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]

                    folium.PolyLine(
                        locations=coords,
                        color="black",
                        weight=1,
                        opacity=0.4,
                    ).add_to(m)

    # print(active_clusters)
    # print(voronoi_dynamic)
    for _, row in voronoi_dynamic.iterrows():
        if row['endpoint_cluster'] in active_clusters:
            folium.Marker([row['lat'], row['lon']], icon=folium.DivIcon(html=f'<div style="font-size:11px;font-weight:bold;color:#111;background:#ffd23f;border:1px solid #111;border-radius:50%;width:22px;height:22px;text-align:center;line-height:20px;">{str(row["terminal_id"]).replace("T","")}</div>')).add_to(m)
    return m

def plot_od_heatmap(od_matrix, norm_method):
    # Szerokość jest ignorowana przez Streamlit (który używa use_container_width=True),
    # ale relatywny stosunek boku X do Y i parametr square określają zachowanie wewnątrz okna
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if od_matrix.empty or (od_matrix.values == 0).all():
        ax.text(0.5, 0.5, "No connection data for the selected terminals", ha='center', va='center')
    else:
        # Dynamiczne ustawienie formatowania i etykiet dla paska
        if norm_method == "None":
            fmt = ".0f" 
            cbar_label = "Total Trips"
        else:
            fmt = ".2f"
            cbar_label = f"{norm_method}-Normalized Flow"

        # square=False jest tutaj kluczowe – pozwala mapie rozciągać się pionowo/poziomo 
        # w zależności od tego, ile przestrzeni daje Streamlit w danym widoku
        sns.heatmap(od_matrix, cmap="mako_r", annot=False, linewidths=0.5, ax=ax, 
                    square=False, 
                    fmt=fmt,
                    cbar_kws={'label': cbar_label, 'shrink': 0.7},
                    xticklabels=True, yticklabels=True)
        
    ax.set_title(f"{norm_method + ' Normalized ' if norm_method != 'None' else 'Raw '}OD Matrix", pad=20)
    ax.set_ylabel("Origin Terminal")
    ax.set_xlabel("Destination Terminal")
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def build_segment_map(
    segments_gdf: gpd.GeoDataFrame,
    center=(52.269922, 20.978376),
    zoom_start: int = 12,
):
    """
    Create a folium map with speed-colored segments.
    """
    m = folium.Map(
        location=list(center),
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    speed_min = float(segments_gdf["median_speed_day"].quantile(0.05))
    speed_max = float(segments_gdf["median_speed_day"].quantile(0.95))

    colormap = cm.LinearColormap(
        ["#d73027", "#fee08b", "#1a9850"],
        vmin=speed_min,
        vmax=speed_max,
    )
    colormap.caption = "Median speed on segments"

    for _, row in segments_gdf.iterrows():
        geom = row.geometry

        coords = [(lat, lon) for lon, lat in geom.coords]

        speed = float(np.clip(row["median_speed_day"], speed_min, speed_max))

        folium.PolyLine(
            locations=coords,
            color=colormap(speed),
            weight=2,
            opacity=0.75,
            tooltip=(
                f"segment: {row.segment_id}<br>"
                f"median speed: {row.median_speed_day:.1f}<br>"
                f"observations: {row.observations}"
            ),
        ).add_to(m)

    colormap.add_to(m)

    return m
