import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from shapely import wkt
import geopandas as gpd

PRESET_ID = [0, 1, 3, 5, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 24, 25, 26, 29, 31]
DATA_PATH= "data/data_2024-11-17.csv"

from data_processing import (
    load_and_preprocess_data, rebuild_voronoi, get_od_flows, 
    extract_top_routes, calculate_od_matrix, normalize_matrix
)
from visualization import plot_dynamic_voronoi, plot_od_heatmap, build_segment_map

st.set_page_config(page_title="Interactive OD Dashboard", layout="wide")

@st.cache_data
def get_cached_data():
    return load_and_preprocess_data()

@st.cache_data
def load_segments_gdf(path: str = DATA_PATH) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.rename(columns={"segmnet_id": "segment_id"})

    segments_daily = (
        df.groupby(["segment_id", "wkt"], as_index=False)
        .agg(
            avg_speed_day=("avg_speed", "mean"),
            median_speed_day=("avg_speed", "median"),
            observations=("avg_speed", "size"),
        )
    )

    segments_daily["geometry"] = segments_daily["wkt"].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(
        segments_daily,
        geometry="geometry",
        crs="EPSG:4326",
    )

    return gdf

#st.title("🗺️ Interactive OD & Voronoi Dashboard")
st.title("🗺️ Interactive Traffic Matrix Generator")

segments_gdf = load_segments_gdf()

with st.spinner("Loading and preprocessing data..."):
    try:
        trip_stats_filtered, cluster_stats, clip_geom = get_cached_data()
    except Exception as e:
        st.error(f"Failed to load CSV files. Error: {e}")
        st.stop()

if "node_editor_df" not in st.session_state:
    df_nodes = cluster_stats[["endpoint_cluster", "terminal_id", "events"]].copy()
    preset_clusters = PRESET_ID
    df_nodes["Active"] = df_nodes["endpoint_cluster"].isin(preset_clusters)
    st.session_state.node_editor_df = df_nodes

if "generated" not in st.session_state:
    st.session_state.generated = False

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration Parameters")
    
    st.markdown("**Terminal Selection (Nodes):**")
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    if col_sel1.button("✅ Select All", use_container_width=True):
        st.session_state.node_editor_df["Active"] = True
        st.rerun()
    if col_sel2.button("❌ Deselect All", use_container_width=True):
        st.session_state.node_editor_df["Active"] = False
        st.rerun()
    if col_sel3.button("🎯 Preset Set", use_container_width=True):
        preset_clusters = PRESET_ID
        st.session_state.node_editor_df["Active"] = st.session_state.node_editor_df["endpoint_cluster"].isin(preset_clusters)
        st.rerun()
        
    with st.form("settings_form"):
        edited_nodes = st.data_editor(
            st.session_state.node_editor_df,
            column_config={
                "Active": st.column_config.CheckboxColumn("Active", default=False),
                "events": st.column_config.NumberColumn("Total Trips", disabled=True),
                "terminal_id": st.column_config.TextColumn("Terminal ID", disabled=True),
                "endpoint_cluster": None
            },
            hide_index=True,
            use_container_width=True
        )
        
        top_n = st.slider(
            "Number of top routes (Top N lines):",
            min_value=0, max_value=50, value=10, step=1
        )
        
        norm_method = st.radio(
            "OD Matrix Normalization:",
            options=["None", "Row", "Column"],
            index=1,
            help="Choose how to normalize the heatmap values."
        )

        region_all = st.radio(
            "Draw all regions or active only:",
            options=["all", "active only"],
            index=1,
            help="Choose whether recalculate regions to show active only division."
        )

        btn_generate = st.form_submit_button("🚀 Generate / Refresh Dashboard", type="primary", use_container_width=True)
    
    if btn_generate:
        st.session_state.generated = True
        st.session_state.node_editor_df = edited_nodes 

active_clusters = edited_nodes[edited_nodes["Active"]]["endpoint_cluster"].tolist()
active_terminal_ids = edited_nodes[edited_nodes["Active"]]["terminal_id"].tolist()

# ==========================================
# MAIN VIEW WIDE LAYOUT
# ==========================================
if st.session_state.generated:
    if len(active_clusters) < 3:
        st.error("Please select at least 3 active terminals in the sidebar to correctly compute Voronoi regions!")
    else:
        with st.spinner("Recalculating Voronoi grid and traffic flows..."):
            if region_all == "all":
                vor_endpoints=edited_nodes["endpoint_cluster"].tolist()
            else:
                vor_endpoints=active_clusters
            vor_dyn = rebuild_voronoi(vor_endpoints, cluster_stats, clip_geom)
            flows = get_od_flows(vor_dyn, trip_stats_filtered)
            
            top_flows_df = extract_top_routes(flows, top_n)
            term_coords = vor_dyn.set_index('terminal_id')[['lat', 'lon']].to_dict('index')
            
            od_matrix = calculate_od_matrix(flows, active_terminal_ids)
            od_matrix_disp = normalize_matrix(od_matrix, norm_method)
            
        # --- 1. DWIE MAPY OBOK SIEBIE (GÓRA) ---
        col_map_vor, col_map_speed = st.columns(2, gap="medium")
        
        with col_map_vor:
            st.subheader("Terminal Zones & Primary Routes")
            m = plot_dynamic_voronoi(vor_dyn, top_flows_df, term_coords, active_clusters, segments_gdf)
            st_folium(m, width="stretch", height=600, returned_objects=[])

        with col_map_speed:
            st.subheader("Median Speed map")
            segment_map = build_segment_map(segments_gdf)
            st_folium(segment_map, width="stretch", height=600, returned_objects=[])

        st.divider()

        # --- 2. MACIERZ I TABELA POPULARNYCH DRÓG (ŚRODEK) ---
        col_matrix, col_table = st.columns([1.2, 1], gap="medium")
        
        with col_matrix:
            st.subheader("Origin-Destination Matrix")
            fig = plot_od_heatmap(od_matrix_disp, norm_method)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        with col_table:
            st.subheader(f"🏆 Top {top_n} Most Frequent Routes")
            if not top_flows_df.empty:
                st.dataframe(
                    top_flows_df.rename(columns={'term1': 'Terminal A', 'term2': 'Terminal B', 'trips': 'Number of Trips'}),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No route data found for current selection.")

        st.divider()

        # --- 3. EKSPORT NA SAMYM DOLE ---
        st.subheader("📥 Data Export")
        col_exp1, col_exp2, _ = st.columns([1, 1, 2])
        
        csv_data = od_matrix.to_csv().encode('utf-8')
        geojson_data = vor_dyn.to_json().encode('utf-8')
        
        with col_exp1:
            st.download_button(
                label="📄 Download Raw OD Matrix (CSV)",
                data=csv_data,
                file_name="od_matrix.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            st.download_button(
                label="🗺️ Download Voronoi Polygons (GeoJSON)",
                data=geojson_data,
                file_name="voronoi_regions.geojson",
                mime="application/json",
                use_container_width=True
            )

else:
    st.info("💡 Adjust parameters in the left sidebar and click 'Generate / Refresh Dashboard' to visualize results.")