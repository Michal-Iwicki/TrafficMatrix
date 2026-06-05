import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from data_processing import (
    load_and_preprocess_data, rebuild_voronoi, get_od_flows, 
    extract_top_routes, calculate_od_matrix, normalize_matrix
)
from visualization import plot_dynamic_voronoi, plot_od_heatmap

st.set_page_config(page_title="Interactive OD Dashboard", layout="wide")

@st.cache_data
def get_cached_data():
    return load_and_preprocess_data()

st.title("🗺️ Interactive OD & Voronoi Dashboard")

with st.spinner("Loading and preprocessing data..."):
    try:
        trip_stats_filtered, cluster_stats, clip_geom = get_cached_data()
    except Exception as e:
        st.error(f"Failed to load CSV files. Error: {e}")
        st.stop()

if "node_editor_df" not in st.session_state:
    df_nodes = cluster_stats[["endpoint_cluster", "terminal_id", "events"]].copy()
    half_idx = len(df_nodes) // 2
    df_nodes["Active"] = [True] * half_idx + [False] * (len(df_nodes) - half_idx)
    st.session_state.node_editor_df = df_nodes

if "generated" not in st.session_state:
    st.session_state.generated = False

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration Parameters")
    
    st.markdown("**Terminal Selection (Nodes):**")
    
    col_sel1, col_sel2 = st.columns(2)
    if col_sel1.button("✅ Select All", width="stretch"):
        st.session_state.node_editor_df["Active"] = True
        st.rerun()
    if col_sel2.button("❌ Deselect All", width="stretch"):
        st.session_state.node_editor_df["Active"] = False
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
            width="stretch"
        )
        
        top_n = st.slider(
            "Number of top routes (Top N lines):",
            min_value=1, max_value=50, value=10, step=1
        )
        
        norm_method = st.radio(
            "OD Matrix Normalization:",
            options=["None", "Row", "Column"],
            index=1,
            help="Choose how to normalize the heatmap values."
        )

        btn_generate = st.form_submit_button("🚀 Generate / Refresh Dashboard", type="primary", width="stretch")
    
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
            vor_dyn = rebuild_voronoi(active_clusters, cluster_stats, clip_geom)
            flows = get_od_flows(vor_dyn, trip_stats_filtered)
            
            top_flows_df = extract_top_routes(flows, top_n)
            term_coords = vor_dyn.set_index('terminal_id')[['lat', 'lon']].to_dict('index')
            
            od_matrix = calculate_od_matrix(flows, active_terminal_ids)
            od_matrix_disp = normalize_matrix(od_matrix, norm_method)
            
            col_map, col_matrix = st.columns([1.2, 1], gap="medium")
            
            with col_map:
                st.subheader("Voronoi Regions & Top Routes Map")
                m = plot_dynamic_voronoi(vor_dyn, top_flows_df, term_coords)
                # Zamiana parametru dla biblioteki streamlit_folium (zwiększona wysokość)
                st_folium(m, width="stretch", height=750, returned_objects=[])
                
            with col_matrix:
                st.subheader("Origin-Destination Matrix")
                fig = plot_od_heatmap(od_matrix_disp, norm_method)
                # Zamiana parametru w st.pyplot
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                
            st.divider()
            
            col_list, col_export = st.columns([1, 1])
            with col_list:
                st.subheader(f"🏆 Top {top_n} Most Frequent Routes")
                if not top_flows_df.empty:
                    st.dataframe(
                        top_flows_df.rename(columns={'term1': 'Terminal A', 'term2': 'Terminal B', 'trips': 'Number of Trips'}),
                        hide_index=True,
                        width="stretch"
                    )
                else:
                    st.info("No route data found for current selection.")
                    
            with col_export:
                st.subheader("📥 Data Export")
                st.markdown("Download computed data layers reflecting only the currently active terminals.")
                
                csv_data = od_matrix.to_csv().encode('utf-8')
                geojson_data = vor_dyn.to_json().encode('utf-8')
                
                st.download_button(
                    label="📄 Download Raw OD Matrix (CSV)",
                    data=csv_data,
                    file_name="od_matrix.csv",
                    mime="text/csv",
                    width="stretch"
                )
                
                st.download_button(
                    label="🗺️ Download Voronoi Polygons (GeoJSON)",
                    data=geojson_data,
                    file_name="voronoi_regions.geojson",
                    mime="application/json",
                    width="stretch"
                )
else:
    st.info("💡 Adjust parameters in the left sidebar and click 'Generate / Refresh Dashboard' to visualize results.")