# DSW-TrafficMatrix

## Project Overview
The goal of this project is to construct a traffic matrix (origin-destination matrix) describing the movement of vehicles between different locations along the S8 expressway in Warsaw and its suburbs. 

Using anonymized location records provided by the Yanosik navigation application, this project extracts vehicle trajectories, identifies meaningful origin and destination zones, and visualizes travel demand. The resulting traffic matrix serves as a foundation for traffic flow analysis, infrastructure planning, and congestion mitigation.

## Key Features
* **Trip Extraction & Filtering:** Segments raw GPS trajectories into valid trips based on time gaps, spatial jumps, and long stops (over 3 minutes).
* **Vehicle Classification:** Filters out bikes and uncertain records using speed-based thresholds to focus exclusively on car trips.
* **Traffic Jam Detection:** Identifies localized congestion using spatial grids and 1-minute time bins to prevent misclassifying traffic jams as trip endpoints.
* **Endpoint Clustering (HDBSCAN):** Groups trip start and end locations into coherent spatial regions using haversine distance and medoid-based center-distance enforcement.
* **POI Integration:** Enhances cluster labels by matching endpoints to the nearest OpenStreetMap Points of Interest (POIs) such as motorway junctions, parking lots, and fuel stations.
* **Region Definition (Voronoi Tessellation):** Partitions the map into dynamic spatial regions based on cluster centers, allowing for manual refinement of active terminal nodes.
* **Interactive Dashboard:** A Streamlit-based web application to dynamically explore spatial boundaries, traffic routes, road segment speeds, and the resulting OD matrix heatmap.

## Setup and Run Instructions

### Prerequisites
Make sure you have Python installed (Python 3.8+ is recommended) and `git` for cloning the repository.

### 1. Clone the Repository
Depending on your GitHub setup, clone the repository using either HTTPS or SSH:

**Using HTTPS:**
```bash
git clone [https://github.com/Michal-Iwicki/TrafficMatrix.git](https://github.com/Michal-Iwicki/TrafficMatrix.git)
cd TrafficMatrix
```

**Using SSH (Recommended if configured):**
```bash
git clone git@github.com:Michal-Iwicki/TrafficMatrix.git
cd TrafficMatrix
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Run the Interactive Dashboard
The main interface for exploring the regional partitioning and the traffic matrix is built with Streamlit. Run the application from your terminal:
```bash
streamlit run run_app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

### 5. Explore Jupyter Notebooks (Optional)
The repository includes several Jupyter Notebooks detailing the Exploratory Data Analysis (EDA), parameter tuning, and clustering methodology:
* `eda_usage_stops.ipynb` – Analysis of application usage time and vehicle stops.
* `trip_extraction_od_matrix.ipynb` – Pipeline for extracting trips and building the initial OD matrix.
* `analiza_dbscan_parametry_FULL.ipynb` – Detailed clustering analysis comparing DBSCAN and HDBSCAN.

To view and interact with these notebooks, start the Jupyter server:
```bash
jupyter notebook
```