import streamlit as st
import pandas as pd
import graphs  

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="Global Cyber Security Dashboard",
    page_icon="🛡️", 
    layout="wide"
)

# --- DATA LOAD ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"'{file_path}' dosyası bulunamadı. Lütfen dosyanın app.py ile aynı klasörde olduğundan emin olun.")
        return None

df = load_data('Global_Cybersecurity_Threats_2015-2024.csv')

if df is None:
    st.stop()

# --- SIDEBAR (FILTERS) ---
st.sidebar.header("Dashboard Settings")

# 1. Data/Metric Selection
st.sidebar.subheader("Metric Configuration")
map_option = st.sidebar.selectbox(
    "Select Metric for Map & Heatmap:",
    ("Financial Loss (in $B)", "Number of Affected Users")
)

st.sidebar.divider()

# 2. Filters
st.sidebar.subheader("Data Filters")

# Attack Type Filter
all_attack_types = sorted(df['Attack Type'].unique())
selected_attack_types = st.sidebar.multiselect(
    "Select Attack Type:",
    options = all_attack_types,
    default = all_attack_types 
)

# Target Industry Filter
all_industries = sorted(df['Target Industry'].unique())
selected_industry = st.sidebar.multiselect(
    "Select Target Industry:",
    options = all_industries,
    default = all_industries
)

# Year Filter
min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
selected_year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value = min_year,
    max_value = max_year,
    value = (min_year, max_year)
)  

# --- APPLY FILTERS ---
df_filtered = df.copy()

# Attack type
if selected_attack_types:
    df_filtered = df_filtered[df_filtered['Attack Type'].isin(selected_attack_types)]

# Target industry
if selected_industry:
    df_filtered = df_filtered[df_filtered['Target Industry'].isin(selected_industry)]

# Year
df_filtered = df_filtered[
    (df_filtered['Year'] >= selected_year_range[0]) &
    (df_filtered['Year'] <= selected_year_range[1])
]

# --- HOME PAGE ---
st.title("🛡️ Global Cyber Security Threats Analyze")
st.markdown("This dashboard was prepared to analyze global cyber security threats between 2015 and 2024.")

if df_filtered.empty:
    st.warning("⚠️ No data matching for the selected filters. Please expand filters.")
else:
    # Metrik Sütununu Belirle
    if map_option == "Financial Loss (in $B)":
        metric_col = "Financial Loss (in Million $)"
    else:
        metric_col = "Number of Affected Users"

    # --- 1. MAP ---
    st.subheader("🌍 Global Threat Map")
    fig_map = graphs.create_choropleth_map(df_filtered, metric_col, map_option)
    st.plotly_chart(fig_map, use_container_width=True)
    
    total_val = df_filtered[metric_col].sum()
    if "Loss" in metric_col:
        total_val_billion = total_val / 1000
        st.info(f"💰 Total Financial Loss in Selected Period: **${total_val_billion:,.2f} Billion**")
    else:
        st.info(f"👥 Total Affected Users in Selected Period: **{int(total_val):,}**")

    # --- 2. TIME SERIES LINE CHART ---
    st.divider()
    st.subheader("📈 Annual Attack Trends")
    fig_line = graphs.create_time_series_line_chart(df_filtered)
    st.plotly_chart(fig_line, use_container_width=True)

    # --- 3. HEATMAP ---
    st.divider()
    st.subheader(f"🔥 Intensity Heatmap: {map_option}")
    fig_heatmap = graphs.create_heatmap(df_filtered, metric_col, map_option)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- 4. SUNBURST CHART ---
    st.divider()
    st.subheader("🎯 Hierarchical Breakdown (Top 10 Countries)")
    fig_sunburst = graphs.create_sunburst_chart(df_filtered, metric_col, map_option)
    st.plotly_chart(fig_sunburst, use_container_width=True)

    # --- 5. RADAR CHART ---
    st.divider()
    st.subheader("🕸️ Defense Mechanism Performance")
    st.markdown("Evaluating defense mechanisms across 5 key metrics (Score 0-1).")
    fig_radar = graphs.create_radar_multichart(df_filtered)
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- 6. SCATTER PLOT ---
    st.divider()
    st.subheader("🔴 Sector Risk Analysis (Scatter Plot)")
    st.markdown("""
    This chart shows the risk positioning of each industry.
    * **Bubble Size:** Total Number of Incidents
    """)
    fig_scatter = graphs.create_scatter_plot(df_filtered)
    st.plotly_chart(fig_scatter, use_container_width=True)