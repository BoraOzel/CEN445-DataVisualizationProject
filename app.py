import streamlit as st
import pandas as pd
import plotly.express as px
import graphs
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

#---------------------------------------------
# Machine Learning Model Training
@st.cache_resource
def train_prediction_model(df):
    feature_cols = ['Country', 'Attack Type', 'Target Industry', 'Attack Source',
                    'Security Vulnerability Type', 'Defense Mechanism Used']
    target_col = 'Financial Loss (in Million $)'

    X = df[feature_cols].copy()
    y = df[target_col]

    # Encoding
    encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, encoders
#---------------------------------------------
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

# Country Filter
all_countries = sorted(df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Country:",
    options=all_countries,
    default=all_countries
)

# Attack Type Filter
all_attack_types = sorted(df['Attack Type'].unique())
selected_attack_types = st.sidebar.multiselect(
    "Select Attack Type:",
    options=all_attack_types,
    default=all_attack_types
)

# Attack Source Filter
all_attack_sources = sorted(df['Attack Source'].unique())
selected_attack_sources = st.sidebar.multiselect(
    "Select Attack Source:",
    options=all_attack_sources,
    default=all_attack_sources
)

# Security Vulnerability Type Filter
all_vuln = sorted(df['Security Vulnerability Type'].unique())
selected_vuln = st.sidebar.multiselect(
    "Select Security Vulnerability:",
    options=all_vuln,
    default=all_vuln
)

# Defense Mechanism Filter
all_defense = sorted(df['Defense Mechanism Used'].unique())
selected_defense = st.sidebar.multiselect(
    "Select Defense Mechanism:",
    options=all_defense,
    default=all_defense
)

# Target Industry Filter
all_industries = sorted(df['Target Industry'].unique())
selected_industry = st.sidebar.multiselect(
    "Select Target Industry:",
    options=all_industries,
    default=all_industries
)

# Incident Resolution Time Filter
min_res = int(df["Incident Resolution Time (in Hours)"].min())
max_res = int(df["Incident Resolution Time (in Hours)"].max())
selected_res_range = st.sidebar.slider(
    "Incident Resolution Time (Hours):",
    min_value=min_res,
    max_value=max_res,
    value=(min_res, max_res)
)

# Year Filter
min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
selected_year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

st.sidebar.divider()

# Sankey Diagram Settings
st.sidebar.subheader("Sankey Diagram Settings")

available_options = [
    'Country',
    'Year',
    'Attack Type',
    'Target Industry',
    'Attack Source',
    'Security Vulnerability Type',
    'Defense Mechanism Used'
]

st.sidebar.markdown("**Select Flow Order (1st → 2nd → 3rd → 4th):**")

dimension_1 = st.sidebar.selectbox("1st Dimension:", available_options, index=2)
remaining_1 = [x for x in available_options if x != dimension_1]

dimension_2 = st.sidebar.selectbox("2nd Dimension:", remaining_1, index=2 if 'Target Industry' in remaining_1 else 0)
remaining_2 = [x for x in remaining_1 if x != dimension_2]

dimension_3 = st.sidebar.selectbox("3rd Dimension:", remaining_2,
                                   index=4 if 'Security Vulnerability Type' in remaining_2 else 0)
remaining_3 = [x for x in remaining_2 if x != dimension_3]

dimension_4 = st.sidebar.selectbox("4th Dimension (optional):", ['None'] + remaining_3, index=0)

dimension_order = [dimension_1, dimension_2, dimension_3]
if dimension_4 != 'None':
    dimension_order.append(dimension_4)

sankey_sample_size = st.sidebar.slider(
    "Sample Size:",
    min_value=100,
    max_value=3000,
    value=500,
    step=100
)
# ----Violin Plot Settings---
st.sidebar.divider()
st.sidebar.subheader("Violin Plot Settings")

violin_x_options = [
    'Country',
    'Attack Type',
    'Target Industry',
    'Attack Source',
    'Security Vulnerability Type',
    'Defense Mechanism Used',
    'Year'
]

violin_x = st.sidebar.selectbox(
    "X-axis (Category):",
    violin_x_options,
    index=1
)

violin_y_options = [
    'Financial Loss (in Million $)',
    'Number of Affected Users',
    'Incident Resolution Time (in Hours)'
]

violin_y = st.sidebar.selectbox(
    "Y-axis (Numeric):",
    violin_y_options
)

available_colors = ['None'] + [col for col in violin_x_options if col != violin_x]
violin_color = st.sidebar.selectbox(
    "Split by (Color - Optional):",
    available_colors,
    index=0
)

violin_points = st.sidebar.selectbox(
    "Show Points:",
    ["all", "outliers", "none"],
    index=1
)

violin_sample_size = st.sidebar.slider(
    "Violin Sample Size:",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100
)

# --- APPLY FILTERS ---
df_filtered = df.copy()

# Country
if selected_countries:
    df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]

# Attack type
if selected_attack_types:
    df_filtered = df_filtered[df_filtered['Attack Type'].isin(selected_attack_types)]

# Attack Source
if selected_attack_sources:
    df_filtered = df_filtered[df_filtered['Attack Source'].isin(selected_attack_sources)]

# Vulnerability
if selected_vuln:
    df_filtered = df_filtered[df_filtered['Security Vulnerability Type'].isin(selected_vuln)]

# Defense Mechanism
if selected_defense:
    df_filtered = df_filtered[df_filtered['Defense Mechanism Used'].isin(selected_defense)]

# Target industry
if selected_industry:
    df_filtered = df_filtered[df_filtered['Target Industry'].isin(selected_industry)]

# Resolution Time
df_filtered = df_filtered[
    (df_filtered['Incident Resolution Time (in Hours)'] >= selected_res_range[0]) &
    (df_filtered['Incident Resolution Time (in Hours)'] <= selected_res_range[1])
    ]

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


    # --- 7. TREEMAP ---
    st.divider()
    st.subheader("Vulnerability-Attack-Industry Breakdown")
    fig_treemap = graphs.create_treemap(df_filtered)
    st.plotly_chart(fig_treemap, use_container_width=True)

    # --- 8. SANKEY DIAGRAM ---
    st.divider()
    st.subheader(" Attack Flow Analysis (Sankey Diagram)")

    _, _, _ = graphs.create_sankey_diagram(
        df_filtered,
        dimension_order,
        sankey_sample_size
    )

    st.markdown("###  Path Highlighting")

    dimension_values = {
        dim: ['Any'] + sorted(df_filtered[dim].unique().tolist())
        for dim in dimension_order
    }

    # select kutusu oluşturan kısım
    cols = st.columns(len(dimension_order))
    selected_path = []

    for idx, dim in enumerate(dimension_order):
        with cols[idx]:
            selected = st.selectbox(
                f"{dim}",
                options=dimension_values[dim],
                index=0,
                key=f"path_dim_{idx}"
            )
            selected_path.append(selected if selected != 'Any' else None)

    # boyut ve yol arasındaki uyuşmazlığı düzeltir
    if len(selected_path) > len(dimension_order):
        selected_path = selected_path[:len(dimension_order)]

    has_selection = any(v is not None for v in selected_path)

    fig_sankey, _, stats = graphs.create_sankey_diagram(
        df_filtered,
        dimension_order,
        sankey_sample_size,
        highlight_path=selected_path if has_selection else None
    )

    # seçilen pathin özellikleri
    if has_selection:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="📊 Total Incidents in Path",
                value=f"{stats['total_incidents']:,}"
            )
        with col2:
            st.metric(
                label="Total Affected Users",
                value=f"{stats['total_affected_users']:,}"
            )

    st.plotly_chart(fig_sankey, use_container_width=True)

    # --- 9. VIOLIN PLOT ---
    st.divider()
    st.subheader("Interactive Violin Plot")

    st.markdown("""
        A violin plot shows the distribution of data across different categories.
        Use X (category) and Y (numeric) selections to explore distributions interactively.
        """)

    df_violin = df_filtered.sample(
        n=min(violin_sample_size, len(df_filtered)),
        random_state=42
    )
# dataları seçmek için outliers yada tğm dataları göstertebilir
    points_val = False
    if violin_points == "all":
        points_val = "all"
    elif violin_points == "outliers":
        points_val = "outliers"

    fig_violin = graphs.create_violin_plot(
        df_violin,
        x_column=violin_x,
        y_column=violin_y,
        color_column=violin_color,
        points=points_val,

    )

    st.plotly_chart(fig_violin, use_container_width=True)

#-------------10. REGRESSİON------

st.divider()
st.header("Prediction Module")

model, encoders = train_prediction_model(df)

col1, col2, col3 = st.columns(3)

with col1:
    pred_country = st.selectbox("Country", sorted(df['Country'].unique()))
    pred_industry = st.selectbox("Target Industry", sorted(df['Target Industry'].unique()))

with col2:
    pred_attack = st.selectbox("Attack Type", sorted(df['Attack Type'].unique()))
    pred_source = st.selectbox("Attack Source", sorted(df['Attack Source'].unique()))

with col3:
    pred_vuln = st.selectbox("Vulnerability", sorted(df['Security Vulnerability Type'].unique()))
    pred_defense = st.selectbox("Defense Mechanism", sorted(df['Defense Mechanism Used'].unique()))

if st.button("Predict"):
    # input hazırlama
    input_data = pd.DataFrame({
        'Country': [pred_country],
        'Attack Type': [pred_attack],
        'Target Industry': [pred_industry],
        'Attack Source': [pred_source],
        'Security Vulnerability Type': [pred_vuln],
        'Defense Mechanism Used': [pred_defense]
    })

    # şifreleme ve çözme
    encoded_input = input_data.copy()
    for col in input_data.columns:
        encoded_input[col] = encoders[col].transform(input_data[col])


    predicted_loss = model.predict(encoded_input)[0]
    st.subheader(f"Predicted Financial Loss: ${predicted_loss:.2f} Million")

    # optimizasyon ayarları(daha iyi bir mekanizma var mı )
    st.write("Optimization Check...")

    best_defense = pred_defense
    min_loss = predicted_loss

    all_defenses = df['Defense Mechanism Used'].unique()

    comparison_data = []

    for defense in all_defenses:
        temp_input = encoded_input.copy()
        temp_input['Defense Mechanism Used'] = encoders['Defense Mechanism Used'].transform([defense])
        temp_loss = model.predict(temp_input)[0]

        comparison_data.append({'Defense': defense, 'Predicted Loss': temp_loss})

        if temp_loss < min_loss:
            min_loss = temp_loss
            best_defense = defense

    if best_defense != pred_defense:
        saving = predicted_loss - min_loss
        st.write(
            f"Recommendation: Switch to '{best_defense}' to reduce loss to ${min_loss:.2f} M. (Saving: ${saving:.2f} M)")
    else:
        st.write(f"Current defense '{pred_defense}' is optimal.")

    # diğer defans mekanizmalarına göre karşılaştırma (çıkarılabiliriz)
    df_comp = pd.DataFrame(comparison_data).sort_values(by='Predicted Loss')
    fig_opt = px.bar(
        df_comp,
        x='Predicted Loss',
        y='Defense',
        orientation='h',
        title="Impact of Defense Mechanisms",
        color='Predicted Loss',
        text_auto='.2f'
    )
    st.plotly_chart(fig_opt, use_container_width=True)
