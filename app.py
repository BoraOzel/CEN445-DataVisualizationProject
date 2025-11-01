import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Amazon Delivery Analyze Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"{file_path} file can not found. Please be sure the file is in the same folder with 'app.py'.")
        return None

df = load_data('amazon_delivery_cleared.csv')

if df is None:
    st.stop()

# --- Sidebar (Filters) ---
st.sidebar.header("Dashboard Filters")

# Filter 1: Category
all_categories = df['Category'].unique()
selected_categories = st.sidebar.multiselect(
    "Select category:",
    options=all_categories,
    default=all_categories
)

# Filter 2: Agent Rating (Slider)
min_rating, max_rating = float(df['Agent_Rating'].min()), float(df['Agent_Rating'].max())
selected_rating_range = st.sidebar.slider(
    "Select agent rating range:",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating)
)

# Filtre 3: Trafik Status (Selectbox)
all_traffic = ['All'] + list(df['Traffic'].unique())
selected_traffic = st.sidebar.selectbox(
    "Select traffic status:",
    options=all_traffic,
    index=0 
)

# --- Filtered Data ---

# 1. Category filter
df_filtered = df[df['Category'].isin(selected_categories)]

# 2. Rating filter
df_filtered = df_filtered[
    (df_filtered['Agent_Rating'] >= selected_rating_range[0]) &
    (df_filtered['Agent_Rating'] <= selected_rating_range[1])
]

# 3. Traffic filter
if selected_traffic != 'All':
    df_filtered = df_filtered[df_filtered['Traffic'] == selected_traffic]


# --- Home ---
st.title("🚚 Amazon Delivery Data Analyze Dashboard")
st.markdown("This dashboard prepared to analyze Amazon delivery data.")

# --- Example Visualization ---
st.header("Example: Affect of traffic delivery time.")

if df_filtered.empty:
    st.warning("No data mathing the filters you selected was found.")
else:
    avg_time_by_traffic = df_filtered.groupby('Traffic')['Delivery_Time'].mean().reset_index()
    
    # Create the graph (plotly)
    fig_traffic = px.bar(
        avg_time_by_traffic,
        x='Traffic',
        y='Delivery_Time',
        title='Average time of delivery time depending on traffic status (Minutes)',
        labels={'Delivery_Time': 'Average delivery time (min)', 'Traffic': 'Traffic Status'},
        template='plotly_white'
    )

    st.plotly_chart(fig_traffic, use_container_width=True)

    st.subheader("Filtered raw values (First 100 rows)")
    st.dataframe(df_filtered.head(100))