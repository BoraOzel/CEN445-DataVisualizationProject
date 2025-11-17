import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Global Cyber Security Threats Dashboard",
    page_icon="🛡️", 
    layout="wide"
)

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"'{file_path}' dosyası bulunamadı. Lütfen 'app.py' ile aynı klasörde olduğundan emin olun.")
        return None

df = load_data('Global_Cybersecurity_Threats_2015-2024.csv')

if df is None:
    st.stop()

st.sidebar.header("Dashboard Filters")

all_attack_types = df['Attack Type'].unique()
selected_attack_types = st.sidebar.multiselect(
    "Select Attack Type:",
    options=all_attack_types,
    default=all_attack_types 
)

min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
selected_year_range = st.sidebar.slider(
"Yıl Aralığı Seçin:",
min_value=min_year,
max_value=max_year,
value=(min_year, max_year)
)  

all_industries = ['All'] + list(df['Target Industry'].unique())
selected_industry = st.sidebar.selectbox(
    "Select Target Industry:",
    options=all_industries,
    index=0  
)

df_filtered = df[df['Attack Type'].isin(selected_attack_types)]

df_filtered = df_filtered[
    (df_filtered['Year'] >= selected_year_range[0]) &
    (df_filtered['Year'] <= selected_year_range[1])
]

if selected_industry != 'All':
    df_filtered = df_filtered[df_filtered['Target Industry'] == selected_industry]


st.title("🛡️ Global Cyber Security Threats Analyze Dashboard")
st.markdown("This dashboard was prepared to analyze global cyber security threats between 2015 and 2024.")

st.header("Example: Total Financial Loss by Attack Type")

if df_filtered.empty:
    st.warning("No data matching to the selected filters.")
else:
    loss_by_attack_type = df_filtered.groupby('Attack Type')['Financial Loss (in Million $)'].sum().reset_index()
    
    fig_loss = px.bar(
        loss_by_attack_type.sort_values(by='Financial Loss (in Million $)', ascending=False),
        x='Attack Type',
        y='Financial Loss (in Million $)',
        title='Total Financial Loss by Attack Type (in Million $)',
        labels={'Financial Loss (in Million $)': 'Financial Loss (in Million $)', 'Attack Type': 'Attack Type'},
        template='plotly_white'
    )

    st.plotly_chart(fig_loss, use_container_width=True)

    st.subheader("Unfiltered Raw Data (First 100 Rows)")
    st.dataframe(df_filtered.head(100))