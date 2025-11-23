import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# --- 1. CHOROPLETH MAP ---
def create_choropleth_map(df, selected_metric_column, selected_metric_label):
    country_stats = df.groupby('Country').agg({
        selected_metric_column: 'sum',
        'Attack Type': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
    }).reset_index()

    country_stats.rename(columns={'Attack Type': 'Dominant Attack'}, inplace=True)

    plot_col = selected_metric_column
    plot_label = selected_metric_label
    tick_fmt = ",.0f"

    if "Loss" in selected_metric_column:
        plot_col = "Financial Loss (Billion $)"
        country_stats[plot_col] = country_stats[selected_metric_column] / 1000
        plot_label = "Financial Loss ($B)"
        color_scale = px.colors.sequential.Reds
        tick_fmt = ",.1f"
    else:
        color_scale = px.colors.sequential.Oranges

    fig = px.choropleth(
        country_stats,
        locations="Country",
        locationmode='country names',
        color=plot_col,
        hover_name="Country",
        hover_data={'Dominant Attack': True, plot_col: True, 'Country': False},
        color_continuous_scale=color_scale,
        labels={plot_col: plot_label, 'Dominant Attack': 'Dominant Attack'},
        title=f"Total {plot_label} by Country"
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=plot_label, tickformat=tick_fmt)
    )
    return fig


# --- 2. TIME SERIES LINE CHART ---
def create_time_series_line_chart(df):
    trend_data = df.groupby('Year').size().reset_index(name='Incident Count')

    fig = px.line(
        trend_data,
        x='Year',
        y='Incident Count',
        markers=True,
        title='Total Volume of Cyber Attacks (2015-2024)',
        labels={'Year': 'Year', 'Incident Count': 'Total Number of Incidents'}
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(title='Total Number of Incidents'),
        hovermode="x unified",
        showlegend=False
    )
    fig.update_traces(line_color='red', line_width=3)
    return fig


# --- 3. HEATMAP ---
def create_heatmap(df, metric_column, metric_label):
    heatmap_data = df.groupby(['Attack Type', 'Target Industry'])[metric_column].sum().reset_index()

    plot_col = metric_column
    if "Loss" in metric_column:
        plot_col = "Financial Loss ($B)"
        heatmap_data[plot_col] = heatmap_data[metric_column] / 1000
        fmt = ".2f"
        c_scale = "Reds"
        final_label = "Financial Loss ($B)"
    else:
        fmt = ",.0f"
        c_scale = "Oranges"
        final_label = metric_label

    heatmap_pivot = heatmap_data.pivot(index='Attack Type', columns='Target Industry', values=plot_col)

    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Target Industry", y="Attack Type", color=final_label),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        text_auto=fmt,
        aspect="auto",
        color_continuous_scale=c_scale
    )

    fig.update_layout(
        title=f"Heatmap: Attack Type vs. Industry ({final_label})",
        xaxis_title="Target Industry",
        yaxis_title="Attack Type",
        coloraxis_colorbar=dict(title=final_label, tickformat=",.0f" if "Loss" not in metric_column else ".1f")
    )
    return fig


# --- 4. SUNBURST CHART ---
def create_sunburst_chart(df, metric_column, metric_label):
    plot_col = metric_column
    final_label = metric_label
    df_processed = df.copy()

    if "Loss" in metric_column:
        plot_col = "Financial Loss ($B)"
        df_processed[plot_col] = df_processed[metric_column] / 1000
        final_label = "Financial Loss ($B)"

    top_countries = df_processed.groupby('Country')[plot_col].sum().nlargest(10).index
    df_filtered_top = df_processed[df_processed['Country'].isin(top_countries)]

    fig = px.sunburst(
        df_filtered_top,
        path=['Country', 'Target Industry', 'Attack Type'],
        values=plot_col,
        color=plot_col,
        color_continuous_scale="RdBu_r",
        title=f"Hierarchical Analysis: {final_label} (Top 10 Countries)",
        height=700
    )

    fig.update_traces(textinfo="label+percent entry")
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0), coloraxis_colorbar=dict(title=final_label))
    return fig


# --- 5. RADAR CHART ---
def create_radar_multichart(df):
    # 1.  Basic Statistics
    defense_stats = df.groupby('Defense Mechanism Used').agg({
        'Incident Resolution Time (in Hours)': 'mean',
        'Financial Loss (in Million $)': 'mean',
        'Number of Affected Users': 'mean',
        'Attack Type': 'count'
    }).rename(columns={'Attack Type': 'Count_Raw'})

    defense_stats['CostPerUser_Raw'] = defense_stats['Financial Loss (in Million $)'] * 1000000 / defense_stats[
        'Number of Affected Users'].replace(0, 1)
    raw_data = defense_stats.copy()

    # 2. Normalization
    def normalize(series, invert=False):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series.apply(lambda x: 1.0 if invert else 0.0)
        norm = (series - min_val) / (max_val - min_val)
        return 1 - norm if invert else norm

    # 3. Scoring
    scores_df = pd.DataFrame(index=defense_stats.index)
    scores_df['Cost Efficiency'] = normalize(defense_stats['Financial Loss (in Million $)'], invert=True)
    scores_df['Speed Score'] = normalize(defense_stats['Incident Resolution Time (in Hours)'], invert=True)
    scores_df['Cost per User Score'] = normalize(defense_stats['CostPerUser_Raw'], invert=True)
    scores_df['Prevalence'] = normalize(defense_stats['Count_Raw'], invert=False)
    scores_df['User Shielding'] = normalize(defense_stats['Number of Affected Users'], invert=True)

    categories = ['Cost Efficiency', 'Speed Score', 'Cost per User Score', 'Prevalence', 'User Shielding']
    scores_df['Mean Score'] = scores_df[categories].mean(axis=1)
    sorted_defenses = scores_df.sort_values('Mean Score', ascending=True).index.tolist()

    # 4. Layout
    num_items = len(sorted_defenses)
    cols = 3
    rows = (num_items + cols - 1) // cols
    specs = [[{'type': 'polar'} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=sorted_defenses,
        vertical_spacing=0.18,
        horizontal_spacing=0.15
    )

    # Color Scale
    n = len(sorted_defenses)
    if n > 1:
        raw_indices = [i / (n - 1) for i in range(n)]
        adjusted_indices = []
        for val in raw_indices:
            if val <= 0.5:
                adjusted_indices.append(val * 0.8)
            else:
                adjusted_indices.append(0.6 + (val - 0.5) * 0.8)
        colors = px.colors.sample_colorscale("RdBu", adjusted_indices)
    else:
        colors = px.colors.sample_colorscale("RdBu", [0])

    # 5. Draw
    for idx, defense in enumerate(sorted_defenses):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        current_raw_vals = [
            raw_data.loc[defense, 'Financial Loss (in Million $)'],
            raw_data.loc[defense, 'Incident Resolution Time (in Hours)'],
            raw_data.loc[defense, 'CostPerUser_Raw'],
            raw_data.loc[defense, 'Count_Raw'],
            raw_data.loc[defense, 'Number of Affected Users']
        ]

        fig.add_trace(go.Scatterpolar(
            r=scores_df.loc[defense, categories].values,
            theta=categories,
            fill='toself',
            name=defense,
            line_color=colors[idx],
            opacity=0.7,
            customdata=current_raw_vals,
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<br>Raw Value: %{customdata:.1f}<extra></extra>"
        ), row=row, col=col)

    dynamic_height = max(900, rows * 450)
    fig.update_layout(
        title_text="Defense Mechanism Performance (Specific Metrics)",
        height=dynamic_height,
        showlegend=False,
        margin=dict(t=80, b=50, l=60, r=60),
        font=dict(color="black")
    )
    fig.update_polars(
        radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10, color="black", family="Arial Black"),
                        gridcolor="lightgray", linecolor="black"),
        angularaxis=dict(tickfont=dict(size=11, color="black"), linecolor="black"),
        bgcolor="white"
    )
    return fig


# --- 6. BUBBLE CHART  ---
def create_scatter_plot(df):
    """
    Sektör Risk Analizi: Ortalama Finansal Kayıp vs Ortalama Etkilenen Kullanıcı
    Boyut: Olay Sayısı
    """
    # Grouping by industries and averaging
    industry_stats = df.groupby('Target Industry').agg({
        'Financial Loss (in Million $)': 'mean',
        'Number of Affected Users': 'mean',
        'Attack Type': 'count'
    }).reset_index()

    industry_stats.rename(columns={'Attack Type': 'Incident Count'}, inplace=True)

    # Create scatter plot
    fig = px.scatter(
        industry_stats,
        x='Financial Loss (in Million $)',
        y='Number of Affected Users',
        size='Incident Count',  # Bubble size
        color='Target Industry',  # Different color for industries
        hover_name='Target Industry',
        title='Sector Risk Analysis: Financial Impact vs. User Impact',
        labels={
            'Financial Loss (in Million $)': 'Avg. Financial Loss (M$)',
            'Number of Affected Users': 'Avg. Affected Users',
            'Incident Count': 'Total Incidents'
        },
        size_max=60  # Max bubble sizeß
    )

    # Reference lines showing averages (Dashboard average)
    avg_loss = industry_stats['Financial Loss (in Million $)'].mean()
    avg_users = industry_stats['Number of Affected Users'].mean()

    fig.add_vline(x=avg_loss, line_dash="dot", line_color="gray", annotation_text="Avg Loss")
    fig.add_hline(y=avg_users, line_dash="dot", line_color="gray", annotation_text="Avg Users")

    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=True)

    return fig

    # ---------------7.TREEMAP--------------------------


def create_treemap(df):
    vuln_attack_industry = df.groupby(['Security Vulnerability Type', 'Attack Type', 'Target Industry']).agg({
        'Number of Affected Users': 'sum',
        'Financial Loss (in Million $)': 'sum'
    }).reset_index()

    fig = px.treemap(
        vuln_attack_industry,
        path=['Security Vulnerability Type', 'Attack Type', 'Target Industry'],
        values='Number of Affected Users',
        color='Number of Affected Users',
        hover_data=['Financial Loss (in Million $)'],
        color_continuous_scale='YlOrBr',
        title='<b>Security Vulnerability > Attack Type > Target Industry</b>',
        labels={
            'Number of Affected Users': 'Affected Users',
            'Security Vulnerability Type': 'Vulnerability Type',
            'Attack Type': 'Attack Type',
            'Target Industry': 'Target Industry',
            'Financial Loss (in Million $)': 'Financial Loss (Million $)'
        }
    )

    fig.update_traces(
        textinfo='label+value',
        texttemplate='<b>%{label}</b><br>%{value:,.0f} users',
        textposition='middle center',
        textfont_size=11
    )

    fig.update_layout(
        width=1400,
        height=800,
        font=dict(size=14),
        title_font_size=20,
        margin=dict(t=80, l=0, r=0, b=0)
    )

    return fig



# ---------------8.SANKEY DIAGRAM--------------------------
def create_sankey_diagram(df, dimension_order, sample_size=500, highlight_path=None):
    """ Creates a Sankey diagram showing flow between selected dimensions
    highlight_path: List of values for each dimension to highlight specific path
    Example: ['DDoS', 'Retail', 'Antivirus'] or ['DDoS', None, 'Antivirus'] """

    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

    df_flow = df_sample.copy()
    #istatistik hesaplama kısmı
    total_incidents = len(df_flow)
    total_affected_users = df_flow['Number of Affected Users'].sum()
    stats = {'total_incidents': total_incidents, 'total_affected_users': total_affected_users}
    is_highlighted_flow = False
# eğer kullanıcı bir path seçtiğiyse ona göre filtreleme
    if highlight_path and len(highlight_path) == len(dimension_order) and any(
            val is not None for val in highlight_path):
        df_highlight = df_sample.copy()

        for dim, val in zip(dimension_order, highlight_path):
            if val is not None:
                df_highlight = df_highlight[df_highlight[dim].isin([val])]

        if not df_highlight.empty:
            df_flow = df_highlight
            total_incidents = len(df_flow)
            total_affected_users = df_flow['Number of Affected Users'].sum()
            stats = {'total_incidents': total_incidents, 'total_affected_users': total_affected_users}
            is_highlighted_flow = True
        else:
            return go.Figure(), [], {'total_incidents': 0, 'total_affected_users': 0}
#düğümlerin oluşturulduğu kısım
    node_set = set()
    for _, row in df_sample.iterrows():
        for dim in dimension_order:
            node_set.add(f"{dim}::{row[dim]}")

    node_list = sorted(list(node_set))
    label_list = [n.split("::")[1] for n in node_list]
    node_to_idx = {node_list[i]: i for i in range(len(node_list))}

    flow_counts = {}
#zincir burada oluşturuluyor
    for _, row in df_flow.iterrows():
        chain = [f"{dim}::{row[dim]}" for dim in dimension_order]
        for i in range(len(chain) - 1):
            pair = (chain[i], chain[i + 1])
            flow_counts[pair] = flow_counts.get(pair, 0) + 1

    sources = []
    targets = []
    values = []
    colors = []

    color_palette = px.colors.qualitative.Set3

    for (src, tgt), count in flow_counts.items():
        sources.append(node_to_idx[src])
        targets.append(node_to_idx[tgt])
        values.append(count)

        dim = src.split("::")[0]
        base = color_palette[hash(dim) % len(color_palette)]
# eğer highlight varsa seçili olan net
        if is_highlighted_flow:
            colors.append(base.replace("rgb", "rgba").replace(")", ",0.9)"))
        else:
            colors.append(base.replace("rgb", "rgba").replace(")", ",0.5)"))
#renk ayarları
    node_colors = []
    for node in node_list:
        dim = node.split("::")[0]
        node_colors.append(color_palette[hash(dim) % len(color_palette)])
#grafik çizdirme kısmı
    fig = go.Figure(data=[go.Sankey(
        textfont=dict(color="black", size=12),
        node=dict(
            label=label_list,
            color=node_colors,
            pad=18,
            thickness=24,
            line=dict(color="black", width=1.0),
            hovertemplate='<b>%{label}</b><br>Incidents: %{value}<extra></extra>',
            hoverlabel=dict(
                bgcolor="red",
                bordercolor="white",
                font=dict(color="white", size=14, family="Arial Black")
            )
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Incidents: %{value}<extra></extra>',
            hoverlabel=dict(
                bgcolor="red",
                bordercolor="white",
                font=dict(color="white", size=14, family="Arial Black")
            )
        ),
        arrangement="snap"
    )])
#grafiğin ayarları
    fig.update_layout(
        title=f"Attack Flow Analysis: {' -> '.join(dimension_order)}",
        font=dict(
            size=13,
            color="black"
        ),
        height=800,
        width=1400,
        margin=dict(t=70, b=40, l=40, r=40)
    )

    return fig, label_list, stats


# --- 9. INTERACTIVE VIOLIN PLOT ---
def create_violin_plot(df, x_column, y_column, color_column=None, points="all", log_scale=False):
    if color_column == 'None':
        color_column = None

    hover_cols = [x_column, y_column, 'Year', 'Attack Type', 'Target Industry']

    if color_column:
        hover_cols.append(color_column)

    fig = px.violin(
        df,
        x=x_column,
        y=y_column,
        color=color_column if color_column else x_column,
        box=True,
        points=points,
        hover_data=hover_cols,
        title=f"Distribution of {y_column} by {x_column}"
    )


    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        violingap=0.2,
        violinmode='group',
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title=color_column if color_column else x_column
    )

    return fig
