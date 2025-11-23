# 🛡️ Global Cybersecurity Threats Dashboard (2015-2024)

##  Project Description

This interactive dashboard provides comprehensive analysis and visualization of global cybersecurity threats spanning from 2015 to 2024. Built with Streamlit and Plotly, the application enables users to explore complex relationships between attack patterns, geographical distributions, industry vulnerabilities, and defense mechanisms through multiple coordinated visualizations.

###  Project Objectives

- **Analyze Global Threat Landscape:** Visualize the distribution and intensity of cyber attacks across different countries and regions
- **Identify Attack Patterns:** Discover relationships between attack types, sources, vulnerabilities, and targeted industries
- **Evaluate Defense Effectiveness:** Assess the performance of various defense mechanisms across multiple metrics
- **Predict Financial Impact:** Leverage machine learning to forecast potential financial losses and recommend optimal defense strategies
- **Enable Interactive Exploration:** Provide dynamic filtering and customizable visualizations for deep-dive analysis

###  Key Features

#### **10  Visualizations**
1. **Choropleth Map** - Global threat distribution with financial loss or affected users metrics
2. **Time Series Chart** - Annual attack trends over the decade
3. **Heatmap** - Intensity matrix showing country vs. attack type relationships
4. **Sunburst Chart** - Hierarchical breakdown of top 10 countries by attack characteristics
5. **Radar Chart** - Multi-dimensional defense mechanism performance evaluation
6. **Scatter Plot** - Industry risk analysis with bubble sizing by incident count
7. **Treemap** - Hierarchical view of vulnerability-attack-industry relationships
8. **Sankey Diagram** - Flow analysis with customizable dimensions and path highlighting
9. **Violin Plot** - Distribution analysis across categories with interactive axis selection
10. **Prediction Module** - Random Forest-based financial loss forecasting with defense optimization

#### **Interactive Filtering System**
- Multi-select filters for countries, attack types, sources, vulnerabilities, and industries
- Range sliders for year and incident resolution time
- Real-time data filtering with synchronized chart updates
- Dynamic metric switching (Financial Loss vs. Affected Users)

#### **Advanced Capabilities**
- **Sankey Path Highlighting:** Select specific attack flow paths to analyze targeted scenarios
- **Customizable Dimensions:** Configure Sankey diagram with 3-4 dimensional flows
- **ML-Powered Predictions:** Predict financial losses based on attack characteristics
- **Defense Optimization:** Automatically recommend the most effective defense mechanism
- **Sample Size Control:** Adjust data sampling for performance optimization

###  Use Cases

- **Security Analysts:** Identify emerging threat patterns and vulnerable sectors
- **Policy Makers:** Understand global cyber threat landscapes for strategic planning
- **Organizations:** Evaluate industry-specific risks and optimize defense investments
- **Researchers:** Explore correlations between attack characteristics and outcomes
- **Educators:** Demonstrate data visualization and machine learning techniques in cybersecurity

###  Dataset Overview

- **Source:** [Global Cybersecurity Threats Dataset (2015-2024)](https://www.kaggle.com/datasets/atharvasoundankar/global-cybersecurity-threats-2015-2024/data)
- **Records:** 3000
- **Time Span:** 10 years (2015-2024)
- **Geographical Coverage:** Global (multiple countries)
- **Key Attributes:**
  - Attack characteristics (type, source, vulnerability)
  - Impact metrics (financial loss, affected users, resolution time)
  - Defense information (mechanisms used)
  - Target details (industry, country)
  - Temporal data (year)
 
## How to Run This Project

#### Step 1: Download the Project
```bash
git clone https://github.com/BoraOzel/CEN445-DataVisualizationProject.git
cd CEN445-DataVisualizationProject
```
*Or download as ZIP and extract it*

---
#### Step 2: Install Required Libraries
Open terminal/command prompt in the project folder and run:
```bash
pip install streamlit pandas plotly scikit-learn numpy
```
**What each library does:**
- **streamlit** - Creates the interactive web dashboard interface
- **pandas** - Handles data loading, filtering, and manipulation
- **plotly** - Generates all interactive charts and visualizations
- **scikit-learn** - Powers the machine learning prediction module (Random Forest)
- **numpy** - Performs numerical calculations and data processing

---
---
#### Step 3: Run the Dashboard
```bash
streamlit run app.py
```

