import base64
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

## Set the page config1uration

st.set_page_config(page_title= "Nassau Candy Distributor", page_icon= "🍭", layout= "wide")
st.title("🚀 Factory Reallocation & Shipping Optimization Recommendation System for Nassau Candy Distributor")
st.markdown('<style>div.block-container{padding-top: 2rem;}</style>', unsafe_allow_html=True)

## Load data

data = st.file_uploader("Upload CSV", type=["csv"])

if data is not None:
    df = pd.read_csv(data)
else:
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "Nassau Candy Distributor.csv")
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

## Feature Engineering

df["Order Date"] = pd.to_datetime(df["Order Date"],errors='coerce')
df = df.dropna(subset=['Order Date'])
# st.write(df['Order Date'].size)
df["Ship Date"] = pd.to_datetime(df["Ship Date"],errors='coerce')
df['Shipping Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df = df[df['Shipping Lead Time'] >= 0]

factory_map = {
    "Wonka Bar - Nutty Crunch Surprise" : "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows" : "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious" : "Lot's O' Nuts", 
    "Wonka Bar - Milk Chocolate" : "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel" : "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS" : "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip" : "Sugar Shack",
    "Fizzy Lifting Drinks" : "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum" : "Secret Factory", 
    "Kazookles" : "The Other Factory",      
    "Hair Toffee": "The Other Factory" }

df["Factory"] = df["Product Name"].map(factory_map)


# df["Shipping Lead Time"] = pd.to_numeric(df["Shipping Lead Time"], errors="coerce")
df['Factory to Customer Region'] = df['Factory'] + " → " + df['Country/Region']
df['Factory to Customer State'] = df['Factory'] + " → " + df['State/Province']
df['City'] = df['City'].str.strip().str.title()
df['State/Province'] = df['State/Province'].str.strip().str.upper()

grouped = df.groupby(['Product Name', 'Factory', 'Region'])['Shipping Lead Time'].mean().reset_index()

results = []
for _, row in grouped.iterrows():
    product = row['Product Name']
    current_factory = row['Factory']
    region = row['Region']
    current_lt = row['Shipping Lead Time']

    for new_factory in df['Factory'].unique():
        if new_factory != current_factory:

            predicted_lt = grouped[
                (grouped['Factory'] == new_factory) &
                (grouped['Region'] == region)
            ]['Shipping Lead Time'].mean()

            if pd.notna(predicted_lt):
                improvement = current_lt - predicted_lt
                percent_improvement = (improvement / current_lt) * 100 if current_lt != 0 else 0

                results.append({
                    'Product': product,
                    'Current Factory': current_factory,
                    'New Factory': new_factory,
                    'Region': region,
                    'Current Lead Time': current_lt,
                    'Predicted Lead Time': predicted_lt,
                    'Improvement': improvement,
                    'Percent Improvement': percent_improvement
                })
simulation = pd.DataFrame(results)

# Best reassignment per product

simulation_df = simulation.loc[simulation.groupby('Product')['Improvement'].idxmax()]

simulation_df = simulation_df[
    simulation_df['Improvement'] > 0]

simulation_df['Old Route'] = (simulation_df['Current Factory'] + " → " + simulation_df['Region'])
simulation_df['New Route'] = (simulation_df['New Factory'] + " → " + simulation_df['Region'])

simulation_df = simulation_df.sort_values(by='Percent Improvement', ascending=False).reset_index(drop=True)

risk_df = df.groupby(
    ['Product Name', 'Factory', 'Region']
)['Shipping Lead Time'].std().reset_index()

risk_df.rename(columns={'Shipping Lead Time': 'Risk'}, inplace=True)
risk_df['Risk'] = risk_df['Risk'].fillna(0)


risk_lookup = (risk_df.set_index(
    ['Product Name', 'Factory', 'Region']
)['Risk'])

risk_lookup = risk_lookup.to_frame()
risk_lookup = risk_lookup.sort_values(by='Risk', ascending=False).head(10).reset_index()

current_risk = risk_lookup.get((product, current_factory, region), None)
new_risk = risk_lookup.get((product, new_factory, region), None)


if pd.notna(current_risk) and pd.notna(new_risk):
    risk_reduction = current_risk - new_risk
else:
    risk_reduction = 0

risk_lookup_series = risk_df.set_index(['Product Name', 'Factory', 'Region'])['Risk']

#Calculate risk reduction for each scenario in the simulation dataframe
def calculate_risk_reduction(row):
    product = row['Product']
    current_factory = row['Current Factory']
    new_factory = row['New Factory']
    region = row['Region']

    current_risk = risk_lookup_series.get((product, current_factory, region), 0)
    new_risk = risk_lookup_series.get((product, new_factory, region), 0)

    # Risk reduction means a decrease in risk (std dev)
    risk_reduction = current_risk - new_risk
    return risk_reduction

simulation['Risk Reduction'] = simulation.apply(calculate_risk_reduction, axis=1)

profit_df = df.groupby(
    ['Product Name', 'Factory', 'Region']
)['Gross Profit'].mean().reset_index()

profit_lookup_series = profit_df.set_index(['Product Name', 'Factory', 'Region'])['Gross Profit']
def calculate_profit_impact(row):
    product = row['Product']
    current_factory = row['Current Factory']
    new_factory = row['New Factory']
    region = row['Region']
    improvement = row['Improvement'] # Assuming 'Improvement' is in the simulation DataFrame

    current_profit = profit_lookup_series.get((product, current_factory, region), 0)
    new_profit = profit_lookup_series.get((product, new_factory, region), current_profit)

    # Define a default weight for lead time improvement's impact on profit
    # This value can be adjusted based on business understanding
    weight_lead_time_impact = 2

    if pd.notna(current_profit) and pd.notna(new_profit):
        profit_change = new_profit - current_profit
        # Add a component for the value of lead time improvement (e.g., reduced holding costs, improved customer satisfaction)
        # A positive improvement (reduction in lead time) should positively impact profit.
        profit_impact = profit_change + (improvement * weight_lead_time_impact)
    else:
        profit_impact = 0 # If profit data is missing for either factory, assume no impact

    return profit_impact

simulation['Profit Impact'] = simulation.apply(calculate_profit_impact, axis=1)




# Company Logo

# st.sidebar.image("logo.png", width=220)

logo_path = os.path.join(base_dir,"logo.png")

with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

st.sidebar.markdown(
    f"""
    <div style="text-align: center; padding: 10px;">
        <img src="data:image/png;base64,{logo_base64}" width="220"
        style="
            box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
        ">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


## Filters

#Navigation filter

page = st.sidebar.selectbox(
    "🧩Modules",
    ["Factory Optimization Simulator", "What-If Scenario Analysis", "Recommendation Dashboard", "Risk & Impact Panel"]
)

# Product/Region/ship Mode/Optimization priority slider

product = st.sidebar.multiselect("📦Select Product", df["Product Name"].unique())
region = st.sidebar.multiselect("🌍Select Region", df["Region"].unique())
ship_mode = st.sidebar.multiselect("🚚Select Ship Mode",df["Ship Mode"].unique())
st.sidebar.markdown("### ⚖️ Optimization Priority")

weight = st.sidebar.slider(
    "Speed 🆚 Profit",
    0.0, 1.0, 0.5
)

strategy = ""
if weight < 0.4:
    strategy = "⚡ Speed-Optimized Strategy"
elif weight > 0.6:
    strategy = "💰 Profit-Optimized Strategy"
else:
    strategy = "🎯 Balanced Optimization Strategy"

# st.markdown(f"""
# <div style="
#     background: linear-gradient(135deg, #6a0dad, #9b59b6);
#     padding:10px;
#     border-radius:10px;
#     color:white;
#     margin-bottom:10px;">
#     {strategy}
# </div>
# """, unsafe_allow_html=True)

speed_weight = 1 - weight  # Improvement focus
profit_weight = weight    # Profit focus
risk_weight = 0.3             # keep fixed or tweak

simulation['Dynamic Score'] = (
    speed_weight * simulation['Improvement'] +
    profit_weight * simulation['Profit Impact'] +
    risk_weight * simulation['Risk Reduction']
)


# Create a copy of the original dataframe to apply filters

filtered_df = df.copy()

# Product 
if product:
    filtered_df = filtered_df[filtered_df["Product Name"].isin(product)]

# Region
if region:
    filtered_df = filtered_df[filtered_df["Region"].isin(region)]


# Ship Mode
if ship_mode:
    filtered_df = filtered_df[filtered_df["Ship Mode"].isin(ship_mode)]



# Col-Metrics(KPIs)
# filtered_df = filtered_df.merge(
#     simulation[['Product', 'Risk Reduction']],
#     on='Product Name',
#     how='left'
#     )

# risk_map = simulation.set_index('Product')['Risk Reduction']

# filtered_df['Risk'] = filtered_df['Product Name'].map(risk_map)

def get_optimization_kpis(simulation, filtered_df):

    # Merge
    filtered_df = filtered_df.merge(
        simulation[['Product', 'Risk Reduction']],
        left_on='Product Name',
        right_on='Product',
        how='left'
    )

    # filtered_df.rename(columns={'Product Name': 'Product'}, inplace=True)

    # Optional mapping (redundant now but okay)
    # risk_map = simulation.set_index('Product')['Risk Reduction']
    # risk_map = simulation.groupby('Product')['Risk Reduction'].mean()
    # filtered_df['Risk'] = filtered_df['Product'].map(risk_map)

        # After merge
    filtered_df.rename(columns={
        'Product Name': 'Product',
        'Risk Reduction': 'Risk'
    }, inplace=True)

    # Remove duplicates
    filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]

    top_n = int((0.1 + 0.4 * weight) * len(filtered_df))
    top_recommendations = (
        simulation
        .sort_values(by='Profit Impact', ascending=False)
        .groupby(['Product', 'Region'])
        .head(top_n)
        )
    
    top_recommendations = top_recommendations[
        top_recommendations['Profit Impact'] > 0
        ]

    top_recommendations = top_recommendations.sort_values(
            by='Dynamic Score', ascending=False
        ).round(2).reset_index(drop=True)

    top_recommendations['Factory Reallocation'] = top_recommendations.apply(
            lambda row: f"{row['Current Factory']}  →  {row['New Factory']}", axis=1)

    # ⚡ Lead Time Reduction (%)
    lead_time_reduction = (
    (top_recommendations['Current Lead Time'].sum() -
     top_recommendations['Predicted Lead Time'].sum())
    /
    (top_recommendations['Current Lead Time'].sum() + 1e-6)
    ) * 100

    # 💰 Profit Stability
    # profit_mean = top_recommendations['Profit Impact'].mean()
    # profit_std = top_recommendations['Profit Impact'].std()
    profit_stability = (
    100 - (
        top_recommendations['Profit Impact'].std() /
        (abs(top_recommendations['Profit Impact']).mean() + 1e-6)
    ) * 100
    )

    # 🧠 Scenario Confidence
    # confidence_score = (
    #     top_recommendations['Risk Reduction'].mean() /
    #     top_recommendations['Risk Reduction'].max()
    # ) * 100 if top_recommendations['Risk Reduction'].max() != 0 else 0

    confidence_score = (
    100 - (
        top_recommendations['Risk Reduction'].std() /
        (abs(top_recommendations['Risk Reduction']).max() + 1e-6)
    ) * 100
    )

    # 🌍 Recommendation Coverage
    # recommendation_coverage = (
    #                         top_recommendations['Product'].nunique() /
    #                         simulation['Product'].nunique()
    #                     ) * 100
    recommendation_coverage = (
    top_recommendations['Product'].nunique() /
    filtered_df['Product'].nunique() + 1e-6
    ) * 100 if filtered_df['Product'].nunique() != 0 else 0

    return lead_time_reduction, profit_stability, confidence_score, recommendation_coverage

lead_time_reduction, profit_stability, confidence_score, recommendation_coverage = get_optimization_kpis(
    simulation,filtered_df
    )

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

def metric_box(label, value, delta=None):
    st.markdown(
    """
    <div style="
        background: rgba(106, 13, 173, 0.7);
        background: linear-gradient(135deg, #6a0dad, #9b59b6);;
        padding:16px;
        border-radius:14px;
        text-align:center;
        color:white;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom:10px;
        ">
    """,
    unsafe_allow_html=True
    )

    st.metric(label, value, delta=delta)

    st.markdown("</div>", unsafe_allow_html=True)



with col1:
    metric_box(
        "⚡ Lead Time Reduction",
        f"{lead_time_reduction:.1f}%",
        delta="Operational Gain"
    )

with col2:
    metric_box(
        "💰 Profit Stability",
        f"{profit_stability:.1f}%",
        delta="Financial Safety"
    )

with col3:
    metric_box(
        "🧠 Scenario Confidence",
        f"{confidence_score:.1f}%",
        delta="Reliability"
    )

with col4:
    metric_box(
        "🌍 Recommendation Coverage",
        f"{recommendation_coverage:.1f}%",
        delta="Scalability"
    )






#Define Functions for KPIs

def factory_performance(filtered_df, product):

    if not product:  # handles None + [] automatically
        prod = filtered_df.copy()

    elif isinstance(product, list):
        prod = filtered_df[filtered_df['Product Name'].isin(product)]

    else:
        prod = filtered_df[filtered_df['Product Name'] == product]

    result = prod.groupby('Factory').agg({
        'Shipping Lead Time': 'mean',
        'Cost': 'mean',
        'Gross Profit': 'mean'
    }).round(2).reset_index()

    st.dataframe(result)

    if result is not None:

        result['profit_norm'] = (
            (result['Gross Profit'] - result['Gross Profit'].min()) /
            (result['Gross Profit'].max() - result['Gross Profit'].min())
            )

        result['speed_norm'] = (
            (1 / result['Shipping Lead Time'] - (1 / result['Shipping Lead Time']).min()) /
            ((1 / result['Shipping Lead Time']).max() - (1 / result['Shipping Lead Time']).min())
            )

        result['Score'] = (
            weight * result['profit_norm'] +
            (1 - weight) * result['speed_norm']
            )


        # 🔥 Create Score
        # result['Score'] = (
        #     (1 - weight) * (1 / result['Shipping Lead Time']) +
        #     weight * result['Gross Profit']
        # )

        # 🏆 Best Factory
        # best = result.loc[result['Score'].idxmax()]
        best = result.sort_values(by='Score', ascending=False).iloc[0]


        st.success(f"🏆 Recommended Factory: {best['Factory']}")

        # 📊 Show sorted table
        result1 = result.sort_values(by='Score', ascending=False)
        st.dataframe(result1)

        # 📈 Optional chart


        fig = px.bar(
            result1,
            x="Score",
            y="Factory",
            orientation="h",
            template="plotly_dark",
            title="🏭 Factory Optimization Insights",
            color="Score",
            color_continuous_scale="plasma",
            hover_data={
                "Shipping Lead Time": True,
                "Cost": True,  
                "Gross Profit": True,
                "Score": True
            }
        )

        # 🎯 Center + style title
        fig.update_layout(
            title={
                'text': "<b>🏭 Factory Optimization Insights</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=24)
            },
            xaxis_title="Optimization Score",
            yaxis_title="Factory",
            height=500
        )

        # 📏 Make spacing clean
        fig.update_layout(
            yaxis=dict(tickmode='linear')
        )

        # 🏆 Highlight best factory
        fig.update_traces(
            text=result1["Score"].round(2),
            textposition="outside"
        )

        # 🎨 OPTIONAL: highlight best one differently
        fig.update_traces(
            marker=dict(
                color=[
                    "#00FFAA" if f == best["Factory"] else val
                    for f, val in zip(result1["Factory"], result1["Score"])
                ]
            )
        )

    return result,result1,fig


def scenario_comparison(simulation):
    # comparison = simulation.groupby(['Product']).agg({
    #     'Current Lead Time': 'mean',
    #     'Predicted Lead Time': 'mean'
    # }).reset_index()

    # comparison['Improvement'] = (
    #     comparison['Current Lead Time'] - comparison['Predicted Lead Time']
    # )
    simulation_df = simulation.loc[simulation.groupby('Product')['Improvement'].idxmax()]

    simulation_df = simulation_df[
        simulation_df['Improvement'] > 0
    ]

    simulation_df['Old Route'] = (simulation_df['Current Factory'] + " → " + simulation_df['Region'])
    simulation_df['New Route'] = (simulation_df['New Factory'] + " → " + simulation_df['Region'])

    simulation_df = simulation_df.sort_values(by='Percent Improvement', ascending=False).reset_index(drop=True)

    # Factory V/S Improvement Chart
    result = simulation_df.groupby('New Route', as_index=False).agg({
        'Current Lead Time': 'mean',
        'Predicted Lead Time': 'mean',
        'Improvement': 'sum',
        }).round(2).sort_values(by="Improvement", ascending=False)

    fig1 = px.bar(
    result,
    x="Improvement",
    y="New Route",
    orientation="h",
    template="plotly_dark",
    title="⚡Total Lead Time Improvement by Factory",
    color="Improvement",
    color_continuous_scale="plasma",
    hover_data={
        "Current Lead Time": True,
        "Predicted Lead Time": True,
        "Improvement": True
    }
    )

    # best = result.sort_values(by="Improvement", ascending=False).iloc[0]

    # fig.update_traces(
    #     text=result["Improvement"].round(2),
    #     textposition="outside",
    #     marker=dict(
    #     color=[
    #         "#00FFAA" if f == best["New Route"] else val
    #         for f, val in zip(result["New Route"], result["Improvement"])
    #         ]
    #     )
    # )

    fig1.update_traces(
        text=result["Improvement"].round(2),
        textposition="outside"
    )

    fig1.update_layout(
        title={
            'text': "<b>⚡Total Lead Time Improvement by Factory</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=24)
        },
        xaxis_title="Lead Time Improvement",
        yaxis_title="New Route",
        height=500,
        yaxis=dict(tickmode='linear')
    )


    fig2 = px.pie(
    result,
    values=result['Improvement'],
    names='New Route',
    hole=0.6,
    color_discrete_sequence=[
    "#4B0082", "#6A0DAD", "#8A2BE2", "#BA55D3"]
    )

    fig2.update_layout(
    title={
        'text': "<b>⏱️Lead Time Gains from Route Optimization</b>",
        'x': 0.5,
        'xanchor': 'center',
        'font': dict(size=24)})

    fig2.update_traces(
        textinfo='percent',
        textposition='inside',
        textfont_size=12,
        hovertemplate="<b>%{label}</b><br>Improvement: %{value}",
        marker=dict(line=dict(color='white', width=1))
    )


    fig2.update_layout(
        showlegend=True,
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',  # transparent bg
    )

    fig2.update_traces(
    marker=dict(
        line=dict(color='rgba(255,255,255,0.2)', width=2))
    )

    fig2.update_layout(
    annotations=[dict(
        text="Total<br>Impact",
        x=0.5, y=0.5,
        showarrow=False,
        font_size=16
    )])


    return simulation_df,fig1,fig2


def recommendation_dashboard(simulation):

    # df_sorted = filtered_df.sort_values(by='Improvement', ascending=False)
    # result = df_sorted[['Product', 'Region', 'Current Factory', 
    #                   'Recommended Factory', 'Improvement']].head(10)

    top_recommendations = (
        simulation
        .sort_values(by='Profit Impact', ascending=False)
        .groupby(['Product', 'Region'])
        .head(150)
        )
    
    top_recommendations = top_recommendations[
    top_recommendations['Profit Impact'] > 0
    ]
    
    # top_recommendations['Score'] = (
    #     0.4 * top_recommendations['Improvement'] +
    #     0.3 * top_recommendations['Risk Reduction'] +
    #     0.3 * top_recommendations['Profit Impact']
    # )
    
    # top_recommendations = (
    # simulation
    # .assign(Score = (
    #     0.4 * simulation['Efficiency Gain'] +
    #     0.3 * simulation['Risk Reduction'] +
    #     0.3 * simulation['Profit Impact']
    # ))
    # .sort_values(by='Score', ascending=False)
    # .groupby(['Product', 'Region'])
    # .head(10)
    # .reset_index(drop=True)
    # )

    top_recommendations = top_recommendations.sort_values(
        by='Dynamic Score', ascending=False
    ).round(2).reset_index(drop=True)

    top_recommendations['Factory Reallocation'] = top_recommendations.apply(
        lambda row: f"{row['Current Factory']}  →  {row['New Factory']}", axis=1)

    clean_df = (
    top_recommendations
    .groupby('Factory Reallocation', as_index=False)
    .agg({'Dynamic Score': 'sum'})   # or 'mean' depending on logic
    .sort_values(by='Dynamic Score', ascending=False)
    )

#Fig1:

    fig1 = px.bar(
    clean_df,
    x="Dynamic Score",
    y="Factory Reallocation",
    orientation="h",
    template="plotly_dark",
    title="🔝 Top Factory Reallocation Recommendations",
    color="Dynamic Score",
    color_continuous_scale="plasma",
    hover_data={
        "Dynamic Score": True
    }
    )

    fig1.update_traces(
        text=top_recommendations["Improvement"].round(2),
        textposition="outside"
    )

    fig1.update_layout(
        title={
            'text': "<b>🔝 Top Factory Reallocation Recommendations</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=24)
        },
        xaxis_title="Overall Score",
        yaxis_title="Factory Reallocation",
        height=500,
        yaxis=dict(tickmode='linear')
    )

#Fig2

    Eff_gain = pd.DataFrame({
    'Metric': ['Improvement', 'Risk Reduction', 'Profit Impact'],
    'Value': [
        top_recommendations['Improvement'].sum(),
        top_recommendations['Risk Reduction'].sum(),
        top_recommendations['Profit Impact'].sum()
    ]})

    Eff_gain['% Contribution'] = (Eff_gain['Value'] / Eff_gain['Value'].sum()) * 100


    fig2 = px.pie(
    Eff_gain,
    names='Metric',
    values='% Contribution',
    hole=0.6,
    color_discrete_sequence=[
    "#4B0082", "#6A0DAD", "#8A2BE2", "#BA55D3"]
    )

    fig2.update_layout(
    title={
        'text': "<b>📊Expected Efficiency Gains</b>",
        'x': 0.5,
        'xanchor': 'center',
        'font': dict(size=24)})

    fig2.update_traces(
        textinfo='percent',
        textposition='inside',
        textfont_size=12,
        hovertemplate="<b>%{label}</b><br>Improvement: %{value}",
        marker=dict(line=dict(color='white', width=1))
    )


    fig2.update_layout(
        showlegend=True,
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',  # transparent bg
    )

    fig2.update_traces(
    marker=dict(
        line=dict(color='rgba(255,255,255,0.2)', width=2))
    )

    fig2.update_layout(
    annotations=[dict(
        text="Overall<br>Score",
        x=0.5, y=0.5,
        showarrow=False,
        font_size=16
    )])

#Fig3: Risk vs Improvement Tradeoff

    fig3 = px.scatter(
    top_recommendations,
    x='Risk Reduction',
    y='Improvement',
    size='Profit Impact',
    color='Profit Impact',
    hover_name='Factory Reallocation',
    size_max=40,
    color_continuous_scale=[
        '#2e0066', '#5e2ca5', '#9b59b6', '#c39bd3', '#f1c40f'  # deep purple → gold highlight
    ]
    )

    fig3.update_traces(
    marker=dict(
        line=dict(width=1, color='white'),
        opacity=0.85
    ),
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "🛡️ Risk Reduction: %{x:.2f}<br>" +
                  "⚡ Improvement: %{y:.2f}<br>" +
                  "💰 Profit Impact: %{marker.size:.2f}<extra></extra>"
    )

    fig3.update_layout(
    title={
        'text': "<b>⚡Risk vs Improvement Tradeoff</b>",
        'x': 0.5,
        'xanchor': 'center',
        'font': dict(size=24)
    },
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),

    xaxis=dict(
        title="🛡️ Risk Reduction",
        showgrid=True,
        gridcolor='rgba(255,255,255,0.08)'
    ),

    yaxis=dict(
        title="⚡ Improvement",
        showgrid=True,
        gridcolor='rgba(255,255,255,0.08)'
    ),

    coloraxis_colorbar=dict(
        title="💰 Profit",
        thickness=12,
        len=0.6
    )
    )

#Fig4

    top_drivers = top_recommendations.groupby('Factory Reallocation').agg({'Improvement': 'sum'}).sort_values(
                  by='Improvement', ascending=False).head(10).reset_index()

    fig4 = px.bar(
    top_drivers,
    x="Improvement",
    y="Factory Reallocation",
    orientation="h",
    template="plotly_dark",
    title="🔍 Top Drivers of Improvement",
    color="Improvement",
    color_continuous_scale="plasma",
    hover_data={
        "Improvement": True
    }
    )

    fig4.update_traces(
        text=top_drivers["Improvement"].round(2),
        textposition="outside"
    )

    fig4.update_layout(
        title={
            'text': "<b>🔍Top Drivers of Improvement</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=24)
        },
        xaxis_title="Lead Time Improvement",
        yaxis_title="Factory Reallocation",
        height=500,
        yaxis=dict(tickmode='linear')
    )

    return top_recommendations,fig1,fig2,fig3,fig4

def risk_impact_panel(simulation):

    # top_recommendations = (
    #     simulation
    #     .sort_values(by='Profit Impact', ascending=False)
    #     .groupby(['Product', 'Region'])
    #     .head(150)
    # )
    
    top_recommendations = (
    simulation
    .sort_values(by='Dynamic Score', ascending=False)
    .groupby(['Product', 'Region'])
    .head(150)
    .round(2)
    .reset_index(drop=True)
    )

    top_recommendations = top_recommendations[
    top_recommendations['Profit Impact'] > 0
    ]
    
    # top_recommendations['Score'] = (
    #     0.4 * top_recommendations['Improvement'] +
    #     0.3 * top_recommendations['Risk Reduction'] +
    #     0.3 * top_recommendations['Profit Impact']
    # )

    # top_recommendations = top_recommendations.sort_values(
    #     by='Dynamic Score', ascending=False
    # ).round(2).reset_index(drop=True)

    top_recommendations['Factory Reallocation'] = top_recommendations.apply(
        lambda row: f"{row['Current Factory']}  →  {row['New Factory']}", axis=1)

    
    threshold_risk = top_recommendations['Risk Reduction'].quantile(0.1)

    risk_alerts = top_recommendations[
    top_recommendations['Risk Reduction'] <= threshold_risk
    ].sort_values(by='Risk Reduction').round(2).reset_index(drop=True)




#Fig1

    fig1 = px.histogram(
    top_recommendations,
    x='Profit Impact',
    nbins=20,
    opacity=0.85,
    color_discrete_sequence=['#9b59b6']  # main purple
    )

    fig1.update_traces(
    marker=dict(
        line=dict(color='white', width=1)  # crisp edges
    ),
    hovertemplate="💰Profit Impact: %{x:.2f}<br>Count: %{y}<extra></extra>",
    )

    fig1.update_layout(
    title="💰 Profit Impact Distribution",
    title_x=0.3,

    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',

    font=dict(color='white'),
    title_font=dict(size=24),
    annotations=[dict(font=dict(size=14))],

    xaxis=dict(
        title="💰 Profit Impact",
        showgrid=True,
        gridcolor='rgba(255,255,255,0.08)'
    ),

    yaxis=dict(
        title="📊Frequency",
        showgrid=True,
        gridcolor='rgba(255,255,255,0.08)'
    ),

    bargap=0.15
    )


    mean_val = top_recommendations['Profit Impact'].mean()

    fig1.add_vline(
    x=mean_val,
    line_dash="dash",
    line_color="#f1c40f",  # gold highlight
    annotation_text="Avg",
    annotation_position="top"
    )

#Fig2

    fig2 = px.box(
    top_recommendations,
    y='Risk Reduction',
    points='all',  # show all points (important insight)
    )

    fig2.update_traces(
    marker=dict(
        color='#9b59b6',  # purple points
        size=6,
        opacity=0.7,
        line=dict(width=0.5, color='white')
    ),
    line=dict(color='#c39bd3'),  # box outline
    fillcolor='rgba(155, 89, 182, 0.3)',  # soft purple fill
    hovertemplate="🛡️ Risk Reduction: %{y:.2f}<extra></extra>"
    )

    fig2.update_layout(
    title="⚠️ Risk Reduction Spread",
    title_x=0.3,

    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',

    font=dict(color='white'),
    title_font=dict(size=24),
    annotations=[dict(font=dict(size=14))],

    yaxis=dict(
        title="🛡️ Risk Reduction",
        showgrid=True,
        gridcolor='rgba(255,255,255,0.08)'
    ),

    xaxis=dict(showticklabels=False),  # cleaner look
    )

    median_val = top_recommendations['Risk Reduction'].median()

    fig2.add_hline(
        y=median_val,
        line_dash="dash",
        line_color="#f1c40f",
        annotation_text="Median",
        annotation_position="right"
    )
 
    return risk_alerts,top_recommendations,fig1,fig2
 




## Main Content

if page == "Factory Optimization Simulator":
    st.header("📊 Predicted Factory Performance (Based on Optimization)")  

    result,result1,fig = factory_performance(filtered_df, product) 

    st.plotly_chart(fig, use_container_width=True,key="fig")

elif page == "What-If Scenario Analysis":
    st.header("🔍 What-If Scenario Analysis")
    st.markdown("Compare current vs predicted shipping lead times across different products and regions to evaluate potential improvements.")

    filtered_df = simulation.copy()

# Product 
    if product:
        filtered_df = filtered_df[filtered_df["Product"].isin(product)]

# Region
    if region:
        filtered_df = filtered_df[filtered_df["Region"].isin(region)]


    simulation_df,fig1,fig2 = scenario_comparison(filtered_df)
    st.dataframe(
    simulation_df.style
    .background_gradient(cmap='Purples', subset=['Current Lead Time', 'Predicted Lead Time', 'Improvement', 'Percent Improvement'])
    )
    st.plotly_chart(fig1, use_container_width=True,key="fig1")
    st.markdown('---')
    st.plotly_chart(fig2, use_container_width=True,key="fig2")

elif page == "Recommendation Dashboard":
    st.header("📈Ranked reassignment suggestions")
    st.markdown("### Top Product-Region combinations with highest predicted lead time improvement and their recommended factory reallocation.")

    filtered_df = simulation.copy()

    # Product 
    if product:
        filtered_df = filtered_df[filtered_df["Product"].isin(product)]

    # Region
    if region:
        filtered_df = filtered_df[filtered_df["Region"].isin(region)]
        
    top_recommendations,fig1,fig2,fig3,fig4 = recommendation_dashboard(filtered_df)

    st.dataframe(
    top_recommendations.style
    .background_gradient(cmap='Purples', subset=['Predicted Lead Time', 'Improvement', 'Percent Improvement','Risk Reduction','Dynamic Score', 'Profit Impact'])
    )

    st.plotly_chart(fig1, use_container_width=True,key="fig1")
    st.markdown('---')
    st.plotly_chart(fig2, use_container_width=True,key="fig2")
    st.markdown('---')
    st.plotly_chart(fig3, use_container_width=True,key="fig3")
    st.markdown('---')
    st.plotly_chart(fig4, use_container_width=True,key="fig4")

elif page == "Risk & Impact Panel":
    st.header("⚠️ High-Risk Reassignments")
    st.markdown("Identify reassignments with high lead time improvement but low risk reduction to ensure balanced decision-making.")

    filtered_df = simulation.copy()

    # Product 
    if product:
        filtered_df = filtered_df[filtered_df["Product"].isin(product)]

    # Region
    if region:
        filtered_df = filtered_df[filtered_df["Region"].isin(region)]
    

    risk_alerts,filtered_df,fig1,fig2 = risk_impact_panel(filtered_df)

    st.dataframe(
    risk_alerts.style
    .background_gradient(cmap='Purples', subset=['Improvement', 'Percent Improvement','Risk Reduction','Dynamic Score', 'Profit Impact'])
    )

    with st.container():
        st.markdown("### 🧠 Key Insights")

        if filtered_df['Profit Impact'].skew() > 1:
            st.info("💰 Profit impact is highly concentrated among a few reallocations.")

        if (filtered_df['Risk Reduction'] < 10).sum() > 5:
            st.warning("⚠️ Multiple reallocations show low risk reduction.")

        if len(risk_alerts) > 0:
            st.error("🚨 High-profit but risky reallocations detected.")

    st.plotly_chart(fig1, use_container_width=True,key="fig1")
    st.markdown('---')
    st.plotly_chart(fig2, use_container_width=True,key="fig2")
    st.markdown("### 🧠 Summary")

    st.success(
        f"Top reallocations yield up to {filtered_df['Profit Impact'].max():.0f} profit impact "
        f"with risk reduction varying significantly across regions."
    )

    best = filtered_df.iloc[0]

    st.info(
        f"🚀 Best Opportunity: {best['Factory Reallocation']} "
        f"→ Profit: {best['Profit Impact']:.0f}, Risk Reduction: {best['Risk Reduction']:.0f}"
    )


