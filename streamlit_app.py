import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configuration with a modern tech title and wide layout
st.set_page_config(
    page_title="Amazon Business Intelligence & Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Amazon-inspired Styling
st.markdown("""
    <style>
    /* CSS hacks for styling custom components */
    .metric-card {
        background-color: #f7f9fa;
        border: 1px solid #e7ebed;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.05);
        border-color: #ff9900;
    }
    .card-title {
        color: #565959;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .card-value {
        color: #111111;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .card-subtitle {
        color: #ff9900;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-critical {
        background-color: #ffd8d8;
        color: #900000;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .badge-high {
        background-color: #ffe8d8;
        color: #904000;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .badge-medium {
        background-color: #d8ecff;
        color: #004090;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    .badge-low {
        background-color: #f0f2f5;
        color: #4b5563;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load datasets safely with caching
@st.cache_data
def load_data():
    data = {}
    try:
        if os.path.exists("Data/amazon_transactions_clean.csv"):
            data['hist'] = pd.read_csv("Data/amazon_transactions_clean.csv")
            data['hist']['OrderDate'] = pd.to_datetime(data['hist']['OrderDate'])
            
        # Combined 2025-2026 forecasts
        for fc_file, key in [
            ("outputs/category_forecast.csv", 'cat_fc'),
            ("outputs/product_forecast.csv",  'prod_fc'),
            ("outputs/business_forecast.csv", 'biz_fc'),
        ]:
            if os.path.exists(fc_file):
                data[key] = pd.read_csv(fc_file)
                data[key]['OrderMonth'] = pd.to_datetime(data[key]['OrderMonth'])

        # Year-split forecasts for Power BI / What-If
        for yr in [2025, 2026]:
            for suffix, key_prefix in [('category_forecast', 'cat_fc'), ('product_forecast', 'prod_fc'), ('business_forecast', 'biz_fc')]:
                f = f"outputs/{suffix}_{yr}.csv"
                if os.path.exists(f):
                    df_yr = pd.read_csv(f)
                    df_yr['OrderMonth'] = pd.to_datetime(df_yr['OrderMonth'])
                    data[f'{key_prefix}_{yr}'] = df_yr

        if os.path.exists("outputs/inventory_plan.csv"):
            data['inv_plan'] = pd.read_csv("outputs/inventory_plan.csv")
            data['inv_plan']['OrderMonth'] = pd.to_datetime(data['inv_plan']['OrderMonth'])

        if os.path.exists("outputs/category_inventory_plan.csv"):
            data['cat_inv_plan'] = pd.read_csv("outputs/category_inventory_plan.csv")
            data['cat_inv_plan']['OrderMonth'] = pd.to_datetime(data['cat_inv_plan']['OrderMonth'])

        if os.path.exists("reports/model_comparison.csv"):
            data['comparison'] = pd.read_csv("reports/model_comparison.csv")

        if os.path.exists("reports/business_insights.csv"):
            data['insights'] = pd.read_csv("reports/business_insights.csv")

        if os.path.exists("reports/category_seasonality.csv"):
            data['cat_seasonality'] = pd.read_csv("reports/category_seasonality.csv")

        if os.path.exists("reports/product_seasonality.csv"):
            data['prod_seasonality'] = pd.read_csv("reports/product_seasonality.csv")

        if os.path.exists("reports/executive_summary.csv"):
            data['exec_summary'] = pd.read_csv("reports/executive_summary.csv")
    except Exception as e:
        st.error(f"Error loading reports or forecasts: {str(e)}")
    return data

data = load_data()

# Navigation Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
st.sidebar.title("BI & Forecasting Platform")

page = st.sidebar.radio(
    "Navigate:",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "📅 Seasonality Analysis",
        "📈 Forecasting (2025–2026)",
        "📦 Inventory Planner",
        "💡 Business Insights",
        "🤖 Model Comparison",
        "📋 Executive Summary",
        "🧠 AI Business Assistant",
        "🔮 What-If Analysis",
        "ℹ️ About Project",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Amazon BI Platform v2.0")
st.sidebar.caption("🗓️ Forecast horizon: 2025–2026")

# Check if data loaded correctly
if not data or 'hist' not in data:
    st.error("Historical dataset not found. Please execute the pipeline by running `run_pipeline.py` first.")
    st.stop()

# Load historical shortcut aggregates
hist_df = data['hist']
total_rev_hist = hist_df['TotalAmount'].sum()
total_orders_hist = hist_df['OrderID'].nunique()
total_cust_hist = hist_df['CustomerID'].nunique()

# Forecast aggregates
total_forecast_rev = 0.0
total_growth = 0.0
avg_mape = 0.0

if 'biz_fc' in data:
    total_forecast_rev = data['biz_fc']['Forecast_Revenue'].sum()
    # Compute YoY growth (comparing last historical year e.g. 2024 to 2026 forecast)
    last_year = hist_df['OrderDate'].dt.year.max()
    last_year_rev = hist_df[hist_df['OrderDate'].dt.year == last_year]['TotalAmount'].sum()
    if last_year_rev > 0:
        total_growth = ((total_forecast_rev - last_year_rev) / last_year_rev) * 100.0

if 'comparison' in data and not data['comparison'].empty:
    avg_mape = data['comparison']['MAPE'].mean()

# -------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------
if page == "🏠 Home":
    st.title("📦 Amazon Business Intelligence & Demand Forecasting Platform")
    st.subheader("Executive summary of demand forecasts, system accuracy, and business metrics.")
    
    st.markdown("---")
    
    # KPI Grid
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Total Revenue (Hist)</div>
                <div class="card-value">${total_rev_hist/1e6:.2f}M</div>
                <div class="card-subtitle">5-Year Cumulative</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Total Orders (Hist)</div>
                <div class="card-value">{total_orders_hist/1e3:.1f}k</div>
                <div class="card-subtitle">Active transactions</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Active Customers</div>
                <div class="card-value">{total_cust_hist/1e3:.1f}k</div>
                <div class="card-subtitle">Unique buyers</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">2026 Projected Rev</div>
                <div class="card-value">${total_forecast_rev/1e6:.2f}M</div>
                <div class="card-subtitle">Complete Year 2026</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Projected Growth</div>
                <div class="card-value">+{total_growth:.1f}%</div>
                <div class="card-subtitle">YoY compared to base</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col6:
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Forecast Error</div>
                <div class="card-value">{avg_mape:.1f}%</div>
                <div class="card-subtitle">Average MAPE</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main grid for high-level charts
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("Historical vs 2026 Forecast Demand Path")
        # Visualizing Business Forecast
        if 'biz_fc' in data:
            biz_hist = hist_df.groupby(hist_df['OrderDate'].dt.to_period('M')).agg({'Quantity': 'sum'}).reset_index()
            biz_hist['OrderMonth'] = biz_hist['OrderDate'].dt.to_timestamp()
            
            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(
                x=biz_hist['OrderMonth'], y=biz_hist['Quantity'],
                name='Historical Demand', line=dict(color='#232f3e', width=2)
            ))
            # Forecast 2026
            fig.add_trace(go.Scatter(
                x=data['biz_fc']['OrderMonth'], y=data['biz_fc']['Forecast_Sales'],
                name='Projected 2026 Demand', line=dict(color='#ff9900', dash='dash', width=2.5)
            ))
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([data['biz_fc']['OrderMonth'], data['biz_fc']['OrderMonth'][::-1]]),
                y=pd.concat([data['biz_fc']['Forecast_Sales_Upper'], data['biz_fc']['Forecast_Sales_Lower'][::-1]]),
                fill='toself', fillcolor='rgba(255, 153, 0, 0.15)', line=dict(color='rgba(255, 153, 0, 0)'),
                name='Confidence Bounds', showlegend=True
            ))
            
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Timeline",
                yaxis_title="Quantity Sold (Units)",
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with right_col:
        st.subheader("Top Category Breakdown")
        cat_shares = hist_df.groupby('Category')['TotalAmount'].sum().reset_index()
        fig_pie = px.pie(
            cat_shares, values='TotalAmount', names='Category',
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Key Architecture & Project Highlights")
    st.markdown("""
        - **Hierarchical Engine**: Prophet is trained for high-trend/seasonal aggregates (Categories), while custom Holt-Winters benchmarks top selling items individually. Custom time-series feature-engineered **XGBoost** handles tail products with sparse data.
        - **Inventory Optimization**: Computes safety stock buffers, procurement lead times, and warning flags based on Z-score analytics.
        - **Dynamically Generated Insights**: Generates over 25+ metrics-driven business indicators dynamically for stakeholders.
    """)

# -------------------------------------------------------------
# DATA ANALYSIS PAGE
# -------------------------------------------------------------
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis & Business Intelligence")
    st.subheader("Exploratory Data Analysis of Sales Performance and Customer Behavior.")
    
    st.markdown("---")
    
    # Sidebar style options for charts
    st.write("### Filter & Segment Historical Performance")
    
    # 1. Row: Monthly Sales & Aggregates
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Monthly Sales Quantity Trend (Moving Average)")
        monthly_sales = hist_df.groupby(hist_df['OrderDate'].dt.to_period('M')).agg({'Quantity': 'sum', 'TotalAmount': 'sum'}).reset_index()
        monthly_sales['OrderMonth'] = monthly_sales['OrderDate'].dt.to_timestamp()
        
        # Calculate moving averages
        monthly_sales['MA_3'] = monthly_sales['Quantity'].rolling(3).mean()
        monthly_sales['MA_6'] = monthly_sales['Quantity'].rolling(6).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_sales['OrderMonth'], y=monthly_sales['Quantity'],
            name='Sales Quantity', marker_color='#232f3e', opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=monthly_sales['OrderMonth'], y=monthly_sales['MA_3'],
            name='3-Month Moving Average', line=dict(color='#ff9900', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=monthly_sales['OrderMonth'], y=monthly_sales['MA_6'],
            name='6-Month Moving Average', line=dict(color='#008296', width=2)
        ))
        fig.update_layout(xaxis_title="Date", yaxis_title="Units", hovermode="x unified", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.write("#### Monthly Revenue Trend ($)")
        fig_rev = px.line(monthly_sales, x='OrderMonth', y='TotalAmount', line_shape='linear', color_discrete_sequence=['#ff9900'])
        fig_rev.update_traces(line=dict(width=3))
        fig_rev.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)", hovermode="x unified")
        st.plotly_chart(fig_rev, use_container_width=True)
        
    st.markdown("---")
    
    # 2. Row: Regional demand & Payment methods
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("#### Revenue Contribution by State")
        state_revenue = hist_df.groupby('State')['TotalAmount'].sum().reset_index().sort_values('TotalAmount', ascending=False)
        fig_state = px.bar(
            state_revenue.head(15), x='TotalAmount', y='State', orientation='h',
            labels={'TotalAmount': 'Revenue ($)', 'State': 'State'},
            color='TotalAmount', color_continuous_scale='Purples'
        )
        fig_state.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fig_state, use_container_width=True)
        
    with col4:
        st.write("#### Order Shares by Payment Method")
        pay_shares = hist_df['PaymentMethod'].value_counts().reset_index()
        fig_pay = px.pie(
            pay_shares, values='count', names='PaymentMethod',
            color_discrete_sequence=px.colors.sequential.Bluyl
        )
        st.plotly_chart(fig_pay, use_container_width=True)
        
    st.markdown("---")
    
    # 3. Row: Top Categories & Top Products
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("#### Sales Contribution by Category (Units)")
        cat_units = hist_df.groupby('Category')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False)
        fig_cat = px.bar(cat_units, x='Category', y='Quantity', color='Category', color_discrete_sequence=px.colors.qualitative.T10)
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col6:
        st.write("#### Top 10 Best Selling Products by Quantity")
        top_prod_sales = hist_df.groupby('ProductName')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False).head(10)
        # Shorten product name to keep chart tidy
        top_prod_sales['ProductName_Short'] = top_prod_sales['ProductName'].apply(lambda x: x[:25] + "...")
        fig_top_prod = px.bar(
            top_prod_sales, x='Quantity', y='ProductName_Short', orientation='h',
            color='Quantity', color_continuous_scale='Oranges',
            hover_data=['ProductName']
        )
        fig_top_prod.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fig_top_prod, use_container_width=True)

# -------------------------------------------------------------
# SEASONALITY ANALYSIS PAGE
# -------------------------------------------------------------
elif page == "📅 Seasonality Analysis":
    st.title("⏳ Demand Seasonality Analysis")
    st.subheader("Detect peak and off-season cycles to guide warehouse and inventory decisions.")
    
    st.markdown("---")
    
    # Toggle Category or product analysis
    seas_type = st.radio("Select analysis hierarchy level:", ["Category", "Top Product"], horizontal=True)
    
    if seas_type == "Category":
        if 'cat_seasonality' not in data:
            st.warning("Category seasonality data not computed. Run pipeline first.")
        else:
            cat_list = data['cat_seasonality']['Category'].unique()
            selected_cat = st.selectbox("Choose Category:", cat_list)
            
            cat_data = data['cat_seasonality'][data['cat_seasonality']['Category'] == selected_cat]
            sub = cat_data.iloc[0]
            
            # Recommendation Banner
            st.info(f"💡 **Business Recommendation**: {sub['BusinessRecommendation']}")
            
            cols = st.columns(4)
            cols[0].metric("Peak Demand Month", sub['PeakMonth'])
            cols[1].metric("Lowest Demand Month", sub['LowestMonth'])
            cols[2].metric("Year-over-Year Growth", f"{sub['YoYGrowth']*100:.1f}%")
            cols[3].metric("Demand Variability (CV)", f"{sub['DemandVariability']:.2f}")
            
            # Seasonality Indices Plot
            st.write(f"#### Monthly Seasonality Index Curve: {selected_cat}")
            fig = px.line(
                cat_data.sort_values('MonthNum'), x='MonthName', y='SeasonalityIndex',
                line_shape='spline', markers=True, color_discrete_sequence=['#ff9900']
            )
            # Add line at index = 1.0 (average baseline)
            fig.add_hline(y=1.0, line_dash="dash", line_color="#232f3e", annotation_text="Baseline (Average Demand)")
            fig.update_layout(xaxis_title="Month", yaxis_title="Seasonality Index")
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        if 'prod_seasonality' not in data:
            st.warning("Product seasonality data not computed. Run pipeline first.")
        else:
            prod_ids = data['prod_seasonality']['ProductID'].unique()
            
            # Create a selection display mapping name to ID
            prod_meta = data['prod_seasonality'].groupby('ProductID').agg({'ProductName': 'first'}).reset_index()
            prod_meta['Display'] = prod_meta['ProductID'] + " - " + prod_meta['ProductName'].apply(lambda x: x[:40])
            
            selected_display = st.selectbox("Choose Top Product:", prod_meta['Display'])
            selected_pid = prod_meta[prod_meta['Display'] == selected_display]['ProductID'].iloc[0]
            
            prod_data = data['prod_seasonality'][data['prod_seasonality']['ProductID'] == selected_pid]
            sub = prod_data.iloc[0]
            
            # Recommendation Banner
            biz_rec = str(sub.get('BusinessRecommendation', 'N/A') or 'N/A')
            st.warning(f"🚨 **Inventory Plan Action**: {biz_rec}")
            
            off_season_raw = sub.get('OffSeason', None)
            off_season_str = str(off_season_raw) if (off_season_raw and str(off_season_raw) != 'nan') else 'None'
            off_season_display = off_season_str[:20] + ("..." if len(off_season_str) > 20 else "")
            
            cols = st.columns(4)
            cols[0].metric("Peak Month", sub['PeakMonth'])
            cols[1].metric("Off-Season Months", off_season_display)
            cols[2].metric("Trend Status", sub['Trend'])
            cols[3].metric("Seasonality Strength (Std)", f"{sub['SeasonalityStrength']:.2f}")
            
            # Seasonality Index plot
            st.write(f"#### Monthly Seasonality Index Curve: {sub['ProductName']}")
            fig = px.line(
                prod_data.sort_values('MonthNum'), x='MonthName', y='SeasonalityIndex',
                line_shape='spline', markers=True, color_discrete_sequence=['#008296']
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#111", annotation_text="Baseline")
            fig.update_layout(xaxis_title="Month", yaxis_title="Seasonality Index")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# FORECASTING PAGE
# -------------------------------------------------------------
elif page == "📈 Forecasting (2025–2026)":
    st.title("🔮 Demand Demand Forecasting (2026)")
    st.subheader("Generate and export forecasted demand quantities and revenue for 2026.")
    
    st.markdown("---")
    
    fc_type = st.radio("Select Forecasting Hierarchy Level:", ["Category Level", "Top Product Level"], horizontal=True)
    
    if fc_type == "Category Level":
        if 'cat_fc' not in data:
            st.warning("Category forecasts data not computed.")
        else:
            cat_list = data['cat_fc']['Category'].unique()
            selected_cat = st.selectbox("Select Category for Forecast:", cat_list)
            
            # Fetch historical
            cat_hist = hist_df[hist_df['Category'] == selected_cat].groupby(hist_df['OrderDate'].dt.to_period('M')).agg({'Quantity': 'sum', 'TotalAmount': 'sum'}).reset_index()
            cat_hist['OrderMonth'] = cat_hist['OrderDate'].dt.to_timestamp()
            
            cat_fc_data = data['cat_fc'][data['cat_fc']['Category'] == selected_cat].sort_values('OrderMonth')
            
            selected_model = cat_fc_data['BestModel_Revenue'].iloc[0] if 'BestModel_Revenue' in cat_fc_data.columns else "Prophet"
            st.success(f"✔️ **Optimal Selected Model**: Runs **{selected_model}/Prophet** (Lowest RMSE on testing period).")
            
            # Plot Quantity Forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cat_hist['OrderMonth'], y=cat_hist['Quantity'], name='Historical Sales', line=dict(color='#232f3e')))
            fig.add_trace(go.Scatter(x=cat_fc_data['OrderMonth'], y=cat_fc_data['Forecast_Sales'], name='Forecast 2026', line=dict(color='#ff9900', dash='dash')))
            fig.add_trace(go.Scatter(
                x=pd.concat([cat_fc_data['OrderMonth'], cat_fc_data['OrderMonth'][::-1]]),
                y=pd.concat([cat_fc_data['Forecast_Sales_Upper'], cat_fc_data['Forecast_Sales_Lower'][::-1]]),
                fill='toself', fillcolor='rgba(255, 153, 0, 0.15)', line=dict(color='rgba(0,0,0,0)'), name='95% Confidence Interval'
            ))
            fig.update_layout(title=f"Category '{selected_cat}' - Sales Quantity Forecast", xaxis_title="Timeline", yaxis_title="Quantity (Units)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Export CSV
            export_df = cat_fc_data[['OrderMonth', 'Category', 'Forecast_Sales', 'Forecast_Sales_Lower', 'Forecast_Sales_Upper', 'Forecast_Revenue', 'Forecast_Revenue_Lower', 'Forecast_Revenue_Upper']]
            csv = export_df.to_csv(index=False)
            st.download_button(
                label=f"💾 Download Forecast dataset as CSV ({selected_cat})",
                data=csv,
                file_name=f"amazon_forecast_2026_{selected_cat.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
    else:
        if 'prod_fc' not in data:
            st.warning("Product forecasts data not computed.")
        else:
            # Let's map display values
            prod_ids = data['prod_fc']['ProductID'].unique()
            prod_meta = data['prod_fc'].groupby('ProductID').agg({'ProductName': 'first'}).reset_index()
            prod_meta['Display'] = prod_meta['ProductID'] + " - " + prod_meta['ProductName'].apply(lambda x: x[:40])
            
            selected_display = st.selectbox("Select Product to Forecast:", prod_meta['Display'])
            selected_pid = prod_meta[prod_meta['Display'] == selected_display]['ProductID'].iloc[0]
            
            p_data_fc = data['prod_fc'][data['prod_fc']['ProductID'] == selected_pid].sort_values('OrderMonth')
            
            # Historical
            prod_hist = hist_df[hist_df['ProductID'] == selected_pid].groupby(hist_df['OrderDate'].dt.to_period('M')).agg({'Quantity': 'sum', 'TotalAmount': 'sum'}).reset_index()
            prod_hist['OrderMonth'] = prod_hist['OrderDate'].dt.to_timestamp()
            
            # Plot Quantity
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prod_hist['OrderMonth'], y=prod_hist['Quantity'], name='Historical Sales', line=dict(color='#232f3e')))
            fig.add_trace(go.Scatter(x=p_data_fc['OrderMonth'], y=p_data_fc['Forecast_Sales'], name='Forecast 2026', line=dict(color='#008296', dash='dash')))
            fig.add_trace(go.Scatter(
                x=pd.concat([p_data_fc['OrderMonth'], p_data_fc['OrderMonth'][::-1]]),
                y=pd.concat([p_data_fc['Forecast_Sales_Upper'], p_data_fc['Forecast_Sales_Lower'][::-1]]),
                fill='toself', fillcolor='rgba(0, 130, 150, 0.15)', line=dict(color='rgba(0,0,0,0)'), name='95% Confidence Interval'
            ))
            fig.update_layout(title=f"Product '{selected_pid}' Sales Quantity Forecast", xaxis_title="Timeline", yaxis_title="Quantity (Units)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Export CSV
            export_df = p_data_fc[['OrderMonth', 'ProductID', 'ProductName', 'Category', 'Forecast_Sales', 'Forecast_Sales_Lower', 'Forecast_Sales_Upper', 'Forecast_Revenue', 'Forecast_Revenue_Lower', 'Forecast_Revenue_Upper']]
            csv = export_df.to_csv(index=False)
            st.download_button(
                label=f"💾 Download Forecast dataset as CSV (Product {selected_pid})",
                data=csv,
                file_name=f"amazon_forecast_2026_product_{selected_pid}.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------
# INVENTORY PLANNER PAGE
# -------------------------------------------------------------
elif page == "📦 Inventory Planner":
    st.title("📦 Inventory Optimization Planner")
    st.subheader("Data-driven inventory stocking recommendations, safety buffers, and procurement timelines.")
    
    st.markdown("---")
    
    if 'inv_plan' not in data:
        st.warning("Inventory recommendations data not completed.")
    else:
        inv_df = data['inv_plan'].copy()
        
        # Sidebar/Top Page Filters
        c_filter = st.multiselect("Filter by Category:", list(inv_df['Category'].unique()), default=list(inv_df['Category'].unique()))
        p_filter = st.multiselect("Filter by Priority Level:", list(inv_df['Priority'].unique()), default=list(inv_df['Priority'].unique()))
        s_filter = st.multiselect("Filter by Recommendation Status:", list(inv_df['InventoryStatus'].unique()), default=list(inv_df['InventoryStatus'].unique()))
        
        # Filter logic
        filtered_df = inv_df[
            (inv_df['Category'].isin(c_filter)) &
            (inv_df['Priority'].isin(p_filter)) &
            (inv_df['InventoryStatus'].isin(s_filter))
        ]
        
        st.write(f"Showing **{len(filtered_df)}** records matching criteria.")
        
        # Metrics summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Highest Priority Alerts", len(filtered_df[filtered_df['Priority'] == 'Critical']))
        col2.metric("Products Requiring Stock Increase", len(filtered_df[filtered_df['InventoryStatus'] == 'Increase Inventory']))
        col3.metric("Total Recommended Safety Buffer (Units)", int(filtered_df['SafetyStock'].sum()))
        
        # Display recommendations table
        st.write("#### Recommended Monthly Stocking Strategy (2026)")
        
        display_df = filtered_df[[
            'ProductID', 'ProductName', 'Category', 'ForecastMonthName', 'Forecast_Sales', 
            'HistMeanSales', 'SafetyStock', 'InventoryStatus', 'Priority', 'ProcurementMonthName'
        ]].rename(columns={
            'ForecastMonthName': 'Target Month',
            'Forecast_Sales': 'Forecasted Demand',
            'HistMeanSales': 'Hist Avg Demand',
            'ProcurementMonthName': 'Restock Deadline'
        })
        
        # Paginate or format dataframe cleanly
        st.dataframe(display_df.style.map(
            lambda val: "color:red; font-weight:bold" if val == 'Critical' else ("color:orange; font-weight:bold" if val == 'High' else ("color:green; font-weight:bold" if val == 'Increase Inventory' else "")),
            subset=['Priority', 'InventoryStatus']
        ), use_container_width=True)
        
        # Download Inventory Plan
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Complete Inventory Plan as CSV",
            data=csv,
            file_name="amazon_inventory_plan_2026.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# BUSINESS INSIGHTS PAGE
# -------------------------------------------------------------
elif page == "💡 Business Insights":
    st.title("💡 Dynamic Business Insights & Strategic Guidelines")
    st.subheader("Data-driven recommendations and actions dynamically generated by the analysis engines.")
    
    st.markdown("---")
    
    if 'insights' not in data:
        st.warning("Business insights data has not been generated.")
    else:
        ins_df = data['insights'].copy()
        
        # Filter insights by priority
        selected_prios = st.multiselect("Filter by Priority Level:", list(ins_df['Priority'].unique()), default=list(ins_df['Priority'].unique()))
        ins_filtered = ins_df[ins_df['Priority'].isin(selected_prios)]
        
        # Metrics summary
        st.markdown(f"Total dynamic insights matching filter: **{len(ins_filtered)}** (at least 25+ guaranteed).")
        
        # Render insights beautifully using container boxes
        for idx, row in ins_filtered.iterrows():
            badge_class = f"badge-{row['Priority'].lower()}"
            
            st.markdown(f"""
                <div style="background-color: #f7f9fa; border-left: 5px solid { '#d9534f' if row['Priority']=='Critical' else ('#f0ad4e' if row['Priority']=='High' else '#5bc0de') }; padding: 15px; border-radius: 4px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.15rem; font-weight: bold; color: #232f3e;">{row['Icon']} {row['Title']}</span>
                        <span class="{badge_class}">{row['Priority']}</span>
                    </div>
                    <div style="margin-top: 8px; color: #333333; font-size: 0.95rem; line-height: 1.5;">
                        {row['Description']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        # Download Insights Plan
        csv = ins_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Business Insights Log as CSV",
            data=csv,
            file_name="amazon_business_insights_2026.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# MODEL COMPARISON PAGE
# -------------------------------------------------------------
elif page == "🤖 Model Comparison":
    st.title("🎛️ Model Benchmarking & Testing metrics")
    st.subheader("Performance metrics across Prophet, Holt-Winters, SARIMA, and XGBoost models.")
    
    st.markdown("---")
    
    if 'comparison' not in data:
        st.warning("Model comparison statistics not computed. Run pipeline first.")
    else:
        comp_df = data['comparison'].copy()
        
        # Table of metrics
        st.write("#### Validation Metrics (Tested on 12-Month Out-of-Sample Holdout)")
        
        # Show cleaner dataframe
        st.dataframe(comp_df.style.highlight_min(axis=0, subset=['MAE', 'RMSE', 'MAPE'], color='#e0f2f1')
                             .highlight_max(axis=0, subset=['R2'], color='#e0f2f1'), use_container_width=True)
        
        # Aggregate Best Models selected
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Selected Best Model Count")
            # Best model is selected based on lowest RMSE per series
            best_model_agg = comp_df.loc[comp_df.groupby('Series')['RMSE'].idxmin()]
            best_count = best_model_agg['Model'].value_counts().reset_index()
            fig = px.bar(best_count, x='Model', y='count', color='Model', color_discrete_sequence=px.colors.qualitative.T10)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("#### Average Training/Inference Speed by Model (Seconds)")
            avg_time = comp_df.groupby('Model')['TrainingTime'].mean().reset_index().sort_values('TrainingTime')
            fig_time = px.bar(avg_time, x='TrainingTime', y='Model', orientation='h', color_discrete_sequence=['#232f3e'])
            st.plotly_chart(fig_time, use_container_width=True)

# -------------------------------------------------------------
# ABOUT PROJECT PAGE
# -------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About the Platform Architecture")
    st.subheader("Implementation outline of the production-level forecasting engine.")
    
    st.markdown("---")
    
    st.write("### Hierarchical Architecture Diagram")
    
    # Render hierarchical layout
    st.code("""
    [Amazon Transaction Dataset (5 Years historical)]
                           │
                           ▼
                 [Data Preprocessing]
           (Clean, Date Engineer, Outlier Handling)
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       [Level 1]      [Level 2]      [Level 3]
       Categories    Top Products   Remaining Products
      (Prophet Fit)  (Prophet/HW)   (XGBoost Model)
            │              │              │
            └──────────────┼──────────────┘
                           ▼
            [Inventory Recommendation Engine]
            (Safety Stock, Lead Times, Priorities)
                           │
                           ▼
          [Streamlit UI & Actionable Reports]
    """)
    
    st.write("### Model Specifications & Fallback Strategy")
    st.markdown("""
        - **Level 1 (Category)**: Prophet fits seasonal patterns and business changes for aggregated product classes (Electronics, Home, Beauty, etc.).
        - **Level 2 (Top Products)**: Top 30 selling items are forecasted using a localized Prophet or Holt-Winters model.
        - **Level 3 (Remaining Tail Products)**: For tail products with sufficient data, XGBoost maps historical monthly lags (1, 2, 3, 12), rolling statistics, sine/cosine calendar components, and linear trend index.
        - **Sparse Fallback Plan**: If a product has less than 12 months of non-zero historical records, its future demand paths are computed directly as a proportion/share of the parent Category forecast, protecting predictions from numerical instability.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Executive Summary":
    st.title("📋 AI Executive Summary Generator")
    st.subheader("Automated management-level business report for 2025–2026.")
    st.markdown("---")

    exec_df = data.get('exec_summary')
    cat_fc  = data.get('cat_fc')
    biz_fc  = data.get('biz_fc')
    hist_df = data['hist']

    if exec_df is None or exec_df.empty:
        st.warning("Executive summary not found. Please re-run `run_pipeline.py` to generate it.")
    else:
        # ── KPI Headline Row ─────────────────────────────────────────────────
        r25 = exec_df[exec_df['ForecastYear'] == 2025].iloc[0] if 2025 in exec_df['ForecastYear'].values else None
        r26 = exec_df[exec_df['ForecastYear'] == 2026].iloc[0] if 2026 in exec_df['ForecastYear'].values else None

        c1, c2, c3, c4 = st.columns(4)
        if r25 is not None:
            c1.metric("2025 Forecast Revenue", f"${r25['ForecastRevenue']/1e6:.2f}M", f"{r25['OverallGrowthPct']:+.1f}% vs baseline")
            c2.metric("2025 Forecast Units",   f"{r25['ForecastQuantity']:,.0f}", "")
        if r26 is not None:
            c3.metric("2026 Forecast Revenue", f"${r26['ForecastRevenue']/1e6:.2f}M", f"{r26['OverallGrowthPct']:+.1f}% vs baseline")
            c4.metric("2026 Forecast Units",   f"{r26['ForecastQuantity']:,.0f}", "")

        st.markdown("---")
        # ── Revenue Trend Chart ──────────────────────────────────────────────
        if cat_fc is not None:
            st.subheader("📈 Monthly Revenue Forecast — 2025 & 2026")
            biz_monthly = cat_fc.copy()
            biz_monthly['OrderMonth'] = pd.to_datetime(biz_monthly['OrderMonth'])
            agg = biz_monthly.groupby('OrderMonth')['Forecast_Revenue'].sum().reset_index()
            agg['Year'] = agg['OrderMonth'].dt.year.astype(str)
            fig = px.area(agg, x='OrderMonth', y='Forecast_Revenue', color='Year',
                          color_discrete_map={'2025': '#ff9900', '2026': '#232f3e'},
                          labels={'Forecast_Revenue': 'Revenue ($)', 'OrderMonth': 'Month'},
                          template='plotly_white')
            fig.update_layout(height=380, legend_title_text='Forecast Year')
            st.plotly_chart(fig, use_container_width=True)

        # ── Top / Declining Categories ────────────────────────────────────────
        if cat_fc is not None:
            hist_df_yr = hist_df[hist_df['OrderDate'].dt.year == hist_df['OrderDate'].dt.year.max()]
            hist_cat_rev = hist_df_yr.groupby('Category')['TotalAmount'].sum()
            col_left, col_right = st.columns(2)
            for yr, col in [(2025, col_left), (2026, col_right)]:
                with col:
                    st.markdown(f"**Top Categories — {yr}**")
                    fc_yr = cat_fc[cat_fc['OrderMonth'].dt.year == yr]
                    cat_rev = fc_yr.groupby('Category')['Forecast_Revenue'].sum()
                    growth = ((cat_rev - hist_cat_rev) / hist_cat_rev.clip(lower=1) * 100).dropna().sort_values(ascending=False)
                    df_g = growth.reset_index(); df_g.columns = ['Category', 'Growth %']
                    df_g['Growth %'] = df_g['Growth %'].round(1)
                    st.dataframe(df_g, use_container_width=True, hide_index=True)

        # ── Opportunities & Risks ─────────────────────────────────────────────
        st.markdown("---")
        r26_info = r26 if r26 is not None else {}
        col_opp, col_risk = st.columns(2)
        with col_opp:
            st.subheader("🚀 Top Opportunities")
            opps = []
            if cat_fc is not None:
                fc26 = cat_fc[cat_fc['OrderMonth'].dt.year == 2026]
                hist_yr = hist_df[hist_df['OrderDate'].dt.year == hist_df['OrderDate'].dt.year.max()]
                cat_g = ((fc26.groupby('Category')['Forecast_Revenue'].sum() -
                          hist_yr.groupby('Category')['TotalAmount'].sum()) /
                         hist_yr.groupby('Category')['TotalAmount'].sum().clip(lower=1) * 100)
                for cat, g in cat_g.sort_values(ascending=False).head(5).items():
                    opps.append(f"📈 **{cat}** projected +{g:.1f}% — expand procurement budget.")
            for o in opps:
                st.markdown(o)

        with col_risk:
            st.subheader("⚠️ Key Risks")
            inv = data.get('inv_plan') if data.get('inv_plan') is not None else data.get('cat_inv_plan')
            if inv is not None and 'Priority' in inv.columns:
                critical = inv[inv['Priority'] == 'Critical']
                id_col = 'ProductID' if 'ProductID' in critical.columns else 'Category'
                for item in critical[id_col].unique()[:5]:
                    st.markdown(f"🚨 **{item}** — Critical inventory risk. Immediate action needed.")
            else:
                st.info("No critical risk items identified.")

        # ── Executive Conclusion ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📝 Executive Conclusion")
        if r26 is not None:
            st.markdown(f"""
> The Amazon platform is forecast to achieve **{r26['OverallGrowthPct']:+.1f}% revenue growth** in 2026 compared to the historical baseline.  
> **{r26['HighestGrowthCategory']}** leads category expansion.  
> Peak revenue month: **{r26['HighestRevenueMonth']}**.  
> **Key actions:** Prioritize procurement for high-growth categories. Maintain safety stock buffers for volatile products.  
> Leverage seasonal peaks to maximise revenue capture.
            """)

        # ── Export ───────────────────────────────────────────────────────────
        st.markdown("---")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            csv = exec_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Executive Summary (CSV)", csv, "executive_summary.csv", "text/csv")
        with c_dl2:
            if cat_fc is not None:
                csv2 = cat_fc.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Category Forecast 2025–2026 (CSV)", csv2, "category_forecast_2025_2026.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# AI BUSINESS ASSISTANT PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 AI Business Assistant":
    st.title("🧠 AI Business Assistant")
    st.subheader("Ask natural language questions about the 2025–2026 forecast data.")
    st.markdown("---")

    # Initialise session chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        import sys, os
        sys.path.insert(0, '.')
        from src.ai_business_assistant import AIBusinessAssistant

        @st.cache_resource
        def get_assistant():
            import pandas as pd
            hist   = pd.read_csv("Data/amazon_transactions_clean.csv")
            hist['OrderDate'] = pd.to_datetime(hist['OrderDate'])
            cat_fc = pd.read_csv("outputs/category_forecast.csv"); cat_fc['OrderMonth'] = pd.to_datetime(cat_fc['OrderMonth'])
            prod_fc= pd.read_csv("outputs/product_forecast.csv"); prod_fc['OrderMonth'] = pd.to_datetime(prod_fc['OrderMonth'])
            cat_s  = pd.read_csv("reports/category_seasonality.csv")
            prod_s = pd.read_csv("reports/product_seasonality.csv")
            inv    = pd.read_csv("outputs/inventory_plan.csv"); inv['OrderMonth'] = pd.to_datetime(inv['OrderMonth'])
            ins    = pd.read_csv("reports/business_insights.csv")
            mc     = pd.read_csv("reports/model_comparison.csv")
            return AIBusinessAssistant(hist, cat_fc, prod_fc, cat_s, prod_s, inv, ins, mc)

        assistant = get_assistant()
        assistant_ready = True
    except Exception as e:
        assistant_ready = False
        st.error(f"Could not load AI Assistant: {e}. Please re-run `run_pipeline.py` first.")

    # Suggested Questions
    st.markdown("**💬 Try asking:**")
    suggested = [
        "Which categories will grow the most in 2026?",
        "What are the biggest inventory risks?",
        "Show me the top investment opportunities",
        "Summarise Electronics forecast",
        "Which products have the highest safety stock requirements?",
        "What is the highest revenue month?",
        "What does the 2025 forecast look like?",
        "Give me an executive summary",
    ]
    cols_sugg = st.columns(4)
    for i, q in enumerate(suggested):
        if cols_sugg[i % 4].button(q, key=f"sugg_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": q})
            if assistant_ready:
                response = assistant.answer(q)
                st.session_state.chat_history.append({"role": "assistant", "response": response})

    # Chat Input
    user_q = st.chat_input("Type your business question here…")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        if assistant_ready:
            response = assistant.answer(user_q)
            st.session_state.chat_history.append({"role": "assistant", "response": response})

    # Render Chat History
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant" and "response" in msg:
            with st.chat_message("assistant"):
                resp = msg["response"]
                cols = st.columns(2)
                fields_left  = ["Forecast", "Explanation", "Seasonality", "Business Insight"]
                fields_right = ["Inventory Recommendation", "Risk Level", "Confidence Score", "Suggested Action"]
                icons = {"Forecast": "📊", "Explanation": "💬", "Seasonality": "📅", "Business Insight": "💡",
                         "Inventory Recommendation": "📦", "Risk Level": "⚠️", "Confidence Score": "🎯", "Suggested Action": "✅"}
                with cols[0]:
                    for f in fields_left:
                        if f in resp and resp[f] and resp[f] != "N/A":
                            st.markdown(f"**{icons.get(f,'')} {f}:**\n\n{resp[f]}")
                            st.markdown("---")
                with cols[1]:
                    for f in fields_right:
                        if f in resp and resp[f] and resp[f] != "N/A":
                            st.markdown(f"**{icons.get(f,'')} {f}:**\n\n{resp[f]}")
                            st.markdown("---")

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# WHAT-IF ANALYSIS PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 What-If Analysis":
    st.title("🔮 What-If Scenario Simulator")
    st.subheader("Adjust growth assumptions and see how they impact 2025–2026 revenue forecasts.")
    st.markdown("---")

    cat_fc = data.get('cat_fc')
    if cat_fc is None:
        st.warning("Forecast data not available. Please run `run_pipeline.py` first.")
    else:
        cat_fc = cat_fc.copy()
        cat_fc['OrderMonth'] = pd.to_datetime(cat_fc['OrderMonth'])
        categories = sorted(cat_fc['Category'].unique())

        st.sidebar.markdown("### 🔮 Scenario Controls")

        col_ctrl, col_chart = st.columns([1, 2])

        with col_ctrl:
            st.markdown("#### Scenario Settings")
            year_sel   = st.selectbox("Forecast Year to Simulate", [2025, 2026])
            demand_adj = st.slider("Demand Growth Adjustment (%)", -50, 100, 0, 5,
                                   help="Positive = higher demand than forecast. Negative = lower demand.")
            price_adj  = st.slider("Average Price Change (%)", -30, 50, 0, 5,
                                   help="Simulates impact of pricing changes on revenue.")
            cat_sel    = st.multiselect("Focus Categories (leave empty = all)", categories)
            scenario_name = st.text_input("Scenario Name", value="Base Scenario")
            run_btn    = st.button("▶ Run Scenario", type="primary")

        with col_chart:
            fc_yr = cat_fc[cat_fc['OrderMonth'].dt.year == year_sel].copy()
            if cat_sel:
                fc_yr_scenario = fc_yr[fc_yr['Category'].isin(cat_sel)].copy()
            else:
                fc_yr_scenario = fc_yr.copy()

            fc_yr_scenario['Adj_Sales']   = fc_yr_scenario['Forecast_Sales'] * (1 + demand_adj / 100)
            fc_yr_scenario['Adj_Revenue'] = fc_yr_scenario['Forecast_Revenue'] * (1 + demand_adj / 100) * (1 + price_adj / 100)

            base_rev = fc_yr_scenario['Forecast_Revenue'].sum()
            adj_rev  = fc_yr_scenario['Adj_Revenue'].sum()
            delta_rev = adj_rev - base_rev
            delta_pct = (delta_rev / max(base_rev, 1)) * 100

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Base Revenue", f"${base_rev/1e6:.2f}M")
            sc2.metric(f"Scenario Revenue ({scenario_name})", f"${adj_rev/1e6:.2f}M", f"{delta_pct:+.1f}%")
            sc3.metric("Revenue Delta", f"${delta_rev/1e6:+.2f}M")

            # Monthly comparison chart
            agg_base = fc_yr_scenario.groupby('OrderMonth')['Forecast_Revenue'].sum().reset_index()
            agg_base['Scenario'] = 'Baseline'
            agg_adj  = fc_yr_scenario.groupby('OrderMonth')['Adj_Revenue'].sum().reset_index()
            agg_adj.columns = ['OrderMonth', 'Forecast_Revenue']
            agg_adj['Scenario'] = scenario_name or 'Adjusted'

            combined = pd.concat([agg_base, agg_adj], ignore_index=True)
            fig = px.line(combined, x='OrderMonth', y='Forecast_Revenue', color='Scenario',
                          color_discrete_map={'Baseline': '#232f3e', (scenario_name or 'Adjusted'): '#ff9900'},
                          labels={'Forecast_Revenue': 'Revenue ($)', 'OrderMonth': 'Month'},
                          title=f"Baseline vs {scenario_name} — {year_sel} Monthly Revenue",
                          template='plotly_white')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Category-Level Impact")
        cat_impact = fc_yr_scenario.groupby('Category').agg(
            BaseRevenue=('Forecast_Revenue', 'sum'),
            AdjRevenue=('Adj_Revenue', 'sum')
        ).reset_index()
        cat_impact['RevenueChange'] = cat_impact['AdjRevenue'] - cat_impact['BaseRevenue']
        cat_impact['ChangePct'] = (cat_impact['RevenueChange'] / cat_impact['BaseRevenue'].clip(lower=1) * 100).round(1)
        cat_impact = cat_impact.sort_values('RevenueChange', ascending=False)

        fig2 = px.bar(cat_impact, x='Category', y='RevenueChange', color='ChangePct',
                      color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                      title="Revenue Delta by Category",
                      labels={'RevenueChange': 'Revenue Δ ($)', 'ChangePct': 'Δ %'},
                      template='plotly_white')
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(cat_impact.rename(columns={
            'BaseRevenue': 'Base Revenue ($)', 'AdjRevenue': f'Adj Revenue – {scenario_name} ($)',
            'RevenueChange': 'Δ Revenue ($)', 'ChangePct': 'Δ %'
        }), use_container_width=True, hide_index=True)

        # ── Export ─────────────────────────────────────────────────────────
        csv_out = fc_yr_scenario[['Category', 'OrderMonth', 'Forecast_Sales', 'Adj_Sales', 'Forecast_Revenue', 'Adj_Revenue']].copy()
        csv_out.columns = ['Category', 'Month', 'Baseline_Units', 'Scenario_Units', 'Baseline_Revenue', 'Scenario_Revenue']
        st.download_button(
            "⬇️ Export Scenario Results (CSV – Power BI Ready)",
            csv_out.to_csv(index=False).encode('utf-8'),
            f"whatif_{scenario_name.replace(' ', '_')}_{year_sel}.csv",
            "text/csv"
        )

