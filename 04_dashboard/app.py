# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Retail AI & Business Health Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA LOADING & ENRICHMENT ---
@st.cache_data
def load_and_prepare_data():
    # Define paths using a robust method
    project_root = Path(__file__).parent.parent
    customer_data_path = project_root / "01_data" / "processed" / "final_dashboard_data.csv"
    transaction_data_path = project_root / "01_data" / "processed" / "cleaned_retail_data.csv"
    
    try:
        # Load the core datasets
        customer_df = pd.read_csv(customer_data_path)
        transaction_df = pd.read_csv(transaction_data_path, parse_dates=['InvoiceDate'])
        
        # --- Data Enrichment ---
        # Add Country to each customer from their last transaction
        customer_country_map = transaction_df.sort_values('InvoiceDate').drop_duplicates('Customer ID', keep='last')[['Customer ID', 'Country']]
        customer_df = pd.merge(customer_df, customer_country_map, on='Customer ID', how='left')

        # Create 'Segment' Column from RFM_Score
        score_bins = [0, 6, 8, 10, 12]
        score_labels = ['Hibernating', 'Needs Attention', 'Loyal Customers', 'Champions']
        customer_df['Segment'] = pd.cut(customer_df['RFM_Score'], bins=score_bins, labels=score_labels)
        
        # Set the order for the 'Segment' column
        segment_order = ['Champions', 'Loyal Customers', 'Needs Attention', 'Hibernating']
        customer_df['Segment'] = pd.Categorical(customer_df['Segment'], categories=segment_order, ordered=True)
        
        # Calculate necessary date columns
        snapshot_date = pd.to_datetime('2010-12-10')  # Based on dataset's last transaction
        customer_df['Last_Purchase_Date'] = snapshot_date - pd.to_timedelta(customer_df['Recency'], unit='d')
        customer_df['Churn_Date'] = customer_df['Last_Purchase_Date'] + pd.to_timedelta(180, unit='d')
        customer_df['Acquisition_Month'] = customer_df.groupby('Customer ID')['Last_Purchase_Date'].transform('min').dt.to_period('M').dt.to_timestamp()
        
        return customer_df, transaction_df
        
    except FileNotFoundError:
        st.error("FATAL ERROR: Could not find necessary data files in '01_data/processed/'.")
        st.error("Please ensure both 'final_dashboard_data.csv' and 'cleaned_retail_data.csv' exist.")
        return None, None

customer_df, transaction_df = load_and_prepare_data()

# --- 3. HELPER FUNCTIONS ---
def create_gauge_chart(value, title, color, ranges, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [ranges[0], ranges[1]]},
            'bar': {'color': color},
            'steps': [
                {'range': [ranges[0], ranges[1]/2], 'color': "lightgreen"},
                {'range': [ranges[1]/2, ranges[1]], 'color': "yellow"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def create_segment_donut(segment_counts):
    fig = px.pie(
        segment_counts,
        values='count',
        names=segment_counts.index,
        title="Customer Segment Mix",
        hole=0.5,
        color=segment_counts.index,
        color_discrete_map={
            'Champions': '#FFD700',
            'Loyal Customers': '#4169E1',
            'Needs Attention': '#778899',
            'Hibernating': '#8B0000'
        }
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

# --- 4. MAIN APPLICATION ---




if customer_df is not None and transaction_df is not None:
    st.title("🚀 AI-Powered Retail Analytics Dashboard")
    st.markdown("A comprehensive overview of customer segmentation, sales performance, and predictive churn insights.")
    st.markdown("---")

    # --- ENHANCED TOP KPI BAR ---
    st.header("Key Performance Indicators")
    total_revenue = customer_df['MonetaryValue'].sum()
    total_acquisitions = customer_df['Customer ID'].nunique()
    total_ceased = customer_df[customer_df['Churn_Status'] == 1]['Customer ID'].nunique()
    churn_rate = (total_ceased / total_acquisitions) * 100
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Revenue", f"${total_revenue:,.0f}")
    kpi_cols[1].metric("Total Customers", f"{total_acquisitions:,}")
    kpi_cols[2].metric("Churned Customers", f"{total_ceased:,}")
    kpi_cols[3].metric("Churn Rate", f"{churn_rate:.1f}%",
                        delta_color="inverse",
                        delta=f"{(churn_rate - 25):.1f}% vs benchmark" if churn_rate else None)

    st.markdown("---")

    # --- CUSTOMER HEALTH & SEGMENTATION OVERVIEW ---
    st.header("Customer Health & Segmentation")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_gauge = create_gauge_chart(
            value=churn_rate,
            title="Current Churn Rate (%)",
            color="darkred",
            ranges=[0, 50],
            threshold=25
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        retention_rate = 100 - churn_rate
        fig_gauge_retention = create_gauge_chart(
            value=retention_rate,
            title="Retention Rate (%)",
            color="royalblue",
            ranges=[50, 100],
            threshold=75
        )
        st.plotly_chart(fig_gauge_retention, use_container_width=True)

    with col3:
        segment_counts = customer_df['Segment'].value_counts()
        fig_donut = create_segment_donut(segment_counts)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    
# =============================================================================
# --- Insight: Monthly Performance Summary ---
# This code creates a snapshot for the latest month of data, inspired by your reference.
# =============================================================================

    st.header("📈 Monthly Performance Snapshot")

    # --- 1. Identify the most recent month in the data ---
    latest_month = transaction_df['InvoiceDate'].max().to_period('M').to_timestamp()
    st.subheader(f"Summary for {latest_month.strftime('%B, %Y')}")

    # --- 2. Calculate Metrics for the Latest Month ---
    # a. New Customers Acquired in the latest month
    new_customers_latest_month = customer_df[customer_df['Acquisition_Month'] == latest_month]

    # b. Customers who Churned in the latest month
    # FIX: Added the .dt accessor before the final .to_timestamp()
    churned_latest_month = customer_df[customer_df['Churn_Date'].dt.to_period('M').dt.to_timestamp() == latest_month]

    # c. Existing Customers who were active in the latest month
    active_customers_df = customer_df[customer_df['Churn_Status'] == 0]
    # Existing active are those who were not newly acquired this month
    existing_active_latest_month = active_customers_df[active_customers_df['Acquisition_Month'] != latest_month]


    # d. Revenue calculations
    # Revenue from new customers in their first month
    # FIX: Added the .dt accessor before the final .to_timestamp()
    revenue_new = transaction_df[
        transaction_df['Customer ID'].isin(new_customers_latest_month['Customer ID']) &
        (transaction_df['InvoiceDate'].dt.to_period('M').dt.to_timestamp() == latest_month)
    ]['TotalPrice'].sum()

    # Total lifetime value of customers who were lost this month
    # This requires the 'CLV' column, which should be in customer_df. Let's ensure it's handled.
    revenue_lost = churned_latest_month['CLV'].sum() if 'CLV' in churned_latest_month.columns else 0


    # Revenue generated *this month* by existing active customers
    # FIX: Added the .dt accessor before the final .to_timestamp()
    revenue_existing_active = transaction_df[
        transaction_df['Customer ID'].isin(existing_active_latest_month['Customer ID']) &
        (transaction_df['InvoiceDate'].dt.to_period('M').dt.to_timestamp() == latest_month)
    ]['TotalPrice'].sum()

    # --- 3. Create the Layout and Display Insights ---
    summary_col1, summary_col2 = st.columns([1.5, 1])

    with summary_col1:
        # --- Summary Table ---
        st.markdown("**Performance Summary Table**")
        summary_data = {
            'Category': ['Existing Active', 'New', 'Lost'],
            'Customers': [len(existing_active_latest_month), len(new_customers_latest_month), len(churned_latest_month)],
            'Revenue Impact ($)': [revenue_existing_active, revenue_new, revenue_lost]
        }
        summary_df = pd.DataFrame(summary_data)
        
        st.dataframe(
            summary_df.style.format({
                'Customers': '{:,.0f}',
                'Revenue Impact ($)': '${:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("""
        - **Existing Active:** Customers acquired in previous months who made a purchase this month.
        - **New:** Customers who made their very first purchase this month.
        - **Lost:** Customers whose inactivity period crossed the 180-day churn threshold this month. Revenue Impact shows their total lifetime value.
        """)

    with summary_col2:
        # --- Donut Charts ---
        st.markdown("**This Month's Customer Base**")
        
        # Calculate values for the donut chart
        new_cust_count = len(new_customers_latest_month)
        existing_cust_count = len(existing_active_latest_month)

        fig_donut = go.Figure(data=[go.Pie(
            labels=['New Customers', 'Existing Customers'],
            values=[new_cust_count, existing_cust_count],
            hole=.5,
            marker_colors=['skyblue', 'royalblue']
        )])
        fig_donut.update_layout(title_text="Composition of Active Customers This Month", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)


    # --- Waterfall Chart for Customer Flow ---
    st.markdown("**Customer Flow Waterfall**")
    net_growth = len(new_customers_latest_month) - len(churned_latest_month)
    total_active_end_month = len(customer_df[customer_df['Churn_Status'] == 0])

    fig_waterfall = go.Figure(go.Waterfall(
        name = "Flow", orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["New Customers Gained", "Customers Lost", "Net Monthly Growth"],
        textposition = "outside",
        text = [f"+{len(new_customers_latest_month)}", f"-{len(churned_latest_month)}", f"{net_growth}"],
        y = [len(new_customers_latest_month), -len(churned_latest_month), net_growth],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig_waterfall.update_layout(title = "Monthly Customer Gain vs. Loss", showlegend = False, yaxis_title="Number of Customers")
    st.plotly_chart(fig_waterfall, use_container_width=True)





    

    # --- STRATEGIC OVERVIEW: CHURN RISK VS. SEGMENT VALUE ---
    st.subheader("Which Segments are Most Valuable to Save?")

    # Calculate data for the chart
    risk_analysis_df = customer_df[customer_df['Churn_Status'] == 0].groupby('Segment', observed=True).agg(
        Avg_Churn_Risk=('Churn_Probability', 'mean'),
        Total_Spendings=('MonetaryValue', 'sum'),
        Customer_Count=('Customer ID', 'nunique')
    ).reset_index()
    risk_analysis_df['Avg_Churn_Risk'] = risk_analysis_df['Avg_Churn_Risk'] * 100

    # Create the scatter plot
    fig_risk_scatter = px.scatter(
        risk_analysis_df,
        x='Total_Spendings',
        y='Avg_Churn_Risk',
        size='Customer_Count',
        color='Segment',
        hover_name='Segment',
        hover_data={
            'Segment': False,
            'Total_Spendings': ':$,.0f',
            'Avg_Churn_Risk': ':.1f%',
            'Customer_Count': ':,'
        },
        size_max=70,
        title='Segment Value vs. Average Churn Risk',
        labels={
            "Total_Spendings": "Total Spendings of Segment ($)",
            "Avg_Churn_Risk": "Average Churn Risk (%)",
            "Customer_Count": "Number of Customers"
        },
        color_discrete_map={
            'Champions': 'gold',
            'Loyal Customers': 'royalblue',
            'Needs Attention': 'lightslategray',
            'Hibernating': 'darkred'
        }
    )

    # Enhance the chart's appearance
    fig_risk_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    avg_risk_line = customer_df[customer_df['Churn_Status'] == 0]['Churn_Probability'].mean() * 100
    fig_risk_scatter.add_hline(
        y=avg_risk_line,
        line_dash="dot",
        annotation_text=f"Avg. Active Customer Risk ({avg_risk_line:.1f}%)",
        annotation_position="bottom right"
    )
    avg_value_line = risk_analysis_df['Total_Spendings'].mean()
    fig_risk_scatter.add_vline(x=avg_value_line, line_dash="dot")
    fig_risk_scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(x=0.5),
        legend_title_text='Segment'
    )

    # Display the chart
    st.plotly_chart(fig_risk_scatter, use_container_width=True)
    st.caption("This chart helps prioritize retention efforts. Segments in the top-right quadrant (High Value, High Risk) are the most critical to save.")
    st.markdown("---")

    # --- ENHANCED SEGMENTED RETENTION ANALYSIS ---
    st.subheader("Retention by Customer Segment")
    
    # Data preparation
    chart_df = customer_df.copy()
    chart_df['Status'] = np.where(chart_df['Churn_Status'] == 0, 'Retained', 'Churned')
    segment_retention_table = chart_df.groupby('Segment', observed=True)['Status'].value_counts().unstack().fillna(0)
    segment_retention_table['Retention_Rate'] = (segment_retention_table['Retained'] / (segment_retention_table['Retained'] + segment_retention_table['Churned'])) * 100

    # Sunburst Chart
    fig = px.sunburst(
        chart_df,
        path=['Segment', 'Status'],
        color='Segment',
        color_discrete_map={
            'Champions': '#FFD700',
            'Loyal Customers': '#4169E1',
            'Needs Attention': '#778899',
            'Hibernating': '#8B0000',
            'Retained': 'green',
            'Churned': 'red'
        },
        title='Customer Retention by Segment'
    )
    fig.update_traces(
        textinfo='label+percent parent',
        insidetextorientation='radial'
    )
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # Data Table Display
    st.dataframe(
        segment_retention_table.style
        .background_gradient(subset=['Retention_Rate'], cmap='RdYlGn', vmin=50, vmax=100)
        .format({'Retention_Rate': '{:.1f}%'}),
        use_container_width=True
    )
    st.markdown("---")

    # --- DETAILED SEGMENT RECOMMENDATIONS ---
    st.header("Actionable Insights by Customer Segment")
    st.markdown("Click on each segment to understand their characteristics and recommended marketing actions.")
    
    segment_analysis = customer_df.groupby('Segment', observed=True).agg(
        Customer_Count=('Recency', 'count'),
        Recency_mean=('Recency', 'mean'),
        Frequency_mean=('Frequency', 'mean'),
        MonetaryValue_mean=('MonetaryValue', 'mean'),
        Churn_Rate=('Churn_Status', lambda x: (x == 1).mean() * 100)
    ).round(1).sort_values(by='MonetaryValue_mean', ascending=False)
    
    segment_analysis.rename(columns={
        'Customer_Count': 'Number of Customers',
        'Recency_mean': 'Avg. Days Since Last Purchase',
        'Frequency_mean': 'Avg. Number of Purchases',
        'MonetaryValue_mean': 'Avg. Total Spent ($)',
        'Churn_Rate': 'Churn Rate (%)'
    }, inplace=True)

    with st.expander("🥇 **Champions** - Your Best & Most Loyal Customers", expanded=True):
        st.dataframe(segment_analysis.loc[['Champions']], use_container_width=True)
        st.markdown("""
        **Characteristics:**  
        - Highest spending customers  
        - Frequent purchases  
        - Recently active  
        
        **Recommended Actions:**  
        - Reward with exclusive offers  
        - Seek testimonials and case studies  
        - Avoid discounts (they'll pay full price)  
        - Offer VIP customer experiences  
        """)

    with st.expander("👍 **Loyal Customers** - Your Consistent Supporters"):
        st.dataframe(segment_analysis.loc[['Loyal Customers']], use_container_width=True)
        st.markdown("""
        **Characteristics:**  
        - Good purchase frequency  
        - Moderate spending  
        - Somewhat recently active  
        
        **Recommended Actions:**  
        - Upsell higher-value products  
        - Offer loyalty program memberships  
        - Keep them engaged with regular content  
        - Request product reviews  
        """)

    with st.expander("👀 **Needs Attention** - Customers on the Fence"):
        st.dataframe(segment_analysis.loc[['Needs Attention']], use_container_width=True)
        st.markdown("""
        **Characteristics:**  
        - Moderate spending but declining  
        - Haven't purchased recently  
        - Higher churn risk  
        
        **Recommended Actions:**  
        - Re-engage with targeted promotions  
        - Offer personalized discounts  
        - Send "we miss you" campaigns  
        - Survey to understand their needs  
        """)

    with st.expander("💤 **Hibernating** - Lapsed or Low-Value Customers"):
        st.dataframe(segment_analysis.loc[['Hibernating']], use_container_width=True)
        st.markdown("""
        **Characteristics:**  
        - Lowest spending  
        - Haven't purchased in a long time  
        - Highest churn rate  
        
        **Recommended Actions:**  
        - Launch a 'win-back' campaign  
        - Offer special reactivation discounts  
        - Focus only on those with high historical value  
        - Consider sunsetting unresponsive customers  
        """)
        
    st.markdown("---")
    
    # --- PERFORMANCE TRENDS & DEEP DIVES ---
    st.header("Performance Trends & Deep Dives")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Customer Lifecycle", "Sales & Products", "Geographical Performance","Churn Drivers & Demographics","Churn Drivers"])
    
    with tab1:
        st.subheader("Monthly Customer Acquisition vs. Churn")
        
        # Prepare data for trend charts
        new_customers_monthly = customer_df.groupby('Acquisition_Month')['Customer ID'].nunique().reset_index().rename(columns={'Customer ID': 'New Customers', 'Acquisition_Month': 'Month'})
        
        churned_customers_monthly = customer_df[customer_df['Churn_Status'] == 1].copy()
        churned_customers_monthly['Churn_Month'] = churned_customers_monthly['Churn_Date'].dt.to_period('M').dt.to_timestamp()
        churned_customers_monthly = churned_customers_monthly.groupby('Churn_Month')['Customer ID'].nunique().reset_index().rename(columns={'Customer ID': 'Churned Customers', 'Churn_Month': 'Month'})
        
        lifecycle_df = pd.merge(new_customers_monthly, churned_customers_monthly, on='Month', how='outer').fillna(0)
        lifecycle_df['Net Growth'] = lifecycle_df['New Customers'] - lifecycle_df['Churned Customers']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lifecycle_df['Month'],
            y=lifecycle_df['New Customers'],
            name='New Customers',
            marker_color='#2ca02c'
        ))
        fig.add_trace(go.Bar(
            x=lifecycle_df['Month'],
            y=-lifecycle_df['Churned Customers'],
            name='Churned Customers',
            marker_color='#d62728'
        ))
        fig.add_trace(go.Scatter(
            x=lifecycle_df['Month'],
            y=lifecycle_df['Net Growth'],
            name='Net Growth',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            barmode='relative',
            title='Customer Acquisition vs. Churn with Net Growth',
            xaxis_title='Month',
            yaxis_title='Number of Customers',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        avg_growth = lifecycle_df['Net Growth'].mean()
        last_growth = lifecycle_df.iloc[-1]['Net Growth']
        
        growth_col1, growth_col2 = st.columns(2)
        growth_col1.metric("Average Monthly Net Growth", f"{avg_growth:.1f}")
        growth_col2.metric("Latest Month Net Growth", last_growth,
                          delta=f"{(last_growth - avg_growth):.1f} vs average")
    
    with tab2:
        st.header("📊 Performance Trends & Deep Dives")
        
        # 1. CUSTOMER TRENDS SECTION
        st.subheader("🔄 Customer Acquisition & Churn Trends")
        
        @st.cache_data
        def prepare_customer_trends():
            new_customers = (customer_df.groupby('Acquisition_Month')['Customer ID']
                           .nunique()
                           .reset_index()
                           .rename(columns={'Customer ID': 'New Customers',
                                       'Acquisition_Month': 'Month'}))
            
            churned = (customer_df[customer_df['Churn_Status'] == 1]
                    .assign(Churn_Month=lambda x: x['Churn_Date'].dt.to_period('M').dt.to_timestamp())
                    .groupby('Churn_Month')['Customer ID']
                    .nunique()
                    .reset_index()
                    .rename(columns={'Customer ID': 'Churned Customers',
                                    'Churn_Month': 'Month'}))
            
            return new_customers, churned
        
        new_customers_monthly, churned_customers_monthly = prepare_customer_trends()
        
        trend_col1, trend_col2 = st.columns(2)
        with trend_col1:
            fig_acq = px.line(new_customers_monthly,
                           x='Month', y='New Customers',
                           title="📈 Monthly Acquisitions",
                           markers=True)
            st.plotly_chart(fig_acq, use_container_width=True)
        
        with trend_col2:
            fig_churn = px.line(churned_customers_monthly,
                           x='Month', y='Churned Customers',
                           title="📉 Monthly Churn",
                           markers=True,
                           color_discrete_sequence=['red'])
            st.plotly_chart(fig_churn, use_container_width=True)
        
        st.markdown("---")
        
        # 2. SALES PERFORMANCE SECTION
        st.subheader("💰 Sales Performance")
        
        @st.cache_data
        def prepare_sales_data():
            monthly_sales = (transaction_df.set_index('InvoiceDate')
                        .resample('ME')['TotalPrice']
                        .sum()
                        .reset_index()
                        .assign(Growth=lambda x: x['TotalPrice'].pct_change() * 100))
            
            top_products = (transaction_df.groupby('Description')['Quantity']
                        .sum()
                        .nlargest(10)
            )
            
            avg_rev = (transaction_df.set_index('InvoiceDate')
                    .resample('ME')
                    .agg(Monthly_Revenue=('TotalPrice', 'sum'),
                            Unique_Customers=('Customer ID', 'nunique'))
                    .reset_index()
                    .assign(Avg_Revenue_per_Customer=lambda x: x['Monthly_Revenue'] / x['Unique_Customers']))
            
            return monthly_sales, top_products, avg_rev
        
        monthly_sales, top_products, avg_rev = prepare_sales_data()
        
        # Revenue and growth chart
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Bar(
            x=monthly_sales['InvoiceDate'],
            y=monthly_sales['TotalPrice'],
            name='Revenue',
            marker_color='#636EFA'
        ))
        fig_revenue.add_trace(go.Scatter(
            x=monthly_sales['InvoiceDate'],
            y=monthly_sales['Growth'],
            name='Growth Rate',
            yaxis='y2',
            line=dict(color='#FFA15A', width=3)
        ))
        fig_revenue.update_layout(
            title='Monthly Revenue with Growth Rate',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            yaxis2=dict(title='Growth Rate (%)', overlaying='y', side='right'),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.1)
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Product and avg revenue charts
        sales_col1, sales_col2 = st.columns(2)
        with sales_col1:
            fig_products = px.bar(
                top_products.reset_index(),
                x='Quantity', y='Description',
                orientation='h',
                title="Top 10 Products by Volume",
                color='Quantity',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_products, use_container_width=True)
        
        with sales_col2:
            fig_avg_rev = px.line(
                avg_rev,
                x='InvoiceDate', y='Avg_Revenue_per_Customer',
                title="Average Revenue per Customer",
                markers=True
            )
            st.plotly_chart(fig_avg_rev, use_container_width=True)
        
        st.markdown("---")
        
        # 3. STRATEGIC CHURN ANALYSIS SECTION
        st.subheader("🔍 Strategic Churn Insights")
        
        # Value at Risk analysis
        st.markdown("#### Value at Risk by Churn Probability")
        
        risk_bins = np.arange(0, 1.1, 0.1)
        risk_labels = [f"{int(i*100)}-{int((i+0.1)*100)}%" for i in risk_bins[:-1]]
        
        value_at_risk_df = (customer_df.assign(
            Risk_Bracket=lambda x: pd.cut(x['Churn_Probability'],
                                        bins=risk_bins,
                                        labels=risk_labels))
                            .groupby('Risk_Bracket', observed=True)['MonetaryValue']
                            .sum()
                            .reset_index())
        
        fig_risk = px.bar(
            value_at_risk_df,
            x='Risk_Bracket', y='MonetaryValue',
            title="Total Customer Value by Churn Risk Level",
            labels={'MonetaryValue': 'Total Value ($)', 'Risk_Bracket': 'Churn Probability Range'},
            text_auto='.2s'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        st.caption("Identifies revenue at risk from potential churn, helping prioritize retention efforts.")
    
    with tab3:
        st.subheader("Geographical Performance Analysis")
        
        # Country-level analysis
        country_stats = customer_df.groupby('Country').agg(
            Total_Customers=('Customer ID', 'nunique'),
            Churned_Customers=('Churn_Status', 'sum'),
            Total_Revenue=('MonetaryValue', 'sum')
        ).reset_index()
        
        country_stats['Churn_Rate'] = (country_stats['Churned_Customers'] / country_stats['Total_Customers']) * 100
        country_stats['Revenue_per_Customer'] = country_stats['Total_Revenue'] / country_stats['Total_Customers']
        
        # Top countries by revenue
        colX, colY = st.columns(2)
        
        with colX:
            st.markdown("**Top Countries by Revenue**")
            top_countries = country_stats.sort_values('Total_Revenue', ascending=False).head(10)
            fig_top_countries = px.bar(
                top_countries,
                x='Country',
                y='Total_Revenue',
                color='Total_Revenue',
                color_continuous_scale='Viridis',
                text_auto='.2s'
            )
            fig_top_countries.update_layout(
                xaxis_title="Country",
                yaxis_title="Total Revenue ($)",
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_top_countries, use_container_width=True)
        
        with colY:
            st.markdown("**Revenue per Customer by Country**")
            fig_rev_per_cust = px.bar(
                country_stats.sort_values('Revenue_per_Customer', ascending=False).head(10),
                x='Country',
                y='Revenue_per_Customer',
                color='Revenue_per_Customer',
                color_continuous_scale='Plasma',
                text_auto='.2s'
            )
            fig_rev_per_cust.update_layout(
                xaxis_title="Country",
                yaxis_title="Revenue per Customer ($)",
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_rev_per_cust, use_container_width=True)
        
        # World Heatmap
        st.markdown("**Worldwide Customer Churn Rate Heatmap**")
        fig_heatmap = px.choropleth(
            country_stats,
            locations="Country",
            locationmode="country names",
            color="Churn_Rate",
            hover_name="Country",
            hover_data=["Total_Customers", "Total_Revenue"],
            color_continuous_scale=px.colors.sequential.YlOrRd,
            range_color=(0, 100)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


    # =============================================================================
# --- NEW TAB: Churn Drivers & Demographics ---
# This code creates a new section for deeper churn analysis.
# =============================================================================
with tab4:
    st.header("🤔 Churn Drivers & Demographic Insights")
    st.info("This section breaks down churn by key customer attributes to understand who is most likely to leave.")

    col1, col2 = st.columns(2)

    # --- Insight 1: Churn Rate by Segment ---
    with col1:
        st.subheader("Churn Rate by Customer Segment")
        
        # Calculate churn rate per segment
        churn_by_segment = customer_df.groupby('Segment', observed=True)['Churn_Status'].value_counts(normalize=True).unstack().fillna(0)
        churn_by_segment['Churn_Rate_%'] = churn_by_segment[1] * 100 # '1' represents Churned

        # Create the bar chart
        fig_churn_bar = px.bar(
            churn_by_segment, 
            y='Churn_Rate_%', 
            x=churn_by_segment.index, 
            text_auto='.1f',
            title="Percentage of Churned Customers in Each Segment",
            color=churn_by_segment.index,
            color_discrete_map={
                'Champions': 'gold', 
                'Loyal Customers': 'royalblue', 
                'Needs Attention': 'lightslategray', 
                'Hibernating': 'darkred'
            }
        )
        fig_churn_bar.update_layout(xaxis_title="Segment", yaxis_title="Churn Rate (%)", showlegend=False)
        st.plotly_chart(fig_churn_bar, use_container_width=True)

    # --- Insight 2: Churn by Tenure (Improved Visualization) ---
    with col2:
        st.subheader("Average Tenure by Customer Segment")
        
        # Calculate tenure (time between first and last purchase)
        customer_df['Tenure_Days'] = (pd.to_datetime(customer_df['Last_Purchase_Date']) - pd.to_datetime(customer_df['Acquisition_Month'])).dt.days
        
        # Calculate average tenure per segment
        avg_tenure_by_segment = customer_df.groupby('Segment', observed=True)['Tenure_Days'].mean().reset_index()

        # Create a more intuitive bar chart
        fig_tenure = px.bar(
            avg_tenure_by_segment,
            x='Segment',
            y='Tenure_Days',
            color='Segment',
            title="Average Customer Tenure by Segment",
            labels={'Tenure_Days': 'Average Tenure (Days)', 'Segment': 'Customer Segment'},
            text='Tenure_Days'
        )
        fig_tenure.update_traces(texttemplate='%{text:.0f} days', textposition='outside')
        fig_tenure.update_layout(showlegend=False)
        st.plotly_chart(fig_tenure, use_container_width=True)

    st.markdown("---")

    # --- Insight 3: Demographics Table ---
    st.subheader("Demographics: Churn Breakdown by Country")
    country_churn_table = customer_df.groupby('Country').agg(
        Total_Customers=('Customer ID', 'nunique'),
        Churned_Accounts=('Churn_Status', lambda x: (x == 1).sum())
    ).reset_index()
    country_churn_table['Active_Accounts'] = country_churn_table['Total_Customers'] - country_churn_table['Churned_Accounts']
    country_churn_table['Accounts_Churned_%'] = (country_churn_table['Churned_Accounts'] / country_churn_table['Total_Customers']) * 100
    
    st.dataframe(
        country_churn_table.sort_values(by='Accounts_Churned_%', ascending=False).head(15),
        column_config={
            "Country": "Country", 
            "Active_Accounts": "Active Accounts", 
            "Churned_Accounts": "Accounts Ceased",
            "Accounts_Churned_%": st.column_config.ProgressColumn(
                "Accounts Churned (%)", format="%.1f%%", min_value=0, max_value=100
            )
        },
        hide_index=True, 
        use_container_width=True
    )
with tab5:
        # =============================================================================
    # --- TAB 5: Churn Drivers ---
    # This code block creates the entire content for the 'Churn Drivers' tab.
    # It assumes the 'customer_df' and 'transaction_df' DataFrames are already loaded.
    # =============================================================================


    # You would place this code inside the 'with tab5:' block in your main app script

    st.header("🔬 Strategic Churn Driver Analysis")
    st.subheader("Does purchasing a wider variety of products increase loyalty?")

    # --- 1. Data Preparation ---
    # Calculate the number of unique products each customer bought from the transaction data
    product_counts = transaction_df.groupby('Customer ID')['Description'].nunique().reset_index().rename(columns={'Description': 'Unique_Product_Count'})

    # Merge this new information into our main customer dataframe
    analysis_df = pd.merge(customer_df, product_counts, on='Customer ID', how='left')
    # Fill any missing product counts with 0 (for customers who may not have transactions)
    analysis_df['Unique_Product_Count'] = analysis_df['Unique_Product_Count'].fillna(0)

    # Group by the number of unique products to get customer counts and churn rates for each group
    churn_by_products_df = analysis_df.groupby('Unique_Product_Count').agg(
        Number_of_Customers=('Customer ID', 'nunique'),
        Churn_Rate=('Churn_Status', lambda x: (x == 1).mean() * 100) # Calculate churn rate as a percentage
    ).reset_index()

    # For a clearer chart, we'll focus on customers who bought up to 20 unique products
    churn_by_products_df = churn_by_products_df[churn_by_products_df['Unique_Product_Count'] <= 20]
    # We also want to ensure we only look at groups with a meaningful number of customers
    churn_by_products_df = churn_by_products_df[churn_by_products_df['Number_of_Customers'] > 5]


    # --- 2. Create the Combination Chart ---
    # This type of chart with two different y-axes requires using make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the Bar Chart for the Number of Customers (left y-axis)
    fig.add_trace(
        go.Bar(
            x=churn_by_products_df['Unique_Product_Count'], 
            y=churn_by_products_df['Number_of_Customers'], 
            name='Number of Customers',
            marker_color='royalblue',
            opacity=0.7
        ),
        secondary_y=False,
    )

    # Add the Line Chart for the Churn Rate (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=churn_by_products_df['Unique_Product_Count'], 
            y=churn_by_products_df['Churn_Rate'], 
            name='Churn Rate',
            mode='lines+markers',
            marker_color='darkred'
        ),
        secondary_y=True,
    )

    # --- 3. Style the Chart for a Professional Look ---
    fig.update_layout(
        title_text="<b>Churn Analysis by Product Variety</b>",
        xaxis_title="Number of Unique Products Purchased",
        plot_bgcolor='rgba(245, 245, 245, 1)', # Light grey background for a modern feel
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Set y-axes titles and formatting
    fig.update_yaxes(title_text="<b>Number of Customers</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Churn Rate (%)</b>", secondary_y=True)


    # --- 4. Display the Chart and Insights ---
    st.plotly_chart(fig, use_container_width=True)

    st.caption("""
    **How to Read This Chart:** The blue bars show how many customers purchased a certain number of unique products (e.g., exactly 1, 2, 3, etc.). The red line tracks the churn rate for each of those groups.

    **Key Insight:** This chart reveals a powerful trend: **the more varied a customer's purchases are, the less likely they are to churn.** The churn rate is highest for customers who only buy one unique product and drops significantly as customers engage with a wider range of your inventory. This suggests that cross-selling and encouraging product discovery are powerful customer retention strategies.
    """)




# Add custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        border-left: 3px solid #8A2BE2;
        padding-left: 1rem;
    }
    .stExpander {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)
            