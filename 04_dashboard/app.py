

# #  Deepseek
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path 
import numpy as np
import plotly.graph_objects as go

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
        snapshot_date = pd.to_datetime('2010-12-10') # Based on dataset's last transaction
        customer_df['Last_Purchase_Date'] = snapshot_date - pd.to_timedelta(customer_df['Recency'], unit='d')
        customer_df['Churn_Date'] = customer_df['Last_Purchase_Date'] + pd.to_timedelta(180, unit='d')
        customer_df['Acquisition_Month'] = customer_df.groupby('Customer ID')['Last_Purchase_Date'].transform('min').dt.to_period('M').dt.to_timestamp()
        
        return customer_df, transaction_df
        
    except FileNotFoundError:
        st.error("FATAL ERROR: Could not find necessary data files in '01_data/processed/'.")
        st.error("Please ensure both 'final_dashboard_data.csv' and 'cleaned_retail_data.csv' exist.")
        return None, None

customer_df, transaction_df = load_and_prepare_data()

# --- 3. MAIN APPLICATION ---
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
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=churn_rate, 
            title={'text': "Current Churn Rate (%)"},
            gauge={
                'axis': {'range': [None, 50]}, 
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"}, 
                    {'range': [20, 35], 'color': "yellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 25  # Benchmark value
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        retention_rate = 100 - churn_rate
        fig_gauge_retention = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=retention_rate, 
            title={'text': "Retention Rate (%)"},
            gauge={
                'axis': {'range': [50, 100]}, 
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [50, 80], 'color': 'lightgray'}, 
                    {'range': [80, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 75  # Benchmark value
                }
            }
        ))
        fig_gauge_retention.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_gauge_retention, use_container_width=True)

    with col3:
        segment_counts = customer_df['Segment'].value_counts()
        fig_donut = px.pie(
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
        fig_donut.update_layout(
            height=300, 
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # --- STRATEGIC OVERVIEW: CHURN RISK VS. SEGMENT VALUE ---
    st.subheader("Which Segments are Most Valuable to Save?")

    # 1. Calculate the necessary data for the chart by grouping by segment
    # We only consider 'Active' customers for this strategic view
    risk_analysis_df = customer_df[customer_df['Churn_Status'] == 0].groupby('Segment', observed=True).agg(
        Avg_Churn_Risk=('Churn_Probability', 'mean'),
        Total_Spendings=('MonetaryValue', 'sum'),
        Customer_Count=('Customer ID', 'nunique')
    ).reset_index()

    # Convert average churn risk to a percentage for readability
    risk_analysis_df['Avg_Churn_Risk'] = risk_analysis_df['Avg_Churn_Risk'] * 100

    # 2. Create the scatter plot (bubble chart) with a more polished look
    fig_risk_scatter = px.scatter(
        risk_analysis_df,
        x='Total_Spendings',
        y='Avg_Churn_Risk',
        size='Customer_Count',
        color='Segment',
        hover_name='Segment',
        hover_data={ # Add more details to the hover tooltip
            'Segment': False, # Hide the default segment name in hover
            'Total_Spendings': ':$,.0f',
            'Avg_Churn_Risk': ':.1f%',
            'Customer_Count': ':,'
        },
        size_max=70, # Increase max size for better visual impact
        title='Segment Value vs. Average Churn Risk',
        labels={
            "Total_Spendings": "Total Spendings of Segment ($)",
            "Avg_Churn_Risk": "Average Churn Risk (%)",
            "Customer_Count": "Number of Customers"
        },
        # Use the same color map for consistency
        color_discrete_map={
            'Champions': 'gold',
            'Loyal Customers': 'royalblue',
            'Needs Attention': 'lightslategray',
            'Hibernating': 'darkred'
        }
    )

    # 3. Enhance the chart's appearance for a professional look
    # Remove the text from the bubbles for a cleaner chart, relying on the legend instead
    fig_risk_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

    # Add a horizontal line representing the average churn risk for all active customers
    avg_risk_line = customer_df[customer_df['Churn_Status'] == 0]['Churn_Probability'].mean() * 100
    fig_risk_scatter.add_hline(
        y=avg_risk_line, 
        line_dash="dot",
        annotation_text=f"Avg. Active Customer Risk ({avg_risk_line:.1f}%)", 
        annotation_position="bottom right"
    )

    # Add a vertical line for the average segment value to create quadrants
    avg_value_line = risk_analysis_df['Total_Spendings'].mean()
    fig_risk_scatter.add_vline(
        x=avg_value_line,
        line_dash="dot"
    )

    # Update layout for a cleaner, professional look
    fig_risk_scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(x=0.5),
        legend_title_text='Segment'
    )


    # 4. Display the chart and an explanation
    st.plotly_chart(fig_risk_scatter, use_container_width=True)
    st.caption("This chart helps prioritize retention efforts. Segments in the top-right quadrant (High Value, High Risk) are the most critical to save.")

    st.markdown("---")



        # --- ENHANCED SEGMENTED RETENTION ANALYSIS ---
    st.subheader("Retention by Customer Segment")

    # --- Data Preparation for this Chart ---
    # Create a copy to avoid changing the main dataframe
    chart_df = customer_df.copy()
    # Create a more descriptive label for the chart
    chart_df['Status'] = np.where(chart_df['Churn_Status'] == 0, 'Retained', 'Churned')

    # Create the data table for display
    segment_retention_table = chart_df.groupby('Segment', observed=True)['Status'].value_counts().unstack().fillna(0)
    segment_retention_table['Retention_Rate'] = (segment_retention_table['Retained'] / (segment_retention_table['Retained'] + segment_retention_table['Churned'])) * 100

    # --- Sunburst Chart with Clearer Labels ---
    # This chart now uses the descriptive 'Status' column in its path
    fig = px.sunburst(
        chart_df,
        path=['Segment', 'Status'], # Using the new 'Status' column
        color='Segment',
        color_discrete_map={
            'Champions': '#FFD700',
            'Loyal Customers': '#4169E1',
            'Needs Attention': '#778899',
            'Hibernating': '#8B0000',
            'Retained': 'green', # Optional: color for the outer ring
            'Churned': 'red'    # Optional: color for the outer ring
        },
        title='Customer Retention by Segment'
    )

    # Update the text on the chart to be clear
    fig.update_traces(
        textinfo='label+percent parent',
        insidetextorientation='radial'
    )
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- Data Table Display ---
    st.dataframe(
        segment_retention_table.style
        .background_gradient(subset=['Retention_Rate'], cmap='RdYlGn', vmin=50, vmax=100) # Set color range
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
    tab1, tab2, tab3 = st.tabs(["Customer Lifecycle", "Sales & Products", "Geographical Performance"])
    
    with tab1:
        st.subheader("Monthly Customer Acquisition vs. Churn")
        
        # Prepare data for trend charts
        new_customers_monthly = customer_df.groupby('Acquisition_Month')['Customer ID'].nunique().reset_index().rename(columns={'Customer ID': 'New Customers', 'Acquisition_Month': 'Month'})
        
        churned_customers_monthly = customer_df[customer_df['Churn_Status'] == 1].copy()
        churned_customers_monthly['Churn_Month'] = churned_customers_monthly['Churn_Date'].dt.to_period('M').dt.to_timestamp()
        churned_customers_monthly = churned_customers_monthly.groupby('Churn_Month')['Customer ID'].nunique().reset_index().rename(columns={'Customer ID': 'Churned Customers', 'Churn_Month': 'Month'})
        
        lifecycle_df = pd.merge(new_customers_monthly, churned_customers_monthly, on='Month', how='outer').fillna(0)
        
        # Net growth calculation
        lifecycle_df['Net Growth'] = lifecycle_df['New Customers'] - lifecycle_df['Churned Customers']
        
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Bar(
            x=lifecycle_df['Month'],
            y=lifecycle_df['New Customers'],
            name='New Customers',
            marker_color='#2ca02c'
        ))
        
        fig.add_trace(go.Bar(
            x=lifecycle_df['Month'],
            y=-lifecycle_df['Churned Customers'],  # Negative for visual distinction
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
        
        
            
    # with tab2:
    #     st.header("Performance Trends & Deep Dives")

    #     # Prepare data for trend charts
    #     new_customers_monthly = customer_df.groupby('Acquisition_Month')['Customer ID'].nunique().reset_index().rename(columns={'Customer ID': 'New Customers', 'Acquisition_Month': 'Month'})
    #     churned_customers_monthly = customer_df[customer_df['Churn_Status'] == 1].copy()
    #     churned_customers_monthly['Churn_Month'] = churned_customers_monthly['Churn_Date'].dt.to_period('M').dt.to_timestamp()
    #     churned_customers_monthly = churned_customers_monthly.groupby('Churn_Month')['Customer ID'].nunique().reset_index().rename(columns={'Customer ID': 'Churned Customers', 'Churn_Month': 'Month'})
        
    #     # New separate charts for Acquisition and Ceased trends
    #     st.subheader("Monthly Acquisitions vs. Ceased Customers Trend")
    #     colA, colB = st.columns(2)
    #     with colA:
    #         fig_acq_trend = px.line(new_customers_monthly, x='Month', y='New Customers', title="Acquisitions Trend", markers=True)
    #         fig_acq_trend.update_layout(yaxis_title="Number of New Customers")
    #         st.plotly_chart(fig_acq_trend, use_container_width=True)
    #     with colB:
    #         fig_ceased_trend = px.line(churned_customers_monthly, x='Month', y='Churned Customers', title="Ceased (Churn) Trend", markers=True)
    #         fig_ceased_trend.update_traces(line_color='darkred')
    #         fig_ceased_trend.update_layout(yaxis_title="Number of Churned Customers")
    #         st.plotly_chart(fig_ceased_trend, use_container_width=True)
        
    #     st.markdown("---")
        
    #     st.subheader("Sales Performance & Product Analysis")
    #     colC, colD = st.columns(2)
    #     with colC:
    #         monthly_revenue = transaction_df.set_index('InvoiceDate').resample('ME')['TotalPrice'].sum().reset_index()
    #         fig_revenue = px.bar(monthly_revenue, x='InvoiceDate', y='TotalPrice', text_auto='.2s', title="Monthly Revenue")
    #         st.plotly_chart(fig_revenue, use_container_width=True)
    #     with colD:
    #         top_products = transaction_df.groupby('Description')['Quantity'].sum().nlargest(10).sort_values(ascending=True)
    #         fig_products = px.bar(top_products, x='Quantity', y=top_products.index, orientation='h', text='Quantity', title="Top 10 Products by Quantity")
    #         st.plotly_chart(fig_products, use_container_width=True)
    #     st.markdown("---")

        

    #             # --- NEW SECTION: Strategic Churn Analysis ---
    # # This block creates three new strategic charts. You can place this entire section
    # # in a new tab or at the end of your main dashboard layout.

    # st.header("📈 Strategic Churn & Value Analysis")

    # # --- Insight 1: Value at Risk Histogram ---
    # st.subheader("Value at Risk: Spendings by Churn Risk Level")

    # # Create bins for the churn probability to group customers
    # risk_bins = np.arange(0, 1.1, 0.1) # Bins from 0-10%, 10-20%, etc.
    # risk_labels = [f"{int(i*100)}-{int((i+0.1)*100)}%" for i in risk_bins[:-1]]

    # # Create a temporary dataframe for this chart
    # value_at_risk_df = customer_df.copy()
    # value_at_risk_df['Risk_Bracket'] = pd.cut(
    #     value_at_risk_df['Churn_Probability'], 
    #     bins=risk_bins, 
    #     labels=risk_labels, 
    #     right=False
    # )

    # # Group by the risk brackets and sum the total spendings (MonetaryValue)
    # value_at_risk_analysis = value_at_risk_df.groupby('Risk_Bracket', observed=True)['MonetaryValue'].sum().reset_index()

    # # Create the bar chart
    # fig_value_risk = px.bar(
    #     value_at_risk_analysis, 
    #     x='Risk_Bracket', 
    #     y='MonetaryValue',
    #     title="Total Spendings of Customers by Churn Risk Bracket",
    #     labels={'Risk_Bracket': 'Churn Risk Bracket', 'MonetaryValue': 'Total Spendings ($)'},
    #     text_auto='.2s'
    # )
    # fig_value_risk.update_traces(marker_color='indigo')
    # st.plotly_chart(fig_value_risk, use_container_width=True)
    # st.caption("This chart shows the total value of customers grouped by their predicted churn risk. High bars in the middle-risk brackets represent significant revenue that could be saved with targeted retention campaigns.")

    # st.markdown("---")

    # # --- Insights 2 & 3 in a two-column layout ---
    # col1, col2 = st.columns(2)

    # with col1:
    #     # --- Insight 2: Churn Risk by Number of Products Purchased ---
    #     st.subheader("Churn Risk by Product Variety")
        
    #     # Calculate the number of unique products each customer bought
    #     product_counts = transaction_df.groupby('Customer ID')['Description'].nunique().reset_index().rename(columns={'Description': 'Unique_Product_Count'})
        
    #     # Merge this data into our main customer dataframe
    #     analysis_df = pd.merge(customer_df, product_counts, on='Customer ID', how='left')
        
    #     # --- FIX IS HERE: Fill NaN only on the specific column ---
    #     analysis_df['Unique_Product_Count'] = analysis_df['Unique_Product_Count'].fillna(0)
        
    #     # Create bins for the number of products
    #     product_bins = [0, 1, 2, 3, 5, 10, np.inf]
    #     product_labels = ['1', '2', '3', '4-5', '6-10', '11+']
    #     analysis_df['Product_Count_Bracket'] = pd.cut(analysis_df['Unique_Product_Count'], bins=product_bins, labels=product_labels)

    #     # Calculate average churn risk for each product count bracket
    #     churn_by_products = analysis_df.groupby('Product_Count_Bracket', observed=True)['Churn_Probability'].mean().reset_index()
    #     churn_by_products['Churn_Probability'] = churn_by_products['Churn_Probability'] * 100 # Convert to percentage

    #     # Create the bar chart
    #     fig_churn_by_products = px.bar(
    #         churn_by_products,
    #         x='Product_Count_Bracket',
    #         y='Churn_Probability',
    #         title="Average Churn Risk by No. of Unique Products Purchased",
    #         labels={'Product_Count_Bracket': 'Number of Unique Products', 'Churn_Probability': 'Average Churn Risk (%)'},
    #         text=churn_by_products['Churn_Probability'].apply(lambda x: f'{x:.1f}%')
    #     )
    #     st.plotly_chart(fig_churn_by_products, use_container_width=True)
    #     st.caption("This chart suggests that customers who purchase a wider variety of products tend to have a lower risk of churning.")

    # with col2:
    #     # --- Insight 3: Strategic Retention Quadrant Chart ---
    #     st.subheader("Which Segments to Prioritize?")

    #     # Calculate data for the quadrant chart
    #     risk_analysis_df = customer_df[customer_df['Churn_Status'] == 0].groupby('Segment', observed=True).agg(
    #         Avg_Churn_Risk=('Churn_Probability', 'mean'),
    #         Total_Spendings=('MonetaryValue', 'sum'),
    #         Customer_Count=('Customer ID', 'nunique')
    #     ).reset_index()
    #     risk_analysis_df['Avg_Churn_Risk'] = risk_analysis_df['Avg_Churn_Risk'] * 100

    #     # Create the bubble chart
    #     fig_risk_scatter = px.scatter(
    #         risk_analysis_df,
    #         x='Total_Spendings',
    #         y='Avg_Churn_Risk',
    #         size='Customer_Count',
    #         color='Segment',
    #         hover_name='Segment',
    #         size_max=70,
    #         title='Segment Value vs. Average Churn Risk',
    #         labels={"Total_Spendings": "Total Segment Value ($)", "Avg_Churn_Risk": "Average Churn Risk (%)"},
    #         color_discrete_map={'Champions': 'gold', 'Loyal Customers': 'royalblue', 'Needs Attention': 'lightslategray', 'Hibernating': 'darkred'}
    #     )
        
    #     # Add median lines to create the quadrants
    #     median_risk = risk_analysis_df['Avg_Churn_Risk'].median()
    #     median_value = risk_analysis_df['Total_Spendings'].median()
    #     fig_risk_scatter.add_hline(y=median_risk, line_dash="dot", annotation_text="Median Risk")
    #     fig_risk_scatter.add_vline(x=median_value, line_dash="dot", annotation_text="Median Value")
        
    #     st.plotly_chart(fig_risk_scatter, use_container_width=True)
    #     st.caption("Segments in the top-right quadrant (High Value, High Risk) are the most critical to save.")













    #     st.subheader("Sales Performance & Product Analysis")
        
    #     # Monthly Revenue Analysis
    #     monthly_revenue = transaction_df.set_index('InvoiceDate').resample('ME')['TotalPrice'].sum().reset_index()
    #     monthly_revenue['Growth'] = monthly_revenue['TotalPrice'].pct_change() * 100
        
    #     fig_revenue = go.Figure()
        
    #     fig_revenue.add_trace(go.Bar(
    #         x=monthly_revenue['InvoiceDate'],
    #         y=monthly_revenue['TotalPrice'],
    #         name='Revenue',
    #         marker_color='#636EFA'
    #     ))
        
    #     fig_revenue.add_trace(go.Scatter(
    #         x=monthly_revenue['InvoiceDate'],
    #         y=monthly_revenue['Growth'],
    #         name='Growth Rate',
    #         yaxis='y2',
    #         line=dict(color='#FFA15A', width=3),
    #         marker=dict(size=8)
    #     ))
        
    #     fig_revenue.update_layout(
    #         title='Monthly Revenue with Growth Rate',
    #         xaxis_title='Month',
    #         yaxis_title='Revenue ($)',
    #         yaxis2=dict(
    #             title='Growth Rate (%)',
    #             overlaying='y',
    #             side='right',
    #             range=[-50, 50]  # Fixed range for better comparison
    #         ),
    #         hovermode='x unified',
    #         legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
    #     )
        
    #     st.plotly_chart(fig_revenue, use_container_width=True)
        
    #     # Product Analysis
    #     colA, colB = st.columns(2)
        
    #     with colA:
    #         st.subheader("Top 10 Bestselling Products")
    #         top_products = transaction_df.groupby('Description')['Quantity'].sum().nlargest(10).sort_values(ascending=True)
    #         fig_products = px.bar(
    #             top_products, 
    #             x='Quantity', 
    #             y=top_products.index, 
    #             orientation='h', 
    #             text='Quantity',
    #             color=top_products.values,
    #             color_continuous_scale='Viridis'
    #         )
    #         fig_products.update_layout(
    #             xaxis_title="Total Quantity Sold", 
    #             yaxis_title="",
    #             showlegend=False,
    #             coloraxis_showscale=False
    #         )
    #         st.plotly_chart(fig_products, use_container_width=True)
        
    #     with colB:
    #         st.subheader("Avg. Revenue per Customer")
    #         monthly_avg_rev_df = transaction_df.set_index('InvoiceDate').resample('ME').agg(
    #             Monthly_Revenue=('TotalPrice', 'sum'),
    #             Unique_Customers=('Customer ID', 'nunique')
    #         ).reset_index()
    #         monthly_avg_rev_df['Avg_Revenue_per_Customer'] = monthly_avg_rev_df['Monthly_Revenue'] / monthly_avg_rev_df['Unique_Customers']
            
    #         fig_avg_rev = px.line(
    #             monthly_avg_rev_df, 
    #             x='InvoiceDate', 
    #             y='Avg_Revenue_per_Customer', 
    #             markers=True, 
    #             title="Monthly Average Revenue per Customer",
    #             line_shape='spline'
    #         )
    #         fig_avg_rev.update_traces(
    #             line_color='#8A2BE2',
    #             line_width=3,
    #             marker=dict(size=8, color='#8A2BE2')
    #         )
    #         fig_avg_rev.update_layout(
    #             xaxis_title="Month", 
    #             yaxis_title="Average Revenue ($)",
    #             hovermode='x unified'
    #         )
    #         st.plotly_chart(fig_avg_rev, use_container_width=True)



    with tab2:
        st.header("📊 Performance Trends & Deep Dives")
        
        # ===========================================
        # 1. CUSTOMER TRENDS SECTION
        # ===========================================
        st.subheader("🔄 Customer Acquisition & Churn Trends")
        
        # Prepare customer trend data
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
        
        # Display in columns
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
        
        # ===========================================
        # 2. SALES PERFORMANCE SECTION
        # ===========================================
        st.subheader("💰 Sales Performance")
        
        # Prepare sales data
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
                        .sort_values(ascending=True))
            
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
        
        # ===========================================
        # 3. STRATEGIC CHURN ANALYSIS SECTION
        # ===========================================
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
        
        # Dual-column insights
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("#### Churn Risk by Product Variety")
            
            product_analysis = (transaction_df.groupby('Customer ID')['Description']
                            .nunique()
                            .reset_index(name='Unique_Products')
                            .merge(customer_df, on='Customer ID')
                            .assign(Product_Count_Bracket=lambda x: pd.cut(
                                x['Unique_Products'].fillna(0),
                                bins=[0, 1, 2, 3, 5, 10, np.inf],
                                labels=['1', '2', '3', '4-5', '6-10', '11+']))
                            .groupby('Product_Count_Bracket', observed=True)['Churn_Probability']
                            .mean()
                            .mul(100)
                            .reset_index())
            
            fig_products = px.bar(
                product_analysis,
                x='Product_Count_Bracket', y='Churn_Probability',
                title="Average Churn Risk by Products Purchased",
                labels={'Churn_Probability': 'Churn Risk (%)', 
                    'Product_Count_Bracket': 'Number of Unique Products'},
                text=[f"{x:.1f}%" for x in product_analysis['Churn_Probability']]
            )
            st.plotly_chart(fig_products, use_container_width=True)
        
        with insight_col2:
            st.markdown("#### Segment Prioritization")
            
            segment_risk = (customer_df[customer_df['Churn_Status'] == 0]
                        .groupby('Segment', observed=True)
                        .agg(Avg_Churn_Risk=('Churn_Probability', 'mean'),
                            Total_Value=('MonetaryValue', 'sum'),
                            Customer_Count=('Customer ID', 'nunique'))
                        .reset_index()
                        .assign(Avg_Churn_Risk=lambda x: x['Avg_Churn_Risk'] * 100))
            
            fig_segment = px.scatter(
                segment_risk,
                x='Total_Value', y='Avg_Churn_Risk',
                size='Customer_Count',
                color='Segment',
                title="Segment Value vs. Churn Risk",
                labels={'Total_Value': 'Total Segment Value ($)',
                    'Avg_Churn_Risk': 'Average Churn Risk (%)'},
                color_discrete_map={
                    'Champions': 'gold',
                    'Loyal Customers': 'royalblue',
                    'Needs Attention': 'lightslategray',
                    'Hibernating': 'darkred'
                }
            )
            
            # Add median lines
            fig_segment.add_hline(y=segment_risk['Avg_Churn_Risk'].median(), 
                                line_dash="dot", 
                                annotation_text="Median Risk")
            fig_segment.add_vline(x=segment_risk['Total_Value'].median(), 
                                line_dash="dot", 
                                annotation_text="Median Value")
            
            st.plotly_chart(fig_segment, use_container_width=True)



    

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
            
    st.markdown("---")
    
    # --- PREDICTIVE INSIGHTS & STRATEGIC TOOLS ---

        # --- PREDICTIVE INSIGHTS & STRATEGIC TOOLS ---
    st.header("🔮 Predictive Insights & Strategic Tools")

    # Add this new subsection for churned customers export
    st.subheader("Churned Customer Data Export")
    st.markdown("Download complete list of churned customers for analysis:")

    # Filter for churned customers
    churned_customers_df = customer_df[customer_df['Churn_Status'] == 1]

    # Show summary stats
    st.write(f"**{len(churned_customers_df)} churned customers** identified")

    # Create download button
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(churned_customers_df)

    st.download_button(
        label="📥 Download All Churned Customers",
        data=csv,
        file_name='churned_customers_full_list.csv',
        mime='text/csv',
        disabled=len(churned_customers_df) == 0,
        help="Download complete list of churned customers with all attributes"
    )

    # Optional: Add filtered export options
    st.markdown("**Export with Filters**")

    col1, col2 = st.columns(2)
    with col1:
        min_value = st.number_input(
            "Minimum Lifetime Value ($)",
            min_value=0,
            max_value=int(churned_customers_df['MonetaryValue'].max()),
            value=0,
            key="churned_min_value"
        )

    with col2:
        selected_segment = st.selectbox(
            "Segment Filter",
            options=["All"] + churned_customers_df['Segment'].unique().tolist(),
            key="churned_segment_filter"
        )

    # Apply filters
    filtered_churned = churned_customers_df.copy()
    if min_value > 0:
        filtered_churned = filtered_churned[filtered_churned['MonetaryValue'] >= min_value]
    if selected_segment != "All":
        filtered_churned = filtered_churned[filtered_churned['Segment'] == selected_segment]

    st.write(f"**{len(filtered_churned)} customers** match your criteria")

    csv_filtered = convert_df_to_csv(filtered_churned)
    st.download_button(
        label="📥 Download Filtered Churned Customers",
        data=csv_filtered,
        file_name=f'churned_customers_filtered_{selected_segment}_{min_value}.csv',
        mime='text/csv',
        disabled=len(filtered_churned) == 0
    )



    st.header("🔮 Predictive Insights & Strategic Tools")

    st.subheader("Value at Risk: Spendings by Churn Risk Level")
    risk_bins = np.arange(0, 1.1, 0.1)
    risk_labels = [f"{int(i*100)}-{int((i+0.1)*100)}%" for i in risk_bins[:-1]]
    value_at_risk_df = customer_df.copy()
    value_at_risk_df['Risk_Bracket'] = pd.cut(
        customer_df['Churn_Probability'], 
        bins=risk_bins, 
        labels=risk_labels, 
        right=False
    )
    
    value_at_risk_analysis = value_at_risk_df.groupby('Risk_Bracket', observed=True).agg(
        Total_Customers=('Customer ID', 'nunique'),
        Total_Spendings=('MonetaryValue', 'sum')
    ).reset_index()
    
    fig_value_risk = px.bar(
        value_at_risk_analysis, 
        x='Risk_Bracket', 
        y='Total_Spendings',
        color='Total_Spendings',
        color_continuous_scale='Magma',
        title="Total Spendings of Customers by Churn Risk Bracket",
        labels={'Risk_Bracket': 'Churn Risk Bracket', 'Total_Spendings': 'Total Spendings ($)'}, 
        text_auto='.2s'
    )
    fig_value_risk.update_layout(
        xaxis_title="Churn Probability Range",
        yaxis_title="Total Spendings at Risk ($)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_value_risk, use_container_width=True)
    
    st.markdown("---")

    colC, colD = st.columns(2)
    with colC:
        st.subheader("Priority List: High Churn Risk Individuals")
        st.markdown("Active customers with highest churn probability:")
        
        high_risk_predictions_df = customer_df[
            (customer_df['Churn_Status'] == 0) & 
            (customer_df['Churn_Probability'] > 0.7)
        ].sort_values(by='Churn_Probability', ascending=False)
        
        if not high_risk_predictions_df.empty:
            st.dataframe(
                high_risk_predictions_df[['Customer ID', 'Segment', 'Country', 'Churn_Probability', 'MonetaryValue']].head(10), 
                column_config={
                    "Churn_Probability": st.column_config.ProgressColumn(
                        "Churn Risk", 
                        format="%.0f%%", 
                        min_value=0, 
                        max_value=100
                    ),
                    "MonetaryValue": st.column_config.NumberColumn(
                        "Lifetime Value",
                        format="$%.2f"
                    )
                }, 
                use_container_width=True
            )
        else:
            st.success("No high-risk customers identified - great job on retention!")
            
        st.metric(
            "Total High-Risk Customers", 
            len(high_risk_predictions_df),
            delta=f"{len(high_risk_predictions_df)/len(customer_df)*100:.1f}% of total"
        )
        
    with colD:
        st.subheader("Customer Segment Exporter")
        st.markdown("Download customer lists for targeted marketing campaigns:")
        
        segment_list = sorted(customer_df['Segment'].unique().tolist())
        selected_segment = st.selectbox(
            "Select a Segment", 
            options=segment_list,
            key="segment_selector"
        )
        
        segment_df = customer_df[customer_df['Segment'] == selected_segment]
        customer_id_df = segment_df[['Customer ID', 'Country', 'Recency', 'MonetaryValue']]
        
        st.write(f"**{len(customer_id_df)} customers** in '{selected_segment}' segment")
        
        # Add filters
        st.markdown("**Apply Filters:**")
        min_value = st.number_input(
            "Minimum Lifetime Value ($)",
            min_value=0,
            max_value=int(customer_id_df['MonetaryValue'].max()),
            value=0,
            key="min_value_filter"
        )
        
        max_recency = st.number_input(
            "Maximum Days Since Last Purchase",
            min_value=0,
            max_value=int(customer_id_df['Recency'].max()),
            value=int(customer_id_df['Recency'].max()),
            key="max_recency_filter"
        )
        
        # Apply filters
        filtered_df = customer_id_df[
            (customer_id_df['MonetaryValue'] >= min_value) &
            (customer_id_df['Recency'] <= max_recency)
        ]
        
        st.write(f"**{len(filtered_df)} customers** match your criteria")
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(filtered_df)
        st.download_button(
            label=f"📥 Download {selected_segment} List", 
            data=csv, 
            file_name=f'{selected_segment}_customers.csv', 
            mime='text/csv',
            disabled=len(filtered_df) == 0
        )

else:
    st.warning("Could not load data to generate dashboard.")

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

