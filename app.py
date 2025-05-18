import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
import networkx as nx
import re
from preprocessing import clean_description, update_stockcode_and_description, categorize_item
from modeling import create_customer_segments, perform_association_analysis

# Page configuration
st.set_page_config(
    page_title="Retail Analysis Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Customer Segmentation", "Product Analysis", "Market Basket Analysis", "About"]
)

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/retail_data.csv")
        
        # Data preprocessing
        df['InvoiceNo'] = df['InvoiceNo'].astype('category')
        df['StockCode'] = df['StockCode'].astype('category')
        df['Description'] = df['Description'].astype('string')
        df['Country'] = df['Country'].astype('category')
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace(',', '.').astype(float)
        
        # Create derived columns
        df['is_member'] = df['CustomerID'].notna()
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
        
        # Handle non-members with synthetic IDs
        non_member_mask = df['CustomerID'].isna()
        if non_member_mask.any():
            invoice_groups = df.loc[non_member_mask, 'InvoiceNo'].unique()
            synthetic_ids = {invoice: -int(i+1) for i, invoice in enumerate(invoice_groups)}
            df.loc[non_member_mask, 'CustomerID'] = df.loc[non_member_mask, 'InvoiceNo'].map(synthetic_ids)
        
        # Fill remaining NAs
        df['CustomerID'] = df['CustomerID'].fillna(-1)
        
        # Clean descriptions
        df["Description"] = df.apply(lambda row: clean_description(row["Description"], row["StockCode"]), axis=1)
        
        # Fix StockCodes
        df['StockCode'] = df['StockCode'].astype(str)
        df = update_stockcode_and_description(df)
        
        # Add category
        df['Category'] = df['Description'].apply(categorize_item)
        
        # Additional time-based features
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        df['HourOfDay'] = df['InvoiceDate'].dt.hour
        df['Month'] = df['InvoiceDate'].dt.month
        df['Year'] = df['InvoiceDate'].dt.year
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

# Title and description
st.title("Retail Analysis Dashboard")
st.write("Interactive dashboard for analyzing online retail data")

# Overview Page
if page == "Overview":
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Time Period:** {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
        st.write(f"**Total Revenue:** ${df['TotalSales'].sum():,.2f}")
        st.write(f"**Countries:** {df['Country'].nunique()}")
        st.write(f"**Unique Products:** {df['StockCode'].nunique()}")
        st.write(f"**Customers:** {df['CustomerID'].nunique()}")
    
    with col2:
        st.subheader("Top Countries")
        country_counts = df['Country'].value_counts().head(10)
        fig = px.bar(
            x=country_counts.index, 
            y=country_counts.values,
            labels={'x': 'Country', 'y': 'Number of Orders'},
            color=country_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig)

    
    st.subheader("Product Categories Distribution")
    category_counts = df['Category'].value_counts().head(15)
    fig = px.pie(
        names=category_counts.index,
        values=category_counts.values,
        title='Top 15 Product Categories',
        hole=0.4,
    )
    st.plotly_chart(fig)


# Customer Segmentation Page
elif page == "Customer Segmentation":
    st.header("Customer Segmentation Analysis")
    
    # Get customer segmentation data
    @st.cache_data
    def get_customer_segments(df):
        return create_customer_segments(df)
    
    customer_analysis = get_customer_segments(df)
    
    st.subheader("KMeans Clustering (k=5)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatterplot of customer clusters
        fig = px.scatter(
            customer_analysis,
            x="PurchaseFrequency",
            y="TotalSalesPerCustomer",
            color="Cluster_KMeans4",
            hover_data=["CustomerID"],
            labels={
                "PurchaseFrequency": "Purchase Frequency",
                "TotalSalesPerCustomer": "Total Sales ($)",
                "Cluster_KMeans4": "Customer Segment"
            },
            title="Customer Segments",
        )
        st.plotly_chart(fig)
    
    with col2:
        # Cluster distribution
        cluster_counts = customer_analysis["Cluster_KMeans4"].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Customers'},
            title="Customers per Segment",
            color=cluster_counts.index
        )
        st.plotly_chart(fig)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    
    cluster_stats = customer_analysis.groupby("Cluster_KMeans4").agg({
        "PurchaseFrequency": ["mean", "median", "min", "max"],
        "TotalSalesPerCustomer": ["mean", "median", "min", "max"],
    }).round(2)
    
    # Format for display
    cluster_stats.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
    cluster_stats = cluster_stats.reset_index()
    
    # Give segments meaningful names based on characteristics
    segment_names = {
        0: "Low-Value Occasional",
        1: "Mid-Value Regular",
        2: "High-Value Frequent",
        3: "Top Spenders",
        4: "One-Time Big Spenders"
    }
    
    cluster_stats["Segment Name"] = cluster_stats["Cluster_KMeans4"].map(segment_names)
    
    # Reorder columns to show segment name first
    cols = cluster_stats.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cluster_stats = cluster_stats[cols]
    
    st.dataframe(cluster_stats)
    
    # Allow user to select a segment to explore
    selected_cluster = st.selectbox(
        "Select a customer segment to explore:",
        options=list(segment_names.values()),
        index=2
    )
    
    # Get cluster number from name
    selected_cluster_num = next(k for k, v in segment_names.items() if v == selected_cluster)
    
    # Filter data for the selected cluster
    cluster_customers = customer_analysis[customer_analysis["Cluster_KMeans4"] == selected_cluster_num]["CustomerID"]
    cluster_data = df[df["CustomerID"].isin(cluster_customers)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Top Categories for {selected_cluster}")
        top_categories = cluster_data["Category"].value_counts().head(10)
        fig = px.bar(
            x=top_categories.index, 
            y=top_categories.values,
            labels={'x': 'Category', 'y': 'Count'},
            color=top_categories.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)
        
    with col2:
        st.subheader(f"Purchase Times for {selected_cluster}")
        hour_counts = cluster_data["HourOfDay"].value_counts().sort_index()
        fig = px.line(
            x=hour_counts.index, 
            y=hour_counts.values,
            labels={'x': 'Hour of Day', 'y': 'Number of Purchases'},
            markers=True
        )
        st.plotly_chart(fig)
    
    # DBSCAN Anomaly Detection
    st.subheader("DBSCAN Anomaly Detection")
    # Plot DBSCAN results
    fig = px.scatter(
        customer_analysis,
        x="PurchaseFrequency",
        y="TotalSalesPerCustomer",
        color="Cluster_DBSCAN",
        hover_data=["CustomerID"],
        labels={
            "PurchaseFrequency": "Purchase Frequency",
            "TotalSalesPerCustomer": "Total Sales ($)",
            "Cluster_DBSCAN": "Cluster (DBSCAN)"
        },
        title="DBSCAN Anomaly Detection",
    )
    st.plotly_chart(fig)
    
    # Explain anomalies
    st.write("Note: In DBSCAN, cluster '-1' represents outliers (potential anomalies)")
    anomalies = customer_analysis[customer_analysis["Cluster_DBSCAN"] == -1]
    if not anomalies.empty:
        st.write(f"Found {len(anomalies)} potential anomalies/outliers")
        st.dataframe(anomalies[["CustomerID", "PurchaseFrequency", "TotalSalesPerCustomer"]].head(10))
    
# Product Analysis Page
elif page == "Product Analysis":
    st.header("Product Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Selling Products")
        top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=top_products.index, 
            y=top_products.values,
            labels={'x': 'Product', 'y': 'Quantity Sold'},
            color=top_products.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig)
        
    with col2:
        st.subheader("Most Profitable Products")
        top_revenue = df.groupby("Description")["TotalSales"].sum().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=top_revenue.index, 
            y=top_revenue.values,
            labels={'x': 'Product', 'y': 'Total Revenue'},
            color=top_revenue.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig)
    
    st.subheader("Product Categories by Revenue")
    
    category_revenue = df.groupby("Category")["TotalSales"].sum().sort_values(ascending=False).reset_index()
    fig = px.treemap(
        category_revenue,
        path=["Category"],
        values="TotalSales",
        color="TotalSales",
        color_continuous_scale='Viridis',
        title="Revenue by Product Category"
    )
    st.plotly_chart(fig)
    
    st.subheader("Product Purchase Patterns")
    
    # Purchase patterns by hour
    hour_category = pd.crosstab(df["HourOfDay"], df["Category"]).apply(lambda x: x/x.sum(), axis=1)
    
    # Select top categories by volume
    top_cats = df["Category"].value_counts().head(5).index.tolist()
    hour_category = hour_category[top_cats]
    
    # Create multi-line chart
    fig = go.Figure()
    
    for category in hour_category.columns:
        fig.add_trace(
            go.Scatter(
                x=hour_category.index, 
                y=hour_category[category],
                mode='lines+markers',
                name=category
            )
        )
    
    fig.update_layout(
        title="Purchase Patterns by Hour of Day (Top 5 Categories)",
        xaxis_title="Hour of Day",
        yaxis_title="Percentage of Category Purchases",
        legend_title="Category"
    )
    
    st.plotly_chart(fig)
    st.subheader("Geographic Distribution of Sales")
    
    # Filter StockCodes yang merupakan 5-digit angka (produk utama)
    df_filtered = df[df['StockCode'].astype(str).str.match(r'^\d{5}$')]
    
    # Hitung total penjualan per negara
    country_sales = df_filtered.groupby("Country")["TotalSales"].sum().reset_index()
    country_sales.columns = ["Country", "TotalSales"]
    country_sales["Country"] = country_sales["Country"].str.strip()
    
    # Buat peta choropleth
    fig = px.choropleth(
        country_sales,
        locations="Country",
        locationmode="country names",
        color="TotalSales",
        title="Geographic Distribution of Sales",
        color_continuous_scale="Sunsetdark",
        projection='natural earth'
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig)
    
    st.subheader("Monthly Sales Analysis")
    
    monthly_sales = df.groupby(df["InvoiceDate"].dt.month).agg(
        TotalSales=("TotalSales", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        TransactionCount=("InvoiceNo", "nunique")
    ).reset_index()
    
    monthly_sales["Month"] = monthly_sales["InvoiceDate"].apply(
        lambda x: pd.Timestamp(2023, x, 1).strftime('%B')  # Konversi nomor bulan ke nama bulan
    )
    
    # Tampilkan visualisasi bulanan
    fig = px.line(
        monthly_sales,
        x="Month",
        y=["TotalSales", "UniqueProducts", "TransactionCount"],
        title="Monthly Sales Metrics",
        markers=True,
        labels={
            "value": "Value",
            "variable": "Metric",
            "Month": "Month"
        }
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]}
    )
    
    st.plotly_chart(fig)
    st.subheader("Category Explorer")
    
    selected_category = st.selectbox(
        "Select a category to explore:",
        options=sorted(df["Category"].unique())
    )
    
    category_data = df[df["Category"] == selected_category]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Top Products in {selected_category}")
        cat_products = category_data.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=cat_products.index, 
            y=cat_products.values,
            labels={'x': 'Product', 'y': 'Quantity Sold'},
            color=cat_products.values,
            color_continuous_scale='Viridis'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig)
        
    with col2:
        st.write(f"{selected_category} Sales by Country")
        cat_countries = category_data.groupby("Country")["TotalSales"].sum().sort_values(ascending=False).head(10)
        fig = px.pie(
            names=cat_countries.index,
            values=cat_countries.values,
            hole=0.4,
        )
        st.plotly_chart(fig)

# Market Basket Analysis Page
elif page == "Market Basket Analysis":
    st.header("Market Basket Analysis")
    
    st.write("""
    Market basket analysis helps identify which products are frequently purchased together.
    This can provide insights for cross-selling, store layouts, and promotional strategies.
    """)
    
    # Get customer segmentation for filtering
    @st.cache_data
    def get_customer_segments(df):
        return create_customer_segments(df)
    
    customer_analysis = get_customer_segments(df)
    
    segment_names = {
        0: "Low-Value Occasional",
        1: "Mid-Value Regular",
        2: "High-Value Frequent",
        3: "Top Spenders",
        4: "One-Time Big Spenders",
        "all": "All Customers"
    }
    
    # Allow user to select a segment for basket analysis
    selected_segment = st.selectbox(
        "Select a customer segment for basket analysis:",
        options=list(segment_names.values()),
        index=5  # Default to "All Customers"
    )
    
    # Parameters for association rules
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider("Minimum Support:", 0.01, 0.5, 0.1, 0.01)
    with col2:
        min_confidence = st.slider("Minimum Confidence:", 0.1, 1.0, 0.5, 0.05)
    with col3:
        min_lift = st.slider("Minimum Lift:", 1.0, 10.0, 1.2, 0.1)
    
    # Get basket analysis results
    @st.cache_data
    def get_basket_analysis(df, segment, min_support, min_confidence, min_lift):
        # Filter by segment if not "All Customers"
        if segment != "All Customers":
            segment_num = next(k for k, v in segment_names.items() if v == segment)
            customer_ids = customer_analysis[customer_analysis["Cluster_KMeans4"] == segment_num]["CustomerID"]
            segment_df = df[df["CustomerID"].isin(customer_ids)]
        else:
            segment_df = df.copy()
        
        # Filter out returns
        segment_df = segment_df[~segment_df["InvoiceNo"].astype(str).str.startswith("C")]
        
        # Run the analysis
        return perform_association_analysis(segment_df, min_support, min_confidence, min_lift)
    
    with st.spinner("Running market basket analysis..."):
        frequent_itemsets, rules, G = get_basket_analysis(df, selected_segment, min_support, min_confidence, min_lift)
    
    if rules.empty:
        st.warning("No association rules found with the current parameters. Try lowering the thresholds.")
    else:
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Frequent Itemsets")
            st.write(f"Found {len(frequent_itemsets)} frequent itemsets")
            st.dataframe(frequent_itemsets)
            
        with col2:
            st.subheader("Association Rules")
            st.write(f"Found {len(rules)} association rules")
            
            # Format rules for better display
            rules_display = rules.copy()
            rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(list(x)))
            rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(list(x)))
            
            st.dataframe(rules_display[["antecedents", "consequents", "support", "confidence", "lift"]])
        
        # Network visualization
        st.subheader("Network Visualization of Association Rules")
        
        # Check if there are nodes in the graph
        if len(G.nodes()) > 0:
            # Convert networkx graph to plotly
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = G.nodes[edge[0]]['pos']
                x1, y1 = G.nodes[edge[1]]['pos']
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = G.nodes[node]['pos']
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))

            # Create edges
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            # Create nodes
            node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=10,
                        colorbar=dict(
                            thickness=15,
                            title=dict(
                                text='Node Connections',
                                side='right'  # Gunakan title sebagai dict dengan parameter side
                            ),
                            xanchor='left'
                        )
                    ))

            # Color nodes by number of connections
            node_adjacencies = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                
            node_trace.marker.color = node_adjacencies

            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=f'Network of Association Rules - {selected_segment}',
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough connections to create a network visualization. Try adjusting the parameters.")
        
        # Top association rules interpretation
        st.subheader("Top Association Rules")
        
        top_rules = rules.sort_values("lift", ascending=False).head(5)
        
        for i, row in enumerate(top_rules.itertuples()):
            antecedents = ", ".join(list(row.antecedents))
            consequents = ", ".join(list(row.consequents))
            st.write(f"**Rule {i+1}:** Customers who purchase **{antecedents}** are likely to also purchase **{consequents}**")
            st.write(f"- Support: {row.support:.3f} (appears in {row.support*100:.1f}% of transactions)")
            st.write(f"- Confidence: {row.confidence:.3f} ({row.confidence*100:.1f}% of customers who bought {antecedents} also bought {consequents})")
            st.write(f"- Lift: {row.lift:.3f} (purchases are {row.lift:.1f}x more likely to occur together than by random chance)")
            st.write("---")

# About Page
elif page == "About":
    st.header("About This Dashboard")
    
    st.write("""
    This interactive dashboard provides analysis of an online retail dataset.
    Created by Threegenerations:
             - Anthony Edbert Feriyanto
             - Carlene Annabel
             - Edbert Halim
             - Natania Deandra
    
    ### Features:
    - **Overview**: General statistics and trends
    - **Customer Segmentation**: KMeans clustering and DBSCAN anomaly detection
    - **Product Analysis**: Top products, categories, and purchase patterns
    - **Market Basket Analysis**: Association rules for product relationships
    
    ### Data Pre-processing Steps:
    - Cleaned product descriptions and standardized stock codes
    - Categorized products into meaningful groups
    - Created customer segments based on purchase behavior
    - Generated synthetic IDs for non-member transactions
    
    ### Technologies Used:
    - Python
    - Pandas & NumPy for data processing
    - Scikit-learn for machine learning
    - MLxtend for association rule mining
    - Plotly & Matplotlib for visualization
    - Streamlit for dashboard deployment
    """)
    
    st.subheader("How to Use This Dashboard")
    
    st.write("""
    1. Use the sidebar to navigate between different analysis pages
    2. On the Customer Segmentation page, select different segments to explore their characteristics
    3. On the Product Analysis page, explore different product categories 
    4. On the Market Basket Analysis page, adjust parameters to find meaningful association rules
    """)
    
# Show timestamp at the bottom
st.sidebar.markdown("---")
st.sidebar.info(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")