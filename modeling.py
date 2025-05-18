import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
import networkx as nx

def create_customer_segments(df):
    """
    Create customer segments using KMeans clustering and DBSCAN for anomaly detection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The retail dataset with required columns
    
    Returns:
    --------
    pandas.DataFrame
        Customer analysis dataframe with cluster assignments
    """
    # Filter out returns and null CustomerIDs
    df_filtered = df[(~df["InvoiceNo"].astype(str).str.startswith("C")) & (df["CustomerID"] != -1)]
    
    # Calculate Purchase Frequency
    customer_frequency = df_filtered.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    customer_frequency.columns = ["CustomerID", "PurchaseFrequency"]
    
    # Calculate Total Sales
    df_filtered_sales = df.copy()
    df_filtered_sales["TotalSales"] = df_filtered_sales["Quantity"] * df_filtered_sales["UnitPrice"]
    customer_sales = df_filtered_sales.groupby("CustomerID")["TotalSales"].sum().reset_index()
    customer_sales.columns = ["CustomerID", "TotalSalesPerCustomer"]
    
    # Merge for clustering
    customer_analysis = pd.merge(customer_frequency, customer_sales, on="CustomerID")
    
    # Add categories purchased by each customer
    customer_category = df.groupby("CustomerID")["Category"].unique().reset_index()
    customer_category["CategoryList"] = customer_category["Category"].apply(lambda x: ", ".join(map(str, x)))
    customer_category.drop("Category", axis=1, inplace=True)
    
    # Merge
    customer_analysis = customer_analysis.merge(customer_category, on="CustomerID", how="left")
    
    # Scale features for clustering
    X = customer_analysis[["PurchaseFrequency", "TotalSalesPerCustomer"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering with k=5
    kmeans = KMeans(n_clusters=5, random_state=42)
    customer_analysis["Cluster_KMeans4"] = kmeans.fit_predict(X_scaled)
    
    # DBSCAN for anomaly detection
    dbscan = DBSCAN(eps=2.5, min_samples=3)
    customer_analysis["Cluster_DBSCAN"] = dbscan.fit_predict(X_scaled)
    
    return customer_analysis

def perform_association_analysis(df, min_support=0.1, min_confidence=0.5, min_lift=1.0):
    """
    Perform market basket analysis using FP-Growth algorithm and create a network visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The retail dataset with required columns
    min_support : float
        Minimum support threshold for FP-Growth
    min_confidence : float
        Minimum confidence threshold for association rules
    min_lift : float
        Minimum lift threshold for association rules
    
    Returns:
    --------
    tuple
        (frequent_itemsets, rules, network_graph)
    """
    # Filter out returns
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    
    # Convert to string format for consistency
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['Category'] = df['Category'].astype(str)
    
    # Create basket: InvoiceNo x Category (binary)
    basket = (df.groupby(['InvoiceNo', 'Category'])['Quantity']
              .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
    
    # Convert to binary format (1 = item purchased, 0 = not purchased)
    basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Run FP-Growth algorithm
    frequent_itemsets = fpgrowth(basket_binary, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift].sort_values(by='lift', ascending=False)
    else:
        # If no frequent itemsets found, return empty dataframe
        rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    
    # Create network graph visualization if rules exist
    G = nx.DiGraph()
    
    if not rules.empty:
        # Add nodes and edges to the graph
        for _, row in rules.iterrows():
            for antecedent in row['antecedents']:
                for consequent in row['consequents']:
                    G.add_edge(antecedent, consequent, weight=row['lift'])
        
        # Set positions for nodes using spring layout
        pos = nx.spring_layout(G, k=0.6, seed=42)
        for node in G.nodes():
            G.nodes[node]['pos'] = pos[node]
    
    return frequent_itemsets, rules, G

def generate_rfm_segments(df):
    """
    Generate RFM (Recency, Frequency, Monetary) segments for customers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The retail dataset with required columns
    
    Returns:
    --------
    pandas.DataFrame
        Customer RFM analysis dataframe with segment assignments
    """
    # Filter out returns and null CustomerIDs
    df_filtered = df[(~df["InvoiceNo"].astype(str).str.startswith("C")) & (df["CustomerID"] != -1)].copy()
    
    # Get the maximum date to calculate recency
    max_date = df_filtered['InvoiceDate'].max()
    
    # Calculate Recency (days since last purchase)
    recency_df = df_filtered.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency_df.columns = ['CustomerID', 'LastPurchaseDate']
    recency_df['Recency'] = (max_date - recency_df['LastPurchaseDate']).dt.days
    
    # Calculate Frequency (number of purchases)
    frequency_df = df_filtered.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    frequency_df.columns = ['CustomerID', 'Frequency']
    
    # Calculate Monetary (total spending)
    df_filtered['TotalSales'] = df_filtered['Quantity'] * df_filtered['UnitPrice']
    monetary_df = df_filtered.groupby('CustomerID')['TotalSales'].sum().reset_index()
    monetary_df.columns = ['CustomerID', 'Monetary']
    
    # Merge all three metrics
    rfm_df = recency_df.merge(frequency_df, on='CustomerID')
    rfm_df = rfm_df.merge(monetary_df, on='CustomerID')
    
    # Create RFM segmentation
    # Score from 1-5 (5 is the best, 1 is the worst)
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    # Calculate RFM Score
    rfm_df['RFM_Score'] = (rfm_df['R_Score'].astype(int) 
                          + rfm_df['F_Score'].astype(int) 
                          + rfm_df['M_Score'].astype(int))
    
    # Define segments based on RFM score
    segment_mapping = {
        13: 'Champions',
        14: 'Champions',
        15: 'Champions',
        10: 'Loyal Customers',
        11: 'Loyal Customers',
        12: 'Loyal Customers',
        7: 'Potential Loyalists',
        8: 'Potential Loyalists',
        9: 'Potential Loyalists',
        5: 'At Risk Customers',
        6: 'At Risk Customers',
        3: 'Hibernating Customers',
        4: 'Hibernating Customers',
    }
    
    # Apply segment names
    rfm_df['RFM_Segment'] = rfm_df['RFM_Score'].map(lambda x: segment_mapping.get(x, 'Needs Attention'))
    
    return rfm_df

def analyze_product_seasonality(df):
    """
    Analyze product seasonality patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The retail dataset with required columns
    
    Returns:
    --------
    dict
        Dictionary with various seasonal analyses
    """
    # Add time-based features if they don't exist
    df = df.copy()
    if 'Month' not in df.columns:
        df['Month'] = df['InvoiceDate'].dt.month
    if 'DayOfWeek' not in df.columns:
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    if 'Quarter' not in df.columns:
        df['Quarter'] = df['InvoiceDate'].dt.quarter
    
    monthly_category = df.groupby(['Month', 'Category'])['TotalSales'].sum().reset_index()
    dow_sales = df.groupby('DayOfWeek')['TotalSales'].sum().reset_index()
    quarterly_sales = df.groupby(['Quarter', 'Category'])['TotalSales'].sum().reset_index()
    
    seasonal_top_products = {}
    for quarter in df['Quarter'].unique():
        quarter_df = df[df['Quarter'] == quarter]
        top_products = quarter_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        seasonal_top_products[f'Q{quarter}'] = top_products
    
    return {
        'monthly_category': monthly_category,
        'dow_sales': dow_sales,
        'quarterly_sales': quarterly_sales,
        'seasonal_top_products': seasonal_top_products
    }