import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import io

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛍️ Customer Segmentation Analysis</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Analysis Controls")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file", 
    type=['csv'],
    help="Upload a CSV file with customer data"
)

# Use sample data if no file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
else:
    st.sidebar.info("Using sample Mall Customers data")
    # Create sample data structure for demo
    np.random.seed(42)
    sample_data = {
        'CustomerID': range(1, 201),
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 140, 200),
        'Spending Score (1-100)': np.random.randint(1, 101, 200)
    }
    df = pd.DataFrame(sample_data)

# Data preprocessing
@st.cache_data
def preprocess_data(data):
    df_processed = data.copy()
    if 'CustomerID' in df_processed.columns:
        df_processed = df_processed.drop("CustomerID", axis=1)
    
    if 'Gender' in df_processed.columns:
        df_processed["Gender"] = df_processed["Gender"].map({"Male": 0, "Female": 1})
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_processed.select_dtypes(include=[np.number]))
    
    return df_processed, scaled_data, scaler

df_processed, scaled_data, scaler = preprocess_data(df)

# Sidebar parameters
st.sidebar.subheader("🎛️ Clustering Parameters")
max_clusters = st.sidebar.slider("Maximum clusters for elbow method", 2, 15, 10)
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)
random_state = st.sidebar.number_input("Random state", 0, 100, 42)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Data Overview", 
    "🔍 Elbow Method", 
    "🎯 Clustering Results", 
    "📊 Cluster Analysis", 
    "🌐 3D Visualization"
])

with tab1:
    st.header("📈 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("**Sample Data:**")
        st.dataframe(df.head(10))
        
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

with tab2:
    st.header("🔍 Elbow Method Analysis")
    
    # Calculate inertia
    @st.cache_data
    def calculate_inertia(data, max_k, rs):
        inertia = []
        k_range = range(1, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=rs, n_init=10)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)
        return k_range, inertia
    
    k_range, inertia = calculate_inertia(scaled_data, max_clusters, random_state)
    
    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), 
        y=inertia,
        mode='lines+markers',
        name='Inertia',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='📈 Elbow Method - Optimal Number of Clusters',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Inertia',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show inertia values
    st.subheader("📋 Inertia Values")
    inertia_df = pd.DataFrame({'Clusters': k_range, 'Inertia': inertia})
    st.dataframe(inertia_df)

with tab3:
    st.header("🎯 Clustering Results")
    
    # Perform clustering
    @st.cache_data
    def perform_clustering(data, n_clust, rs):
        kmeans = KMeans(n_clusters=n_clust, random_state=rs, n_init=10)
        clusters = kmeans.fit_predict(data)
        return clusters, kmeans
    
    clusters, kmeans_model = perform_clustering(scaled_data, n_clusters, random_state)
    df_clustered = df_processed.copy()
    df_clustered['Cluster'] = clusters
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Distribution")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        
        fig = px.pie(
            values=cluster_counts.values, 
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title='Distribution of Customers by Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔢 Cluster Sizes")
        for i, count in enumerate(cluster_counts):
            st.markdown(f"""
            <div class="metric-container">
                <h4>Cluster {i}</h4>
                <h2>{count} customers ({count/len(df_clustered)*100:.1f}%)</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # PCA Visualization
    st.subheader("🎨 PCA Visualization")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    fig = px.scatter(
        x=pca_data[:, 0], 
        y=pca_data[:, 1], 
        color=[f'Cluster {c}' for c in clusters],
        title='Customer Clusters (PCA Projection)',
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📊 Cluster Analysis")
    
    # Cluster summary
    cluster_summary = df_clustered.groupby("Cluster").agg({
        col: ['mean', 'std', 'count'] for col in df_clustered.select_dtypes(include=[np.number]).columns if col != 'Cluster'
    }).round(2)
    
    st.subheader("📋 Cluster Summary Statistics")
    st.dataframe(cluster_summary)
    
    # Individual cluster analysis
    st.subheader("🔍 Detailed Cluster Analysis")
    
    selected_cluster = st.selectbox("Select cluster to analyze:", 
                                   options=sorted(df_clustered['Cluster'].unique()),
                                   format_func=lambda x: f"Cluster {x}")
    
    cluster_data = df_clustered[df_clustered['Cluster'] == selected_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Cluster {selected_cluster} Overview**")
        st.write(f"Size: {len(cluster_data)} customers")
        st.dataframe(cluster_data.describe())
    
    with col2:
        # Feature comparison
        numeric_features = df_clustered.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != 'Cluster']
        
        if len(numeric_features) > 0:
            selected_feature = st.selectbox("Select feature for distribution:", numeric_features)
            
            fig = px.histogram(
                df_clustered, 
                x=selected_feature, 
                color=[f'Cluster {c}' for c in df_clustered['Cluster']],
                title=f'Distribution of {selected_feature} by Cluster',
                marginal='box'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("🌐 3D Visualization")
    
    numeric_cols = [col for col in df_clustered.select_dtypes(include=[np.number]).columns if col != 'Cluster']
    
    if len(numeric_cols) >= 3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis:", numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        with col3:
            z_axis = st.selectbox("Z-axis:", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
        
        # 3D scatter plot
        fig = px.scatter_3d(
            df_clustered, 
            x=x_axis, 
            y=y_axis, 
            z=z_axis,
            color=[f'Cluster {c}' for c in df_clustered['Cluster']],
            title=f'3D Cluster Visualization: {x_axis} vs {y_axis} vs {z_axis}',
            height=700
        )
        
        fig.update_layout(scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=z_axis
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 3 numeric columns for 3D visualization")

# Download results
st.sidebar.subheader("💾 Download Results")
if st.sidebar.button("Download Clustered Data"):
    csv = df_clustered.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    "**Customer Segmentation Analysis** | Built with Streamlit 🚀 | "
    "Upload your data and explore customer segments!"
)