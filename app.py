import streamlit as st
import pandas as pd
import numpy as np
import io

from utils.data_processor import load_data, auto_detect_columns, clean_data, scale_features
from utils.clustering import calculate_elbow_and_silhouette, find_optimal_k, fit_kmeans, generate_cluster_personas
from utils.visualizations import (
    plot_elbow_curve, plot_silhouette_scores, plot_2d_clusters, 
    plot_3d_clusters, plot_cluster_comparison, plot_correlation_heatmap, plot_gender_breakdown
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Analytics Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    
    /* Persona Card */
    .persona-card {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .dark .persona-card {
        background-color: #1e293b;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🎯 Customer Segmentation & Analytics Dashboard</h1>
        <p class="header-subtitle">Uncover hidden customer personas using K-Means Clustering & Automatic Elbow Method Optimization</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("📁 Data & Configuration")

# 1. Dataset Selection
data_source = st.sidebar.radio(
    "Choose Dataset Source:",
    ["Benchmark Dataset (Mall Customers)", "Upload Custom CSV File"],
    index=0
)

df_raw = None
error_msg = None

if data_source == "Benchmark Dataset (Mall Customers)":
    default_path = "d:/Customer_Segmentation/Mall_Customers.csv"
    df_raw, error_msg = load_data(default_path)
    if error_msg:
        st.sidebar.error(error_msg)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Customer CSV File", type=["csv"])
    if uploaded_file is not None:
        df_raw, error_msg = load_data(uploaded_file)
        if error_msg:
            st.sidebar.error(error_msg)
    else:
        st.info("👈 Please upload a CSV dataset from the sidebar, or switch to the Benchmark Dataset.")
        st.stop()

if df_raw is None or df_raw.empty:
    st.error("No valid dataset available to display. Please verify the CSV file.")
    st.stop()

# 2. Auto-Detect Columns
column_mapping = auto_detect_columns(df_raw)
numeric_columns = column_mapping['numeric_cols']

if len(numeric_columns) < 2:
    st.error("⚠️ The dataset must contain at least 2 numerical features for clustering.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Feature Selection")

# Select X and Y features for clustering
default_x_idx = numeric_columns.index(column_mapping['income']) if column_mapping['income'] in numeric_columns else 0
default_y_idx = numeric_columns.index(column_mapping['spending']) if column_mapping['spending'] in numeric_columns and len(numeric_columns) > 1 else min(1, len(numeric_columns)-1)

x_feature = st.sidebar.selectbox("Select Primary Feature (e.g. Annual Income):", numeric_columns, index=default_x_idx)
y_feature = st.sidebar.selectbox("Select Secondary Feature (e.g. Spending Score):", numeric_columns, index=default_y_idx)

# Optional Z Feature for 3D
other_numeric = [col for col in numeric_columns if col not in [x_feature, y_feature]]
z_feature_options = ["None"] + numeric_columns
default_z_idx = z_feature_options.index(column_mapping['age']) if column_mapping['age'] in z_feature_options else 0
z_feature = st.sidebar.selectbox("Select Optional 3D Feature (e.g. Age):", z_feature_options, index=default_z_idx)

selected_features = [x_feature, y_feature]
if z_feature != "None" and z_feature not in selected_features:
    selected_features.append(z_feature)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Clustering Parameters")

scaler_type = st.sidebar.selectbox("Feature Scaler:", ["StandardScaler", "MinMaxScaler"], index=0)
max_k = st.sidebar.slider("Max K for Elbow Method:", min_value=3, max_value=15, value=10)

# Clean and scale data
cleaned_df, clean_report = clean_data(df_raw, selected_features, handle_missing="impute")
scaled_data, scaler_instance = scale_features(cleaned_df, selected_features, method=scaler_type)

# Calculate Elbow & Silhouette curves
k_range, inertia_list, silhouette_list = calculate_elbow_and_silhouette(scaled_data, max_k=max_k)
auto_k = find_optimal_k(k_range, inertia_list)

use_auto_k = st.sidebar.checkbox("Auto-detect Optimal K (Elbow Method)", value=True)

if use_auto_k:
    n_clusters = auto_k
    st.sidebar.success(f"🎯 Recommended K = **{n_clusters}** (Elbow Point)")
else:
    n_clusters = st.sidebar.slider("Select Manual Number of Clusters (K):", min_value=2, max_value=max_k, value=auto_k)

# Fit K-Means
cluster_labels, kmeans_model, current_sil_score, centroids_scaled = fit_kmeans(scaled_data, n_clusters)
cleaned_df['Cluster'] = cluster_labels

# Convert centroids back to original scale
centroids_orig = scaler_instance.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_orig, columns=selected_features)

# Generate Persona Mappings
income_col_mapped = column_mapping['income'] if column_mapping['income'] in selected_features else x_feature
spending_col_mapped = column_mapping['spending'] if column_mapping['spending'] in selected_features else y_feature
persona_mapping = generate_cluster_personas(cleaned_df, 'Cluster', income_col_mapped, spending_col_mapped)

# ---------------------------------------------------------
# Key Metric Cards (Top Section)
# ---------------------------------------------------------
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

total_cust = len(cleaned_df)
avg_inc = cleaned_df[x_feature].mean()
avg_spn = cleaned_df[y_feature].mean()

with col_m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Customers</div>
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-subtitle">Processed Rows</div>
        </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg {x_feature[:14]}</div>
            <div class="metric-value">{avg_inc:.1f}</div>
            <div class="metric-subtitle">Dataset Average</div>
        </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg {y_feature[:14]}</div>
            <div class="metric-value">{avg_spn:.1f}</div>
            <div class="metric-subtitle">Dataset Average</div>
        </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Cluster Count (K)</div>
            <div class="metric-value">{n_clusters}</div>
            <div class="metric-subtitle">{'Auto Elbow' if use_auto_k else 'Manual Select'}</div>
        </div>
    """, unsafe_allow_html=True)

with col_m5:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Silhouette Score</div>
            <div class="metric-value">{current_sil_score:.3f}</div>
            <div class="metric-subtitle">Fit Quality</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview & EDA", 
    "📈 Elbow Method", 
    "🎯 Customer Segments", 
    "👥 Cluster Profiles", 
    "🔮 Live Predictor",
    "📥 Export Data"
])

# ---------------------------------------------------------
# TAB 1: OVERVIEW & EDA
# ---------------------------------------------------------
with tab1:
    st.subheader("📋 Dataset Preview & Exploratory Data Analysis")
    
    col_t1_left, col_t1_right = st.columns([6, 4])
    
    with col_t1_left:
        st.markdown("**Sample Data (First 10 Rows)**")
        st.dataframe(cleaned_df.head(10), use_container_width=True)
        
        if clean_report['duplicates_removed'] > 0 or clean_report['missing_values_imputed'] > 0:
            st.info(f"🧹 **Data Cleaning Notice**: Removed {clean_report['duplicates_removed']} duplicate rows and imputed {clean_report['missing_values_imputed']} missing values.")

    with col_t1_right:
        st.markdown("**Numerical Summary Statistics**")
        st.dataframe(cleaned_df[numeric_columns].describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)

    st.markdown("---")
    col_t1_bot1, col_t1_bot2 = st.columns(2)
    
    with col_t1_bot1:
        st.plotly_chart(plot_correlation_heatmap(cleaned_df, numeric_columns), use_container_width=True)
        
    with col_t1_bot2:
        st.markdown("### 📌 Feature Selection Summary")
        st.write(f"- **Primary Feature (X):** `{x_feature}`")
        st.write(f"- **Secondary Feature (Y):** `{y_feature}`")
        if z_feature != "None":
            st.write(f"- **3D Feature (Z):** `{z_feature}`")
        st.write(f"- **Scaling Technique:** `{scaler_type}`")
        st.write(f"- **Detected Gender Column:** `{column_mapping['gender'] or 'None'}`")
        st.write(f"- **Detected Age Column:** `{column_mapping['age'] or 'None'}`")

# ---------------------------------------------------------
# TAB 2: ELBOW METHOD & OPTIMIZATION
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Determining Optimal Clusters (K)")
    
    col_t2_1, col_t2_2 = st.columns(2)
    
    with col_t2_1:
        st.plotly_chart(plot_elbow_curve(k_range, inertia_list, auto_k), use_container_width=True)
        
    with col_t2_2:
        st.plotly_chart(plot_silhouette_scores(k_range, silhouette_list, n_clusters), use_container_width=True)
        
    with st.expander("🎓 **How does the Elbow Method & Silhouette Coefficient work?**", expanded=True):
        st.markdown("""
        ### Mathematical Rationale for Project Defense
        
        1. **Elbow Method (Inertia / WCSS)**:
           - **Within-Cluster Sum of Squares (WCSS)** measures the sum of squared distances between each point and its assigned cluster centroid.
           - Formula: $$WCSS = \\sum_{k=1}^{K} \\sum_{x_i \\in C_k} ||x_i - \\mu_k||^2$$
           - As $K$ increases, WCSS naturally decreases towards 0. The **Elbow Point** represents the point of diminishing returns, where adding more clusters yields significantly smaller drops in inertia.
           - *Our algorithm automatically detects this knee using the maximum perpendicular distance from the curve chord.*
           
        2. **Silhouette Coefficient**:
           - Measures how similar an object is to its own cluster compared to other clusters. Range: **[-1, +1]**.
           - Higher values indicate well-separated, dense clusters.
        """)

# ---------------------------------------------------------
# TAB 3: CUSTOMER SEGMENTS VISUALIZATION
# ---------------------------------------------------------
with tab3:
    st.subheader("🎯 Customer Segments Visualization")
    
    st.plotly_chart(plot_2d_clusters(cleaned_df, x_feature, y_feature, 'Cluster', centroids_df, persona_mapping), use_container_width=True)
    
    if z_feature != "None":
        st.markdown("---")
        st.subheader(f"🌐 3D Cluster Visualization ({x_feature} vs {y_feature} vs {z_feature})")
        st.plotly_chart(plot_3d_clusters(cleaned_df, x_feature, y_feature, z_feature, 'Cluster', persona_mapping), use_container_width=True)
        
    st.markdown("### 🏷️ Identified Customer Personas")
    
    p_cols = st.columns(min(n_clusters, 3))
    for idx, (c_id, info) in enumerate(persona_mapping.items()):
        col_target = p_cols[idx % min(n_clusters, 3)]
        with col_target:
            st.markdown(f"""
                <div class="persona-card">
                    <h4>{info['name']}</h4>
                    <p style="font-size:0.9rem;">{info['description']}</p>
                    <hr style="margin: 8px 0;">
                    <p style="margin:0; font-size:0.85rem;">👥 <b>Count:</b> {info['count']} ({info['pct']}%)</p>
                    {f"<p style='margin:0; font-size:0.85rem;'>💰 <b>Mean {x_feature}:</b> {info.get('mean_income', 'N/A')}</p>" if 'mean_income' in info else ''}
                    {f"<p style='margin:0; font-size:0.85rem;'>🛍️ <b>Mean {y_feature}:</b> {info.get('mean_spending', 'N/A')}</p>" if 'mean_spending' in info else ''}
                </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4: CLUSTER PROFILES & DEEP-DIVE
# ---------------------------------------------------------
with tab4:
    st.subheader("👥 Cluster Profiles & Demographic Breakdown")
    
    col_t4_1, col_t4_2 = st.columns(2)
    
    with col_t4_1:
        st.plotly_chart(plot_cluster_comparison(cleaned_df, 'Cluster', [x_feature, y_feature], persona_mapping), use_container_width=True)
        
    with col_t4_2:
        gender_col = column_mapping['gender']
        if gender_col and gender_col in cleaned_df.columns:
            st.plotly_chart(plot_gender_breakdown(cleaned_df, 'Cluster', gender_col, persona_mapping), use_container_width=True)
        else:
            st.info("No Gender column detected in dataset to display gender breakdown.")
            
    st.markdown("### 📊 Comprehensive Cluster Statistics Table")
    agg_dict = {x_feature: ['mean', 'median', 'min', 'max'], y_feature: ['mean', 'median', 'min', 'max']}
    if column_mapping['age'] and column_mapping['age'] in cleaned_df.columns:
        agg_dict[column_mapping['age']] = ['mean', 'median']
        
    summary_table = cleaned_df.groupby('Cluster').agg(agg_dict)
    st.dataframe(summary_table, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: LIVE CUSTOMER PREDICTOR
# ---------------------------------------------------------
with tab5:
    st.subheader("🔮 Predict Customer Segment for New Entry")
    st.markdown("Enter customer details below to dynamically classify which segment they belong to.")
    
    with st.form("prediction_form"):
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            val_x = st.number_input(
                f"Enter {x_feature}:",
                min_value=float(cleaned_df[x_feature].min()),
                max_value=float(cleaned_df[x_feature].max() * 2),
                value=float(cleaned_df[x_feature].median())
            )
            
        with col_p2:
            val_y = st.number_input(
                f"Enter {y_feature}:",
                min_value=float(cleaned_df[y_feature].min()),
                max_value=float(cleaned_df[y_feature].max() * 2),
                value=float(cleaned_df[y_feature].median())
            )
            
        val_z = None
        if z_feature != "None":
            val_z = st.number_input(
                f"Enter {z_feature}:",
                min_value=float(cleaned_df[z_feature].min()),
                max_value=float(cleaned_df[z_feature].max() * 2),
                value=float(cleaned_df[z_feature].median())
            )
            
        submit_btn = st.form_submit_button("🎯 Predict Segment")
        
    if submit_btn:
        input_data = [val_x, val_y]
        if val_z is not None:
            input_data.append(val_z)
            
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler_instance.transform(input_array)
        pred_cluster = kmeans_model.predict(input_scaled)[0]
        
        persona_info = persona_mapping.get(pred_cluster, {})
        p_name = persona_info.get('name', f'Cluster {pred_cluster}')
        p_desc = persona_info.get('description', 'Classified Customer Segment')
        
        st.success(f"### 🎉 Result: Assigned to **{p_name}** (Cluster {pred_cluster})")
        st.markdown(f"**Persona Rationale:** {p_desc}")

# ---------------------------------------------------------
# TAB 6: DATA EXPORT & FILTERING
# ---------------------------------------------------------
with tab6:
    st.subheader("📥 Filter & Export Segmented Data")
    
    # Add Persona names to dataframe for export
    export_df = cleaned_df.copy()
    export_df['Segment_Persona'] = export_df['Cluster'].apply(lambda c: persona_mapping.get(c, {}).get('name', f"Cluster {c}"))
    
    selected_cluster_filter = st.multiselect(
        "Filter Data by Segment:",
        options=list(export_df['Segment_Persona'].unique()),
        default=list(export_df['Segment_Persona'].unique())
    )
    
    filtered_export_df = export_df[export_df['Segment_Persona'].isin(selected_cluster_filter)]
    st.dataframe(filtered_export_df, use_container_width=True)
    
    # CSV Download Button
    csv_buffer = io.StringIO()
    filtered_export_df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Segmented CSV File",
        data=csv_str,
        file_name="segmented_customers.csv",
        mime="text/csv",
        type="primary"
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Customer Segmentation & Analytics Dashboard | Built with Streamlit, Pandas & Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
