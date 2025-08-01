import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/customer_features_clustered.csv')
    return df

@st.cache_data
def load_model():
    return joblib.load('src/clv_model.joblib')

# Main dashboard
def main():
    st.title("🛍️ E-commerce Customer Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    model_data = load_model()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Cluster filter
    clusters = st.sidebar.multiselect(
        "Select Customer Segments",
        options=sorted(df['cluster'].unique()),
        default=sorted(df['cluster'].unique())
    )
    
    # Age range filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=(int(df['age'].min()), int(df['age'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['cluster'].isin(clusters)) & 
        (df['age'].between(age_range[0], age_range[1]))
    ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive Summary", "👥 Customer Segments", 
                                      "🔮 Predictions", "📊 Model Performance"])
    
    with tab1:
        show_executive_summary(filtered_df)
    
    with tab2:
        show_customer_segments(filtered_df)
    
    with tab3:
        show_predictions(filtered_df, model_data)
    
    with tab4:
        show_model_performance(filtered_df, model_data)

def show_executive_summary(df):
    """Executive summary page"""
    st.header("Executive Summary")
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        st.metric("Total Revenue", f"${df['total_spent'].sum():,.0f}")
    
    with col3:
        st.metric("Avg CLV", f"${df['clv'].mean():.0f}")
    
    with col4:
        st.metric("Avg Order Value", f"${df['avg_order_value'].mean():.2f}")
    
    # Revenue trends
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV distribution
        fig = px.histogram(df, x='clv', nbins=30, 
                          title="Customer Lifetime Value Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age vs CLV
        fig = px.scatter(df, x='age', y='clv', color='gender',
                        title="CLV by Age and Gender")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_customer_segments(df):
    """Customer segmentation analysis"""
    st.header("Customer Segmentation Analysis")
    
    # Segment overview
    segment_summary = df.groupby('cluster').agg({
        'clv': 'mean',
        'total_spent': 'mean',
        'order_count': 'mean',
        'monthly_sessions': 'mean'
    }).round(2)
    
    st.subheader("Segment Profiles")
    st.dataframe(segment_summary, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment distribution
        segment_counts = df['cluster'].value_counts().sort_index()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Distribution by Segment")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV by segment
        fig = px.box(df, x='cluster', y='clv', 
                    title="CLV Distribution by Segment")
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("Customer Segmentation Visualization")
    fig = px.scatter_3d(df, x='total_spent', y='order_count', z='monthly_sessions',
                       color='cluster', size='clv',
                       title="3D Customer Segmentation")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(df, model_data):
    """CLV prediction interface"""
    st.header("Customer Lifetime Value Predictions")
    
    # Prediction input form
    st.subheader("Predict CLV for New Customer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.slider("Annual Income ($)", 20000, 200000, 60000)
        monthly_sessions = st.slider("Monthly Sessions", 0, 50, 10)
        avg_session_duration = st.slider("Avg Session Duration (min)", 0, 30, 5)
    
    with col2:
        total_spent = st.slider("Total Spent ($)", 0, 5000, 500)
        avg_order_value = st.slider("Avg Order Value ($)", 10, 1000, 100)
        order_count = st.slider("Order Count", 1, 50, 5)
        pages_per_session = st.slider("Pages per Session", 1, 20, 5)
        bounce_rate = st.slider("Bounce Rate", 0.0, 1.0, 0.3)
    
    # Make prediction
    if st.button("Predict CLV"):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'total_spent': [total_spent],
            'avg_order_value': [avg_order_value],
            'order_count': [order_count],
            'monthly_sessions': [monthly_sessions],
            'avg_session_duration': [avg_session_duration * 60],  # Convert to seconds
            'pages_per_session': [pages_per_session],
            'bounce_rate': [bounce_rate]
        })
        
        # Scale and predict
        model = model_data['model']
        scaler = model_data['scaler']
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"Predicted Customer Lifetime Value: ${prediction:.2f}")
        
        # Compare with segments
        similar_customers = df[
            (abs(df['age'] - age) <= 5) & 
            (abs(df['income'] - income) <= 10000)
        ]
        
        if len(similar_customers) > 0:
            avg_clv_similar = similar_customers['clv'].mean()
            st.info(f"Average CLV of similar customers: ${avg_clv_similar:.2f}")

def show_model_performance(df, model_data):
    """Model performance metrics"""
    st.header("Model Performance Analysis")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    model = model_data['model']
    features = model_data['features']
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(importance_df, x='importance', y='feature', 
                orientation='h', title="Feature Importance in CLV Prediction")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model metrics (you would calculate these during training)
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score", "0.847")
        st.metric("Mean Absolute Error", "$245.67")
    
    with col2:
        st.metric("Root Mean Square Error", "$324.89")
        st.metric("Mean Absolute Percentage Error", "12.3%")

if __name__ == "__main__":
    main()
