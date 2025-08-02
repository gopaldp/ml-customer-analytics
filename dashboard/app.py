import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import streamlit as st

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('..')

# Import visualizations
try:
    from app import EnhancedVisualizations
    ENHANCED_VIZ_AVAILABLE = True
except ImportError:
    try:
        from visualization import EnhancedVisualizations
        ENHANCED_VIZ_AVAILABLE = True
    except ImportError:
        ENHANCED_VIZ_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Customer Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # Forces sidebar to be visible
)

# **CRITICAL: Data Availability Function**
@st.cache_data
def ensure_data_availability():
    """Generate sample data if no data files exist - fixes deployment issue"""
    if not os.path.exists('data/raw/customers.csv'):
        os.makedirs('data/raw', exist_ok=True)
        
        # Generate sample customers with German cities (including your location Weimar)
        sample_customers = pd.DataFrame({
            'customer_id': range(1, 201),
            'name': [f'Customer_{i}' for i in range(1, 201)],
            'age': np.random.randint(18, 80, 200),
            'income': np.random.randint(20000, 100000, 200),
            'city': np.random.choice(['Berlin', 'Munich', 'Hamburg', 'Weimar', 'Dresden', 'Frankfurt', 'Stuttgart'], 200),
            'latitude': 50.979492 + np.random.normal(0, 2, 200),  # Around Weimar coordinates
            'longitude': 11.323544 + np.random.normal(0, 3, 200),
            'clv': np.random.uniform(500, 2000, 200),
            'cluster': np.random.randint(0, 5, 200),
            'total_spent': np.random.uniform(100, 3000, 200),
            'avg_order_value': np.random.uniform(50, 500, 200),
            'order_count': np.random.randint(1, 20, 200),
            'monthly_sessions': np.random.randint(1, 30, 200),
            'avg_session_duration': np.random.uniform(60, 1800, 200),
            'pages_per_session': np.random.uniform(1, 10, 200),
            'bounce_rate': np.random.uniform(0.1, 0.9, 200)
        })
        
        sample_customers.to_csv('data/raw/customers.csv', index=False)
        
        # Generate sample relationships for network analysis
        sample_relationships = pd.DataFrame({
            'customer_1': np.random.choice(range(1, 201), 100),
            'customer_2': np.random.choice(range(1, 201), 100),
            'relationship_type': np.random.choice(['referral', 'family', 'colleague', 'neighbor', 'social_media'], 100),
            'strength': np.random.uniform(0.1, 1.0, 100),
            'created_date': pd.date_range('2023-01-01', '2024-12-31', periods=100)
        })
        
        sample_relationships.to_csv('data/raw/customer_relationships.csv', index=False)
        
        # Generate sample processed data (what your preprocessing would create)
        processed_customers = sample_customers.copy()
        processed_customers['customer_segment'] = processed_customers['cluster'].map({
            0: 'Low Value', 1: 'Medium Value', 2: 'High Value', 3: 'VIP', 4: 'Churned'
        })
        
        os.makedirs('data/processed', exist_ok=True)
        processed_customers.to_csv('data/processed/customer_features.csv', index=False)
        processed_customers.to_csv('data/processed/customer_features_clustered.csv', index=False)
    
    return True

# **CALL THIS FUNCTION IMMEDIATELY**
data_ready = ensure_data_availability()

# Data loading with fallback
@st.cache_data
def load_data():
    """Load data with comprehensive error handling and fallbacks"""
    customers_df = pd.DataFrame()
    relationships_df = pd.DataFrame()
    
    # Try multiple data sources in order of preference
    data_files = [
        'data/processed/customer_features_clustered.csv',
        'data/processed/customer_features.csv', 
        'data/raw/customers.csv'
    ]
    
    for file_path in data_files:
        try:
            if os.path.exists(file_path):
                customers_df = pd.read_csv(file_path)
                break
        except Exception:
            continue
    
    # Try to load relationships
    try:
        if os.path.exists('data/raw/customer_relationships.csv'):
            relationships_df = pd.read_csv('data/raw/customer_relationships.csv')
    except Exception:
        pass
    
    return customers_df, relationships_df

# Rest of your dashboard code continues here...

# Clean CSS styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
        background-color: #fafafa;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #3b82f6 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Force sidebar visibility */
    [data-testid="stSidebar"] {
        display: block !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Data loading without status messages
@st.cache_data
def load_data():
    customers_df = pd.DataFrame()
    relationships_df = pd.DataFrame()
    
    data_files = [
        'data/processed/customer_features_clustered.csv',
        'data/processed/customer_features.csv', 
        'data/raw/customers.csv'
    ]
    
    for file_path in data_files:
        try:
            if os.path.exists(file_path):
                customers_df = pd.read_csv(file_path)
                break
        except Exception:
            continue
    
    # Try to load relationships
    try:
        if os.path.exists('data/raw/customer_relationships.csv'):
            relationships_df = pd.read_csv('data/raw/customer_relationships.csv')
    except Exception:
        pass
    
    return customers_df, relationships_df

def create_kpi_section(df):
    """Clean KPI metrics without messages"""
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        if 'city' in df.columns:
            st.metric("Active Cities", f"{df['city'].nunique()}")
        else:
            st.metric("Data Points", f"{len(df):,}")
    
    with col3:
        value_col = 'clv' if 'clv' in df.columns else 'income'
        if value_col in df.columns:
            st.metric(f"Avg {value_col.upper()}", f"${df[value_col].mean():,.0f}")
        else:
            st.metric("Avg Age", f"{df['age'].mean():.1f}")
    
    with col4:
        if 'total_spent' in df.columns:
            st.metric("Total Revenue", f"${df['total_spent'].sum():,.0f}")
        else:
            st.metric("Features", len(df.columns))

def show_geographic_analysis(enhanced_viz, df):
    """Geographic analysis without status messages"""
    st.markdown("## 🌍 Geographic Distribution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'latitude' in df.columns:
            st.metric("Geographic Spread", f"{df['latitude'].std():.2f}°")
        else:
            st.metric("Customers", f"{len(df):,}")
    
    with col2:
        if 'city' in df.columns and len(df) > 0:
            top_city = df['city'].value_counts().index[0]
            top_city_count = df['city'].value_counts().iloc[0]
            st.metric("Top City", top_city, delta=f"{top_city_count} customers")
        else:
            st.metric("Coverage", "National")
    
    with col3:
        value_col = 'clv' if 'clv' in df.columns else 'income'
        if 'city' in df.columns and value_col in df.columns:
            city_avg = df.groupby('city')[value_col].mean().max()
            st.metric("Highest City Value", f"${city_avg:.0f}")
        else:
            st.metric("Coverage", "Available")
    
    with col4:
        if 'latitude' in df.columns and 'longitude' in df.columns:
            has_coords = len(df.dropna(subset=['latitude', 'longitude']))
            st.metric("Geo-Located", f"{has_coords:,}")
        else:
            st.metric("Mapping", "Ready")
    
    # Map section
    try:
        geo_map = enhanced_viz.create_geographic_map()
        if geo_map:
            st.markdown("#### 🗺️ Interactive Customer Map")
            try:
                from streamlit_folium import st_folium
                st_folium(geo_map, width=None, height=500)
            except ImportError:
                pass
    except Exception:
        pass

def show_analytics_insights(enhanced_viz, df):
    """Analytics with RESPONSIVE controls"""
    st.markdown("## 📊 Customer Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("#### 🎛️ Analysis Controls")
        
        # RESPONSIVE CONTROLS with session state
        if 'selected_cities' not in st.session_state:
            st.session_state.selected_cities = []
        if 'value_range' not in st.session_state:
            st.session_state.value_range = None
        
        # City filter with callback
        if 'city' in df.columns:
            cities = sorted(df['city'].unique())
            selected_cities = st.multiselect(
                "Filter by Cities",
                options=cities,
                default=cities[:5],
                key="city_filter"
            )
            st.session_state.selected_cities = selected_cities
        else:
            selected_cities = []
        
        # Value range filter with callback
        value_col = 'clv' if 'clv' in df.columns else 'income'
        if value_col in df.columns:
            min_val, max_val = float(df[value_col].min()), float(df[value_col].max())
            value_range = st.slider(
                f"{value_col.upper()} Range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key="value_filter"
            )
            st.session_state.value_range = value_range
        else:
            value_range = None
        
        # Apply filters to dataframe IN REAL-TIME
        filtered_viz_df = df.copy()
        
        if selected_cities and 'city' in filtered_viz_df.columns:
            filtered_viz_df = filtered_viz_df[filtered_viz_df['city'].isin(selected_cities)]
        
        if value_range and value_col in filtered_viz_df.columns:
            filtered_viz_df = filtered_viz_df[
                filtered_viz_df[value_col].between(value_range[0], value_range[1])
            ]
        
        # Update enhanced viz with filtered data
        try:
            filtered_enhanced_viz = EnhancedVisualizations(filtered_viz_df, enhanced_viz.relationships)
        except:
            filtered_enhanced_viz = enhanced_viz
        
        # Show filter results
        if len(filtered_viz_df) != len(df):
            st.info(f"**Filtered:** {len(filtered_viz_df):,} of {len(df):,} customers")
        
        # Key insights that update with filters
        st.markdown("#### 🔍 Key Insights")
        
        if 'cluster' in filtered_viz_df.columns:
            n_clusters = filtered_viz_df['cluster'].nunique()
            st.write(f"**{n_clusters} customer segments** in selection")
        
        if 'city' in filtered_viz_df.columns:
            n_cities = filtered_viz_df['city'].nunique()
            st.write(f"**{n_cities} cities** in selection")
        
        if value_col in filtered_viz_df.columns and len(filtered_viz_df) > 0:
            high_value_pct = (filtered_viz_df[value_col] > filtered_viz_df[value_col].quantile(0.8)).mean() * 100
            st.write(f"**{high_value_pct:.1f}%** are high-value customers")
    
    with col1:
        # 3D visualization that updates with filters
        try:
            fig_3d = filtered_enhanced_viz.create_3d_customer_scatter()
            if fig_3d:
                st.markdown("#### 🎯 3D Customer Segmentation")
                st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': False})
            else:
                fallback_fig = filtered_enhanced_viz.create_simple_metrics_chart()
                if fallback_fig:
                    st.markdown("#### 📈 Customer Demographics")
                    st.plotly_chart(fallback_fig, use_container_width=True, config={'displayModeBar': False})
        except Exception as e:
            st.error(f"Visualization error: {e}")

def show_network_relationships(enhanced_viz, df):
    """Network analysis without status messages"""
    st.markdown("## 🔗 Customer Network Analysis")
    
    try:
        network_fig = enhanced_viz.create_customer_network()
        if network_fig:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.plotly_chart(network_fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                st.markdown("#### 📈 Network Metrics")
                
                if not enhanced_viz.relationships.empty:
                    total_relationships = len(enhanced_viz.relationships)
                    avg_strength = enhanced_viz.relationships['strength'].mean()
                    relationship_types = enhanced_viz.relationships['relationship_type'].nunique()
                    
                    st.metric("Total Connections", f"{total_relationships:,}")
                    st.metric("Avg Strength", f"{avg_strength:.2f}")
                    st.metric("Relationship Types", relationship_types)
                    
                    st.markdown("#### 🏷️ Connection Types")
                    rel_types = enhanced_viz.relationships['relationship_type'].value_counts()
                    for rel_type, count in rel_types.head(5).items():
                        st.write(f"• **{rel_type.title()}:** {count}")
    except Exception:
        pass

def show_city_insights(enhanced_viz, df):
    """City insights without status messages"""
    st.markdown("## 🏙️ Geographic Insights")
    
    if 'city' not in df.columns:
        return
    
    try:
        city_fig = enhanced_viz.create_city_comparison_chart()
        if city_fig:
            st.plotly_chart(city_fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("### 🏆 City Performance Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👥 **Top Cities by Customer Count**")
            city_counts = df['city'].value_counts().head(8)
            
            city_ranking_df = pd.DataFrame({
                'Rank': range(1, len(city_counts) + 1),
                'City': city_counts.index,
                'Customers': city_counts.values,
                'Share': (city_counts.values / len(df) * 100).round(1)
            })
            
            st.dataframe(city_ranking_df, hide_index=True)
        
        with col2:
            value_col = 'clv' if 'clv' in df.columns else 'income'
            if value_col in df.columns:
                st.markdown(f"#### 💰 **Top Cities by Avg {value_col.upper()}**")
                city_values = df.groupby('city')[value_col].mean().sort_values(ascending=False).head(8)
                
                city_value_df = pd.DataFrame({
                    'Rank': range(1, len(city_values) + 1),
                    'City': city_values.index,
                    f'Avg {value_col.upper()}': city_values.values.round(0),
                    'Customers': [df[df['city'] == city].shape[0] for city in city_values.index]
                })
                
                st.dataframe(city_value_df, hide_index=True)
    except Exception:
        pass

def main():
    """Main dashboard without status messages"""
    
    st.markdown("# Customer Analytics Hub")
    st.markdown("*Advanced machine learning insights for customer intelligence*")
    
    # Load data silently
    customers_df, relationships_df = load_data()
    
    # Only show error if no data
    if customers_df.empty:
        st.error("⚠️ **No customer data found**")
        
        with st.container():
            st.markdown("""
            ### 🚀 **Quick Start Guide**
            
            1. **Generate Data:** Run `python src/data_generation.py`
            2. **Process Data:** Run `python src/data_preprocessing.py` 
            3. **Train Models:** Run `python src/models.py`
            4. **Refresh Page:** Click refresh to load your data
            """)
        return
    
    # Initialize visualizations silently
    if ENHANCED_VIZ_AVAILABLE:
        try:
            enhanced_viz = EnhancedVisualizations(customers_df, relationships_df)
        except Exception:
            return
    else:
        return
    
    # Sidebar for filtering (without messages)
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        
        if 'city' in customers_df.columns:
            st.markdown("### 📍 Geographic Filter")
            all_cities = sorted(customers_df['city'].unique())
            default_cities = all_cities[:min(5, len(all_cities))]
            
            cities = st.multiselect(
                "Select Cities",
                options=all_cities,
                default=default_cities
            )
            
            if cities:
                filtered_df = customers_df[customers_df['city'].isin(cities)]
            else:
                filtered_df = customers_df
        else:
            filtered_df = customers_df
        
        # Value range filter
        value_col = 'clv' if 'clv' in filtered_df.columns else 'income'
        if value_col in filtered_df.columns and len(filtered_df) > 0:
            st.markdown(f"### 💰 {value_col.upper()} Range")
            min_val, max_val = float(filtered_df[value_col].min()), float(filtered_df[value_col].max())
            if max_val > min_val:
                value_range = st.slider(
                    "Filter by value range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filtered_df = filtered_df[
                    filtered_df[value_col].between(value_range[0], value_range[1])
                ]
    
    # Update visualizations with filtered data
    try:
        enhanced_viz = EnhancedVisualizations(filtered_df, relationships_df)
    except Exception:
        enhanced_viz = EnhancedVisualizations(customers_df, relationships_df)
        filtered_df = customers_df
    
    # KPI section
    create_kpi_section(filtered_df)
    st.markdown("---")
    
    # Clean tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 **Geographic**", 
        "📊 **Analytics**", 
        "🔗 **Network**",
        "🏙️ **Cities**"
    ])
    
    with tab1:
        show_geographic_analysis(enhanced_viz, filtered_df)
    
    with tab2:
        show_analytics_insights(enhanced_viz, filtered_df)
    
    with tab3:
        show_network_relationships(enhanced_viz, filtered_df)
        
    with tab4:
        show_city_insights(enhanced_viz, filtered_df)
    
    # Clean footer
    st.markdown("---")
    st.markdown(
        f"*Customer Analytics Hub • {len(filtered_df):,} customers analyzed • "
        f"Updated {pd.Timestamp.now().strftime('%H:%M')}*"
    )

if __name__ == "__main__":
    main()
