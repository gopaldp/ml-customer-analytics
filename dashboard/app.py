import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('..')

# Import visualizations
try:
    from visualization import EnhancedVisualizations
    ENHANCED_VIZ_AVAILABLE = True
except ImportError:
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
    initial_sidebar_state="expanded"
)

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
    
    [data-testid="stSidebar"] {
        display: block !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# **COMPLETE DATA LOADING FUNCTION**
@st.cache_data
def ensure_data_availability():
    """
    Complete data loading function that mimics local behavior
    - First tries to load existing data (like locally)
    - Falls back to generating sample data if none exists
    - Maintains compatibility with your Enhanced Visualizations
    """
    
    # Try to load existing data files first (prioritize processed data)
    data_sources = [
        'data/processed/customer_features_clustered.csv',
        'data/processed/customer_features.csv',
        'data/raw/customers.csv'
    ]
    
    customers_df = pd.DataFrame()
    relationships_df = pd.DataFrame()
    
    # Attempt to load existing data
    for file_path in data_sources:
        if os.path.exists(file_path):
            try:
                customers_df = pd.read_csv(file_path)
                break
            except Exception as e:
                continue
    
    # Try to load relationships if customers data was found
    if not customers_df.empty:
        try:
            if os.path.exists('data/raw/customer_relationships.csv'):
                relationships_df = pd.read_csv('data/raw/customer_relationships.csv')
        except Exception:
            pass
        
        return customers_df, relationships_df
    
    # If no existing data found, generate sample data (deployment fallback)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Set random seed for consistent data generation
    np.random.seed(42)
    
    # Sample size (smaller for deployment performance)
    sample_size = 250
    
    # German cities including your location (Weimar, Thüringen)
    german_cities = [
        'Berlin', 'Munich', 'Hamburg', 'Weimar', 'Dresden', 'Frankfurt', 
        'Stuttgart', 'Düsseldorf', 'Dortmund', 'Essen', 'Leipzig', 'Bremen',
        'Hanover', 'Nuremberg', 'Cologne', 'Mannheim'
    ]
    
    # Generate comprehensive customer data
    customers_data = {
        'customer_id': range(1, sample_size + 1),
        'name': [f'Customer_{i:03d}' for i in range(1, sample_size + 1)],
        'age': np.random.randint(18, 80, sample_size),
        'gender': np.random.choice(['M', 'F'], sample_size),
        'income': np.random.lognormal(10.5, 0.5, sample_size),
        'city': np.random.choice(german_cities, sample_size),
        'country': ['Germany'] * sample_size,
        'registration_date': pd.date_range('2022-01-01', '2024-12-01', periods=sample_size).astype(str)
    }
    
    # Add geographic coordinates (centered around German locations)
    city_coords = {
        'Berlin': (52.5200, 13.4050),
        'Munich': (48.1351, 11.5820),
        'Hamburg': (53.5511, 9.9937),
        'Weimar': (50.9793, 11.3235),  # Your location
        'Dresden': (51.0504, 13.7373),
        'Frankfurt': (50.1109, 8.6821),
        'Stuttgart': (48.7758, 9.1829),
        'Düsseldorf': (51.2277, 6.7735),
        'Dortmund': (51.5136, 7.4653),
        'Essen': (51.4556, 7.0116),
        'Leipzig': (51.3397, 12.3731),
        'Bremen': (53.0793, 8.8017),
        'Hanover': (52.3759, 9.7320),
        'Nuremberg': (49.4521, 11.0767),
        'Cologne': (50.9375, 6.9603),
        'Mannheim': (49.4875, 8.4660)
    }
    
    # Assign coordinates with some random variation
    latitudes = []
    longitudes = []
    
    for city in customers_data['city']:
        base_lat, base_lon = city_coords.get(city, (51.1657, 10.4515))  # Default to German center
        latitudes.append(base_lat + np.random.normal(0, 0.1))
        longitudes.append(base_lon + np.random.normal(0, 0.1))
    
    customers_data['latitude'] = latitudes
    customers_data['longitude'] = longitudes
    
    # Generate transaction-based features
    customers_data['total_spent'] = np.random.lognormal(7, 1, sample_size)
    customers_data['avg_order_value'] = customers_data['total_spent'] / np.random.randint(1, 20, sample_size)
    customers_data['order_count'] = (customers_data['total_spent'] / customers_data['avg_order_value']).astype(int)
    
    # Web analytics features
    customers_data['monthly_sessions'] = np.random.poisson(15, sample_size)
    customers_data['avg_session_duration'] = np.random.exponential(300, sample_size)
    customers_data['pages_per_session'] = np.random.gamma(2, 2, sample_size)
    customers_data['bounce_rate'] = np.random.beta(2, 3, sample_size)
    
    # Customer Lifetime Value (CLV)
    customers_data['clv'] = (
        customers_data['total_spent'] * 1.2 + 
        customers_data['monthly_sessions'] * 10 + 
        np.random.normal(0, 100, sample_size)
    )
    
    # Customer segmentation (5 clusters like your local setup)
    customers_data['cluster'] = np.random.randint(0, 5, sample_size)
    
    # Create DataFrame
    customers_df = pd.DataFrame(customers_data)
    
    # Save customer data to files (multiple formats for compatibility)
    customers_df.to_csv('data/raw/customers.csv', index=False)
    customers_df.to_csv('data/processed/customer_features.csv', index=False)
    customers_df.to_csv('data/processed/customer_features_clustered.csv', index=False)
    
    # Generate customer relationships for network analysis
    n_relationships = min(150, sample_size // 2)  # Reasonable number of relationships
    
    relationships_data = {
        'customer_1': np.random.choice(range(1, sample_size + 1), n_relationships),
        'customer_2': np.random.choice(range(1, sample_size + 1), n_relationships),
        'relationship_type': np.random.choice([
            'referral', 'family', 'colleague', 'neighbor', 'social_media'
        ], n_relationships, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'strength': np.random.uniform(0.1, 1.0, n_relationships),
        'created_date': pd.date_range('2023-01-01', '2024-12-01', periods=n_relationships).astype(str)
    }
    
    # Remove self-relationships
    relationships_df = pd.DataFrame(relationships_data)
    relationships_df = relationships_df[relationships_df['customer_1'] != relationships_df['customer_2']]
    relationships_df = relationships_df.drop_duplicates(subset=['customer_1', 'customer_2'])
    
    # Save relationships
    relationships_df.to_csv('data/raw/customer_relationships.csv', index=False)
    
    return customers_df, relationships_df

# **LOAD DATA AT STARTUP**
try:
    customers_df, relationships_df = ensure_data_availability()
    DATA_LOADED = True
except Exception as e:
    st.error(f"Critical error loading data: {e}")
    DATA_LOADED = False
    customers_df = pd.DataFrame()
    relationships_df = pd.DataFrame()

# **DATA LOADING WITH FALLBACK**
@st.cache_data
def load_data():
    """Simple data loader that uses pre-loaded data"""
    return customers_df, relationships_df

def create_kpi_section(df):
    """Create clean KPI metrics"""
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
    """Geographic analysis section"""
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
                st.info("Install streamlit-folium for interactive maps: `pip install streamlit-folium`")
    except Exception as e:
        st.warning(f"Map visualization not available: {e}")

def show_analytics_insights(enhanced_viz, df):
    """Analytics with responsive controls"""
    st.markdown("## 📊 Customer Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("#### 🎛️ Analysis Controls")
        
        # Responsive controls with session state
        if 'city' in df.columns:
            cities = sorted(df['city'].unique())
            selected_cities = st.multiselect(
                "Filter by Cities",
                options=cities,
                default=cities[:5],
                key="city_filter"
            )
        else:
            selected_cities = []
        
        # Value range filter
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
        else:
            value_range = None
        
        # Apply filters in real-time
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
        
        # Key insights
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
    """Network analysis section"""
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
        else:
            st.info("Network analysis requires relationship data. Generate data locally for full network features.")
    except Exception as e:
        st.warning(f"Network analysis not available: {e}")

def show_city_insights(enhanced_viz, df):
    """City insights section"""
    st.markdown("## 🏙️ Geographic Insights")
    
    if 'city' not in df.columns:
        st.warning("City data not available for geographic analysis")
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
    except Exception as e:
        st.error(f"City analysis error: {e}")

def main():
    """Main dashboard application"""
    
    st.markdown("# Customer Analytics Hub")
    st.markdown("*Advanced machine learning insights for customer intelligence*")
    
    # Check if data loaded successfully
    if not DATA_LOADED or customers_df.empty:
        st.error("⚠️ **No customer data found**")
        
        with st.container():
            st.markdown("""
            ### 🚀 **Quick Start Guide**
            
            1. **Generate Data:** Run `python src/data_generation.py`
            2. **Process Data:** Run `python src/data_preprocessing.py` 
            3. **Train Models:** Run `python src/models.py`
            4. **Refresh Page:** Click refresh to load your data
            
            *Or the system will generate sample data automatically on first run.*
            """)
        return
    
    # Load data using cached function
    filtered_customers_df, filtered_relationships_df = load_data()
    
    # Initialize visualizations
    if ENHANCED_VIZ_AVAILABLE:
        try:
            enhanced_viz = EnhancedVisualizations(filtered_customers_df, filtered_relationships_df)
        except Exception as e:
            st.error(f"Visualization initialization error: {e}")
            return
    else:
        st.error("Enhanced visualizations not available - check installation")
        return
    
    # Sidebar for global filtering
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        
        if 'city' in filtered_customers_df.columns:
            st.markdown("### 📍 Geographic Filter")
            all_cities = sorted(filtered_customers_df['city'].unique())
            default_cities = all_cities[:min(5, len(all_cities))]
            
            cities = st.multiselect(
                "Select Cities",
                options=all_cities,
                default=default_cities
            )
            
            if cities:
                filtered_customers_df = filtered_customers_df[filtered_customers_df['city'].isin(cities)]
            
        # Value range filter
        value_col = 'clv' if 'clv' in filtered_customers_df.columns else 'income'
        if value_col in filtered_customers_df.columns and len(filtered_customers_df) > 0:
            st.markdown(f"### 💰 {value_col.upper()} Range")
            min_val, max_val = float(filtered_customers_df[value_col].min()), float(filtered_customers_df[value_col].max())
            if max_val > min_val:
                value_range = st.slider(
                    "Filter by value range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filtered_customers_df = filtered_customers_df[
                    filtered_customers_df[value_col].between(value_range[0], value_range[1])
                ]
    
    # Update visualizations with filtered data
    try:
        enhanced_viz = EnhancedVisualizations(filtered_customers_df, filtered_relationships_df)
    except Exception:
        enhanced_viz = EnhancedVisualizations(customers_df, relationships_df)
        filtered_customers_df = customers_df
    
    # KPI section
    create_kpi_section(filtered_customers_df)
    st.markdown("---")
    
    # Tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 **Geographic**", 
        "📊 **Analytics**", 
        "🔗 **Network**",
        "🏙️ **Cities**"
    ])
    
    with tab1:
        show_geographic_analysis(enhanced_viz, filtered_customers_df)
    
    with tab2:
        show_analytics_insights(enhanced_viz, filtered_customers_df)
    
    with tab3:
        show_network_relationships(enhanced_viz, filtered_customers_df)
        
    with tab4:
        show_city_insights(enhanced_viz, filtered_customers_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"*Customer Analytics Hub • {len(filtered_customers_df):,} customers analyzed • "
        f"Updated {datetime.now().strftime('%H:%M')}*"
    )

if __name__ == "__main__":
    main()
