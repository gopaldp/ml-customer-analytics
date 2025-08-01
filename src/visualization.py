import pandas as pd
import numpy as np
import folium
from folium import plugins
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class EnhancedVisualizations:
    def __init__(self, customer_df, relationships_df=None):
        self.customers = customer_df.copy() if not customer_df.empty else pd.DataFrame()
        self.relationships = relationships_df if relationships_df is not None else pd.DataFrame()
        self.validate_data_for_plotting()
    
    def validate_data_for_plotting(self):
        """Validate and clean data before plotting"""
        if self.customers.empty:
            return
            
        # Ensure numeric columns are properly typed
        numeric_columns = ['clv', 'total_spent', 'order_count', 'monthly_sessions', 'age', 'income']
        
        for col in numeric_columns:
            if col in self.customers.columns:
                self.customers[col] = pd.to_numeric(self.customers[col], errors='coerce').fillna(0)
        
        # Ensure cluster column exists and is numeric
        if 'cluster' not in self.customers.columns:
            if 'city' in self.customers.columns:
                unique_cities = self.customers['city'].unique()
                city_to_cluster = {city: i % 5 for i, city in enumerate(unique_cities)}
                self.customers['cluster'] = self.customers['city'].map(city_to_cluster)
            else:
                self.customers['cluster'] = 0
        else:
            self.customers['cluster'] = pd.to_numeric(self.customers['cluster'], errors='coerce').fillna(0)
        
        # Clean coordinate data
        if 'latitude' in self.customers.columns and 'longitude' in self.customers.columns:
            self.customers = self.customers.dropna(subset=['latitude', 'longitude'])
            self.customers = self.customers[
                (self.customers['latitude'].between(-90, 90)) & 
                (self.customers['longitude'].between(-180, 180))
            ]

    def get_safe_layout(self, title="Chart", height=600):
        """Return a safe, validated layout configuration"""
        return {
            'title': title,
            'height': height,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
            'showlegend': False,
            'hovermode': 'closest',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        }

    def get_safe_3d_layout(self, title="3D Chart", height=600):
        """Return a safe 3D layout configuration"""
        return {
            'title': title,
            'height': height,
            'margin': {'l': 0, 'r': 0, 't': 40, 'b': 0},
            'showlegend': False,
            'scene': {
                'xaxis': {'title': 'X Axis'},
                'yaxis': {'title': 'Y Axis'},
                'zaxis': {'title': 'Z Axis'}
            }
        }

    def create_geographic_map(self):
        """Create interactive geographic customer distribution map"""
        if self.customers.empty or 'latitude' not in self.customers.columns or 'longitude' not in self.customers.columns:
            return None
            
        center_lat = self.customers['latitude'].mean()
        center_lon = self.customers['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        clv_col = 'clv' if 'clv' in self.customers.columns else 'income'
        clv_values = self.customers[clv_col]
        
        if len(clv_values) > 0:
            clv_min, clv_max = clv_values.min(), clv_values.max()
            
            for _, customer in self.customers.head(100).iterrows():  # Limit for performance
                clv = customer[clv_col]
                
                if clv_max > clv_min:
                    color_intensity = (clv - clv_min) / (clv_max - clv_min)
                else:
                    color_intensity = 0.5
                    
                if color_intensity < 0.3:
                    color = 'blue'
                elif color_intensity < 0.6:
                    color = 'orange'
                else:
                    color = 'red'
                
                folium.CircleMarker(
                    location=[customer['latitude'], customer['longitude']],
                    radius=8,
                    popup=f"<b>{customer['name']}</b><br>City: {customer['city']}<br>Value: ${clv:.2f}",
                    color=color,
                    fill=True,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
        
        return m

    def create_3d_customer_scatter(self):
        """Create 3D scatter plot with safe layout"""
        if self.customers.empty or len(self.customers) < 3:
            return None
            
        # Select appropriate columns with fallbacks
        x_col = 'total_spent' if 'total_spent' in self.customers.columns else 'income'
        y_col = 'order_count' if 'order_count' in self.customers.columns else 'age'
        z_col = 'monthly_sessions' if 'monthly_sessions' in self.customers.columns else 'clv'
        size_col = 'clv' if 'clv' in self.customers.columns else 'income'
        
        if not all(col in self.customers.columns for col in [x_col, y_col, z_col]):
            return None
        
        # Prepare data
        x_data = self.customers[x_col].values
        y_data = self.customers[y_col].values  
        z_data = self.customers[z_col].values
        color_data = self.customers['cluster'].astype(float).values
        size_data = self.customers[size_col].values
        
        # Normalize sizes
        if len(size_data) > 0 and size_data.max() > size_data.min():
            normalized_sizes = 5 + (size_data - size_data.min()) / (size_data.max() - size_data.min()) * 20
        else:
            normalized_sizes = [10] * len(size_data)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker={
                'size': normalized_sizes,
                'color': color_data,
                'colorscale': 'Viridis',
                'opacity': 0.8,
                'line': {'width': 1, 'color': 'black'}
            },
            text=[f"Customer: {name}<br>City: {city}" for name, city in zip(self.customers['name'], self.customers['city'])],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
        
        # Apply safe layout
        layout = self.get_safe_3d_layout("3D Customer Analysis")
        layout['scene']['xaxis']['title'] = x_col.replace('_', ' ').title()
        layout['scene']['yaxis']['title'] = y_col.replace('_', ' ').title()
        layout['scene']['zaxis']['title'] = z_col.replace('_', ' ').title()
        
        fig.update_layout(layout)
        
        return fig

    def create_customer_network(self, max_nodes=50):
        """Create customer relationship network with safe layout"""
        if self.relationships.empty or self.customers.empty:
            return None
        
        # Limit nodes for performance
        clv_col = 'clv' if 'clv' in self.customers.columns else 'income'
        top_customers = self.customers.nlargest(min(max_nodes, len(self.customers)), clv_col)
        
        filtered_relationships = self.relationships[
            (self.relationships['customer_1'].isin(top_customers['customer_id'])) &
            (self.relationships['customer_2'].isin(top_customers['customer_id']))
        ]
        
        if filtered_relationships.empty:
            return None
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for _, customer in top_customers.iterrows():
            G.add_node(
                customer['customer_id'],
                name=customer['name'],
                city=customer['city'],
                clv=float(customer.get(clv_col, 500)),
                cluster=float(customer.get('cluster', 0))
            )
        
        # Add edges
        for _, rel in filtered_relationships.iterrows():
            if G.has_node(rel['customer_1']) and G.has_node(rel['customer_2']):
                G.add_edge(rel['customer_1'], rel['customer_2'], weight=float(rel['strength']))
        
        if len(G.nodes()) == 0:
            return None
        
        # Calculate layout
        try:
            pos = nx.spring_layout(G, k=1, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line={'width': 0.5, 'color': 'rgba(125,125,125,0.5)'},
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            adjacencies = list(G.neighbors(node))
            node_info.append(f"Customer: {node_data['name']}<br>City: {node_data['city']}<br>CLV: ${node_data['clv']:.2f}")
            
            node_colors.append(float(node_data.get('cluster', 0)))
            
            clv_size = (node_data['clv'] / 1000 * 15) if node_data['clv'] > 0 else 5
            connection_size = len(adjacencies) * 3
            node_sizes.append(max(min(clv_size + connection_size + 10, 40), 5))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_info,
            marker={
                'showscale': True,
                'colorscale': 'Viridis',
                'color': node_colors,
                'size': node_sizes,
                'line': {'width': 2, 'color': 'black'}
            },
            showlegend=False
        )
        
        # Create figure with safe layout
        fig = go.Figure(data=[edge_trace, node_trace])
        
        layout = self.get_safe_layout("Customer Relationship Network")
        layout.update({
            'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
            'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False}
        })
        
        fig.update_layout(layout)
        
        return fig

    def create_city_comparison_chart(self):
        """Create city comparison with safe layout"""
        if self.customers.empty or 'city' not in self.customers.columns:
            return None
            
        clv_col = 'clv' if 'clv' in self.customers.columns else 'income'
        
        city_stats = self.customers.groupby('city').agg({
            'customer_id': 'count',
            clv_col: 'mean',
            'age': 'mean',
            'income': 'mean'
        }).reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Customer Count by City', f'Average {clv_col.upper()}', 
                          'Age vs Income', 'Income Distribution')
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=city_stats['city'], y=city_stats['customer_id'], 
                  name='Customers', marker_color='lightblue', showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=city_stats['city'], y=city_stats[clv_col],
                  name=f'Avg {clv_col.upper()}', marker_color='orange', showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=city_stats['age'], y=city_stats['income'],
                      mode='markers+text', text=city_stats['city'],
                      textposition='middle right',
                      marker={'size': 10, 'color': 'green'},
                      name='Cities', showlegend=False),
            row=2, col=1
        )
        
        top_cities = city_stats.nlargest(min(5, len(city_stats)), 'customer_id')
        fig.add_trace(
            go.Bar(x=top_cities['city'], y=top_cities['income'],
                  name='Avg Income', marker_color='red', showlegend=False),
            row=2, col=2
        )
        
        # Apply safe layout
        fig.update_layout(height=800, showlegend=False, title="City Analysis Dashboard")
        fig.update_xaxes(tickangle=45)
        
        return fig

    def create_simple_metrics_chart(self):
        """Create simple fallback visualization"""
        if self.customers.empty:
            return None
            
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Age Distribution', 'Income Distribution'))
        
        fig.add_trace(
            go.Histogram(x=self.customers['age'], name='Age', marker_color='blue', showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=self.customers['income'], name='Income', marker_color='green', showlegend=False),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title="Customer Demographics")
        return fig
