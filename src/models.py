import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('src', exist_ok=True)  # For saving model files

class CustomerSegmentation:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
    
    def fit_transform(self, df):
        """Perform customer segmentation"""
        # Select features for clustering
        features = ['total_spent', 'avg_order_value', 'order_count', 
                   'monthly_sessions', 'avg_session_duration']
        
        X = df[features].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df['cluster'] = clusters
        
        # Create cluster profiles
        cluster_profiles = df.groupby('cluster')[features + ['clv']].mean().round(2)
        
        return df, cluster_profiles

class CLVPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def prepare_features(self, df):
        """Prepare features for CLV prediction"""
        features = ['age', 'income', 'total_spent', 'avg_order_value', 
                   'order_count', 'monthly_sessions', 'avg_session_duration',
                   'pages_per_session', 'bounce_rate']
        
        X = df[features].fillna(0)
        y = df['clv']
        
        self.feature_names = features
        return X, y
    
    def train(self, df):
        """Train the CLV prediction model"""
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X):
        """Make CLV predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.feature_names
        }, filepath)

# Usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/customer_features.csv')
    
    # Customer Segmentation
    segmenter = CustomerSegmentation()
    df_with_clusters, cluster_profiles = segmenter.fit_transform(df)
    
    print("Cluster Profiles:")
    print(cluster_profiles)
    
    # CLV Prediction
    clv_predictor = CLVPredictor()
    feature_importance = clv_predictor.train(df_with_clusters)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save results
    df_with_clusters.to_csv('data/processed/customer_features_clustered.csv', index=False)
    clv_predictor.save_model('src/clv_model.joblib')
