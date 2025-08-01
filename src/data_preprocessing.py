import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self):
        """Load all datasets"""
        self.customers = pd.read_csv('data/raw/customers.csv')
        self.transactions = pd.read_csv('data/raw/transactions.csv')
        self.web_analytics = pd.read_csv('data/raw/web_analytics.csv')

        # Convert dates
        self.customers['registration_date'] = pd.to_datetime(self.customers['registration_date'])
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])

        return self

    def create_customer_features(self):
        """Create aggregated customer features"""
        # Transaction aggregations
        trans_agg = self.transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'transaction_date': ['min', 'max']
        }).round(2)

        trans_agg.columns = ['total_spent', 'avg_order_value', 'order_count',
                             'total_quantity', 'first_purchase', 'last_purchase']

        # Calculate customer lifetime (days)
        trans_agg['customer_lifetime_days'] = (
                trans_agg['last_purchase'] - trans_agg['first_purchase']
        ).dt.days

        # Merge all data
        customer_features = self.customers.merge(trans_agg, on='customer_id', how='left')
        customer_features = customer_features.merge(self.web_analytics, on='customer_id', how='left')

        # Fill missing values
        numeric_columns = customer_features.select_dtypes(include=[np.number]).columns
        customer_features[numeric_columns] = customer_features[numeric_columns].fillna(0)

        return customer_features

    def calculate_clv(self, customer_features):
        """Calculate Customer Lifetime Value"""
        # Simple CLV calculation: (Average Order Value) × (Purchase Frequency) × (Customer Lifespan)
        customer_features['purchase_frequency'] = (
                customer_features['order_count'] /
                (customer_features['customer_lifetime_days'] / 365 + 1)  # Avoid division by zero
        )

        customer_features['clv'] = (
                customer_features['avg_order_value'] *
                customer_features['purchase_frequency'] *
                2  # Assume 2-year projection
        )

        return customer_features


# Usage
preprocessor = DataPreprocessor()
preprocessor.load_data()
customer_features = preprocessor.create_customer_features()
customer_features = preprocessor.calculate_clv(customer_features)
customer_features.to_csv('data/processed/customer_features.csv', index=False)
