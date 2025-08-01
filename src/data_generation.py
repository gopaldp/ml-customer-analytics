import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

# Ensure directories exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

fake = Faker()
np.random.seed(42)


def generate_customers(n_customers=10000):
    """Generate customer demographic data"""
    customers = []
    for i in range(n_customers):
        customer = {
            'customer_id': i + 1,
            'name': fake.name(),
            'email': fake.email(),
            'age': np.random.randint(18, 81),  # Random integers between 18-80
            'gender': np.random.choice(['M', 'F'], p=[0.5, 0.5]),
            'city': fake.city(),
            'country': fake.country(),
            'registration_date': fake.date_between(start_date='-2y', end_date='today'),
            'income': np.random.lognormal(10.5, 0.5)  # Log-normal distribution for income
        }
        customers.append(customer)

    return pd.DataFrame(customers)


def generate_transactions(customers_df, n_transactions=50000):
    """Generate transaction data"""
    transactions = []

    # Product categories and prices
    products = {
        'Electronics': [200, 1500],
        'Clothing': [20, 200],
        'Books': [5, 50],
        'Home': [50, 500],
        'Sports': [25, 300]
    }

    for _ in range(n_transactions):
        customer_id = np.random.choice(customers_df['customer_id'])
        category = np.random.choice(list(products.keys()))
        price_range = products[category]

        transaction = {
            'transaction_id': len(transactions) + 1,
            'customer_id': customer_id,
            'transaction_date': fake.date_between(start_date='-1y', end_date='today'),
            'product_category': category,
            'amount': np.random.uniform(price_range[0], price_range[1]),
            'quantity': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        }
        transactions.append(transaction)

    return pd.DataFrame(transactions)


def generate_web_analytics(customers_df):
    """Generate web analytics data"""
    analytics = []

    for customer_id in customers_df['customer_id']:
        # Some customers are more active than others
        activity_level = np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2])

        if activity_level == 'low':
            sessions = np.random.poisson(5)
        elif activity_level == 'medium':
            sessions = np.random.poisson(15)
        else:
            sessions = np.random.poisson(30)

        analytics.append({
            'customer_id': customer_id,
            'monthly_sessions': sessions,
            'avg_session_duration': np.random.exponential(300),  # seconds
            'pages_per_session': np.random.gamma(2, 2),
            'bounce_rate': np.random.beta(2, 3)
        })

    return pd.DataFrame(analytics)


# Generate all datasets
if __name__ == "__main__":
    customers = generate_customers(10000)
    transactions = generate_transactions(customers, 50000)
    web_analytics = generate_web_analytics(customers)

    # Save to CSV
    customers.to_csv('data/raw/customers.csv', index=False)
    transactions.to_csv('data/raw/transactions.csv', index=False)
    web_analytics.to_csv('data/raw/web_analytics.csv', index=False)

    print("Data generation completed!")
