import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

fake = Faker(['en_US', 'de_DE'])  # US and German locale for variety
np.random.seed(42)

def get_coordinates(city, country='Germany'):
    """Get coordinates for a city"""
    geolocator = Nominatim(user_agent="ml_customer_analytics")
    try:
        location = geolocator.geocode(f"{city}, {country}")
        if location:
            return location.latitude, location.longitude
        else:
            # Default to German coordinates if not found
            return 51.1657 + np.random.normal(0, 2), 10.4515 + np.random.normal(0, 3)
    except (GeocoderTimedOut, GeocoderServiceError):
        # Return random German coordinates
        return 51.1657 + np.random.normal(0, 2), 10.4515 + np.random.normal(0, 3)

# German cities for more realistic data
GERMAN_CITIES = [
    'Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 'Stuttgart',
    'Düsseldorf', 'Dortmund', 'Essen', 'Leipzig', 'Bremen', 'Dresden',
    'Hanover', 'Nuremberg', 'Duisburg', 'Weimar'  # Added your location
]

def generate_customers(n_customers=1000):  # Reduced for faster geocoding
    """Generate customer demographic data with coordinates"""
    customers = []
    
    # Pre-calculate coordinates for German cities to avoid API limits
    city_coords = {}
    for city in GERMAN_CITIES:
        lat, lon = get_coordinates(city)
        city_coords[city] = (lat, lon)
    
    for i in range(n_customers):
        city = np.random.choice(GERMAN_CITIES)
        lat, lon = city_coords[city]
        
        customer = {
            'customer_id': i + 1,
            'name': fake.name(),
            'email': fake.email(),
            'age': max(18, int(np.random.normal(40, 15))),
            'gender': np.random.choice(['M', 'F'], p=[0.5, 0.5]),
            'city': city,
            'country': 'Germany',
            'latitude': lat + np.random.normal(0, 0.1),  # Add small random offset
            'longitude': lon + np.random.normal(0, 0.1),
            'registration_date': fake.date_between(start_date='-2y', end_date='today'),
            'income': np.random.lognormal(10.5, 0.5)
        }
        customers.append(customer)
    
    return pd.DataFrame(customers)

def generate_customer_relationships(customers_df, n_relationships=500):
    """Generate customer relationships for network analysis"""
    relationships = []
    customer_ids = customers_df['customer_id'].tolist()
    
    for _ in range(n_relationships):
        # Select two different customers
        customer_1, customer_2 = np.random.choice(customer_ids, 2, replace=False)
        
        # Define relationship types with different strengths
        relationship_type = np.random.choice([
            'referral', 'family', 'colleague', 'neighbor', 'social_media'
        ], p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        # Assign relationship strength
        strength_mapping = {
            'family': np.random.uniform(0.7, 1.0),
            'referral': np.random.uniform(0.5, 0.9),
            'colleague': np.random.uniform(0.3, 0.7),
            'neighbor': np.random.uniform(0.2, 0.6),
            'social_media': np.random.uniform(0.1, 0.4)
        }
        
        relationship = {
            'customer_1': customer_1,
            'customer_2': customer_2,
            'relationship_type': relationship_type,
            'strength': strength_mapping[relationship_type],
            'created_date': fake.date_between(start_date='-1y', end_date='today')
        }
        relationships.append(relationship)
    
    return pd.DataFrame(relationships)

# Generate datasets
if __name__ == "__main__":
    print("Generating customer data...")
    customers = generate_customers(1000)
    
    print("Generating transaction data...")
    # Use existing transaction generation function but with fewer customers
    def generate_transactions(customers_df, n_transactions=5000):
        transactions = []
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
    
    transactions = generate_transactions(customers, 5000)
    
    print("Generating web analytics...")
    # Use existing web analytics function
    def generate_web_analytics(customers_df):
        analytics = []
        for customer_id in customers_df['customer_id']:
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
                'avg_session_duration': np.random.exponential(300),
                'pages_per_session': np.random.gamma(2, 2),
                'bounce_rate': np.random.beta(2, 3)
            })
        
        return pd.DataFrame(analytics)
    
    web_analytics = generate_web_analytics(customers)
    
    print("Generating customer relationships...")
    relationships = generate_customer_relationships(customers)
    
    # Save all datasets
    customers.to_csv('data/raw/customers.csv', index=False)
    transactions.to_csv('data/raw/transactions.csv', index=False)
    web_analytics.to_csv('data/raw/web_analytics.csv', index=False)
    relationships.to_csv('data/raw/customer_relationships.csv', index=False)
    
    print("Enhanced data generation completed!")
    print(f"Generated {len(customers)} customers across {len(customers['city'].unique())} German cities")
    print(f"Generated {len(relationships)} customer relationships")
