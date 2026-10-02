# Architecture & System Design

`ml-customer-analytics` is structured as a modular data science and web application pipeline. It ingests/generates synthetic customer records and transaction data, transforms and computes advanced features, trains machine learning models, and provides interactive exploration via a Streamlit dashboard.

## Component Overview

```
┌────────────────────────┐
│  src/data_generation.py│ ──► Generates synthetic customers, transactions,
└────────────────────────┘     web analytics, and network relationships.
            │
            ▼
┌─────────────────────────┐
│ src/data_preprocessing.py│ ──► Aggregates features, computes Customer Lifetime
└─────────────────────────┘     Value (CLV), and outputs processed datasets.
            │
            ▼
┌────────────────────────┐
│     src/models.py      │ ──► Trains K-means Clustering & Random Forest
└────────────────────────┘     CLV regression models; saves model artifacts.
            │
            ▼
┌────────────────────────┐
│    dashboard/app.py    │ ──► Streamlit Web Application for interactive analytics,
└────────────────────────┘     geographic maps, 3D clusters, and network graphs.
```

## Module Descriptions

- **`run_project.py`**: Pipeline orchestration script executing the data generation, preprocessing, model training, and dashboard launch sequence sequentially.
- **`src/data_generation.py`**: Uses `Faker` and `geopy` to generate realistic synthetic customer profiles across 16 German cities (including Weimar), transactions across 5 categories, web analytics metrics, and customer relationship networks.
- **`src/data_preprocessing.py`**: Implements the `DataPreprocessor` class to aggregate transaction histories, compute customer lifespans, merge web analytics data, and estimate baseline Customer Lifetime Value (CLV).
- **`src/models.py`**: Contains `CustomerSegmentation` (K-means clustering on spending, order counts, and session metrics) and `CLVPredictor` (Random Forest regression with feature scaling and evaluation metrics like MSE and $R^2$).
- **`src/visualization.py`** *(imported or embedded)*: Provides enhanced plotting capabilities using Plotly and Folium for geographic maps, 3D scatter plots, and network graphs.
- **`dashboard/app.py`**: A comprehensive Streamlit dashboard providing interactive filtering, multi-tab analytics, geographic distribution mapping, clustering insights, and deep learning / model evaluation views.

## Data & Control Flow

1. **Generation Phase**: Raw datasets (`customers.csv`, `transactions.csv`, `web_analytics.csv`, `customer_relationships.csv`) are written to `data/raw/`.
2. **Preprocessing Phase**: Raw records are joined and aggregated into `customer_features.csv` in `data/processed/`.
3. **Modeling Phase**: Processed features are segmented and used to train regression models, outputting clustered datasets and trained estimators (`src/clv_model.joblib`).
4. **Presentation Phase**: The Streamlit dashboard loads processed features and model outputs to render real-time interactive charts and spatial visualizations.
