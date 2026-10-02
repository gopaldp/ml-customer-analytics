# Architecture & System Design

`ml-customer-analytics` is structured as a modular data science and web application pipeline. It generates synthetic customer records, transactions, web analytics metrics, and customer relationship networks, transforms and computes advanced features and Customer Lifetime Value (CLV), trains machine learning models (K-Means clustering and Random Forest regression), and provides interactive visualization and deep learning model training via a Streamlit dashboard.

## Component Overview

```
┌────────────────────────┐
│  src/data_generation.py│ ──► Generates synthetic customers, transactions across
└────────────────────────┘     5 product categories, web analytics, and network relationships.
            │
            ▼
┌─────────────────────────┐
│ src/data_preprocessing.py│ ──► Aggregates transaction histories, computes customer
└─────────────────────────┘     lifespans, merges web logs, and calculates baseline CLV.
            │
            ▼
┌────────────────────────┐
│     src/models.py      │ ──► Trains K-Means Clustering (5 clusters) & Random Forest
└────────────────────────┘     CLV regression models; saves model artifacts to src/clv_model.joblib.
            │
            ▼
┌────────────────────────┐
│    dashboard/app.py    │ ──► Streamlit Web Application featuring 5 interactive tabs:
└────────────────────────┘     Geographic, Analytics, Network, Cities, and Deep Learning.
```

## Module Descriptions

- **`run_project.py`**: Pipeline orchestration script executing the data generation, preprocessing, model training, and dashboard launch sequence sequentially using `subprocess` and `sys.executable`.
- **`src/data_generation.py`**: Uses `Faker` (US and German locales) and `geopy` (Nominatim with coordinate fallbacks) to generate synthetic customer profiles across 16 German cities (Berlin, Hamburg, Munich, Cologne, Frankfurt, Stuttgart, Düsseldorf, Dortmund, Essen, Leipzig, Bremen, Dresden, Hanover, Nuremberg, Duisburg, and Weimar). Also generates transactions across 5 product categories (`Electronics`, `Clothing`, `Books`, `Home`, `Sports`), web analytics metrics (`monthly_sessions`, `avg_session_duration`, `pages_per_session`, `bounce_rate`), and customer relationship networks across 5 relationship types (`referral`, `family`, `colleague`, `neighbor`, `social_media`).
- **`src/data_preprocessing.py`**: Implements the `DataPreprocessor` class to aggregate transaction histories (total spent, average order value, order count, total quantity, first/last purchase dates, customer lifetime days), merge web analytics, handle missing values, and calculate Customer Lifetime Value (CLV).
- **`src/models.py`**: Contains `CustomerSegmentation` (K-Means clustering with $k=5$ on spending, order counts, and session metrics, scaling features with `StandardScaler`) and `CLVPredictor` (Random Forest regression with 100 estimators, feature scaling, evaluation metrics including MSE and $R^2$, feature importances, and joblib model serialization).
- **`src/visualization.py`**: Implements `EnhancedVisualizations`, providing robust plotting utilities via Plotly and Folium. It includes `create_geographic_map` (Folium circle markers with CLV color coding), `create_3d_customer_scatter` (3D scatter plots with cluster color scaling and dynamic marker sizing), `create_customer_network` (NetworkX spring layout graphs with relationship strength filtering), and `create_city_comparison_chart` (subplot analytics comparing customer counts, CLV, age, and income across cities).
- **`dashboard/app.py`**: A comprehensive Streamlit web application with state management, fallback synthetic data generation, global sidebar filtering (cities and value ranges), KPI summary metrics, and a 5-tab interface:
  1. **Geographic**: Interactive customer distribution map using `streamlit-folium`.
  2. **Analytics**: 3D customer segmentation scatter plot with responsive filters and real-time subset metrics.
  3. **Network**: Customer relationship network graph and connection metrics.
  4. **Cities**: City performance rankings by customer count and average CLV/income.
  5. **Deep Learning**: TensorFlow/Keras neural network training interface (Dense 64 -> Dense 32 -> Dense 1) predicting CLV with test evaluation metrics (MSE, $R^2$), actual vs. predicted scatter plots, and sample prediction tables.

## Data & Control Flow

1. **Generation Phase**: Raw datasets (`customers.csv`, `transactions.csv`, `web_analytics.csv`, `customer_relationships.csv`) are generated and stored in `data/raw/`.
2. **Preprocessing Phase**: Raw records are joined and aggregated into `customer_features.csv` in `data/processed/`.
3. **Modeling Phase**: Processed features are segmented and used to train regression models, outputting clustered datasets (`customer_features_clustered.csv`) and trained estimator artifacts (`src/clv_model.joblib`).
4. **Presentation Phase**: The Streamlit dashboard loads `data/processed/customer_features.csv` (falling back to `data/raw/customers.csv` or generating sample data if missing) and `data/raw/customer_relationships.csv`. It serves interactive visualizations and trains a neural network model directly in the Deep Learning tab.
