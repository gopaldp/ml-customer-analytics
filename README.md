# Enhanced Customer Analytics Dashboard

## 📊 Overview

A comprehensive machine learning-powered customer analytics dashboard built with Streamlit, featuring advanced data visualization, geographic mapping, and network analysis. This project demonstrates end-to-end data science capabilities from data generation to interactive web deployment.

**🌟 Live Demo:** [Customer Analytics Hub](https://ml-customer-analytics-aw7njacg33wprkegmsegjm.streamlit.app)

## ✨ Key Features

### 🌍 **Geographic Analysis**
- Interactive customer distribution maps with Folium and `streamlit-folium`
- City-based performance analytics across 16 German metropolitan areas (Berlin, Hamburg, Munich, Cologne, Frankfurt, Stuttgart, Düsseldorf, Dortmund, Essen, Leipzig, Bremen, Dresden, Hanover, Nuremberg, Duisburg, and Weimar)
- Geographic spread visualization centered around German locations

### 📊 **3D Customer Segmentation**
- Interactive 3D scatter plots (`plotly`) with ML-powered customer clustering (5 clusters)
- Real-time filtering and responsive controls in the Streamlit dashboard
- Multi-dimensional customer analysis (CLV, spending, order counts, monthly sessions)
- Rotate, zoom, and hover for detailed customer insights

### 🔗 **Network Analysis**
- Customer relationship network visualization using `NetworkX` and Plotly
- Connection strength analysis and relationship mapping across 5 relationship types: `referral`, `family`, `colleague`, `neighbor`, and `social_media`
- Interactive network exploration with node sizing by influence and CLV

### 🏙️ **City-Based Intelligence**
- Performance rankings by customer count and lifetime value across cities
- Geographic distribution analysis and comparison charts with demographic insights (age vs. income)
- Regional customer intelligence and market penetration metrics

### 🤖 **Machine Learning & Deep Learning Integration**
- **Clustering:** K-Means customer segmentation (`src/models.py`) identifying 5 distinct customer groups
- **Regression:** Customer Lifetime Value (CLV) prediction using a Random Forest model (`src/models.py`), outputting evaluation metrics (MSE and $R^2$) and feature importances
- **Deep Learning Tab:** Interactive TensorFlow/Keras neural network training directly in the Streamlit dashboard (`dashboard/app.py`), featuring a 3-layer architecture (Dense 64 -> Dense 32 -> Dense 1) trained on customer features to predict CLV with actual vs. predicted evaluation plots

## 🛠️ Tech Stack

- **Python 3.11**
- **Data Processing & ML:** Pandas, NumPy, Scikit-Learn (KMeans, RandomForestRegressor, StandardScaler), TensorFlow, Keras, SHAP (optional)
- **Visualization:** Plotly (Express & Graph Objects), Seaborn, Matplotlib, Folium, Streamlit-Folium
- **Network Analysis:** NetworkX
- **Data Generation & Utilities:** Faker (`en_US`, `de_DE`), Geopy (Nominatim geocoding with coordinate fallbacks), SQLAlchemy, Joblib

## ⚙️ Prerequisites & Installation

See [Setup Guide](docs/setup.md) for detailed setup instructions.

1. Clone the repository:
   ```bash
   git clone https://github.com/gopaldp/ml-customer-analytics.git
   cd ml-customer-analytics
   ```
2. Create virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   > **Note:** `tensorflow` is large and may take longer to install. It is required: `dashboard/app.py` imports it at startup.

## 🚀 Usage

Run the complete end-to-end pipeline (data generation, feature preprocessing, model training, and dashboard launch) using:

```bash
python run_project.py
```

Or run individual components:
- **Generate Data:** `python src/data_generation.py`
- **Preprocess Data:** `python src/data_preprocessing.py`
- **Train Models:** `python src/models.py`
- **Launch Dashboard:** `streamlit run dashboard/app.py`

## ⚙️ Configuration

The project runs out-of-the-box with default synthetic data generation. No external environment variables (`.env`) are required for standard local execution. If geocoding (`geopy`) encounters network limits during data generation, fallback coordinates for German metropolitan areas are used automatically.

## 📁 Project Structure

```
ml-customer-analytics/
├── .devcontainer/         # Dev container configuration (Python 3.11, port 8501)
├── dashboard/
│   └── app.py             # Streamlit web application (5 tabs: Geographic, Analytics, Network, Cities, Deep Learning)
├── data/                  # Generated raw and processed datasets (git ignored)
├── docs/
│   ├── architecture.md    # Architecture and data/control flow
│   └── setup.md           # Local setup, dev container configuration, and troubleshooting
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis notebook (for context)
├── src/
│   ├── data_generation.py # Synthetic customer, transaction, web analytics & network generator
│   ├── data_preprocessing.py# Feature aggregation and CLV computation
│   ├── models.py          # K-Means clustering and Random Forest CLV predictor
│   └── visualization.py   # Plotly & Folium visualization helpers (EnhancedVisualizations)
├── README.md              # Project overview and instructions
├── requirements.txt       # Python dependencies
└── run_project.py         # End-to-end pipeline orchestration script
```

## 🧪 Testing & Validation

The project includes modular Python scripts and automated pipeline orchestration (`run_project.py`). Model performance and metrics (such as MSE and $R^2$ scores) are printed during model training (`src/models.py` and within the Streamlit Deep Learning tab).

## 📚 Documentation

- [Architecture & System Design](docs/architecture.md)
- [Setup & Installation Guide](docs/setup.md)
