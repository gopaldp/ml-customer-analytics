# Enhanced Customer Analytics Dashboard

## 📊 Overview

A comprehensive machine learning-powered customer analytics dashboard built with Streamlit, featuring advanced data visualization, geographic mapping, and network analysis. This project demonstrates end-to-end data science capabilities from data generation to interactive web deployment.

**🌟 Live Demo:** [Customer Analytics Hub](https://ml-customer-analytics-aw7njacg33wprkegmsegjm.streamlit.app)

## ✨ Key Features

### 🌍 **Geographic Analysis**
- Interactive customer distribution maps with Folium
- City-based performance analytics across German metropolitan areas
- Geographic spread visualization centered around Weimar, Thüringen

### 📊 **3D Customer Segmentation**
- Interactive 3D scatter plots with ML-powered customer clustering
- Real-time filtering and responsive controls
- Multi-dimensional customer analysis (CLV, spending, behavior)
- Rotate, zoom, and hover for detailed customer insights

### 🔗 **Network Analysis**
- Customer relationship network visualization using NetworkX
- Connection strength analysis and relationship mapping
- Interactive network exploration with node sizing by influence
- Relationship type breakdowns (family, referral, colleague, neighbor, social media)

### 🏙️ **City-Based Intelligence**
- Performance rankings by customer count and lifetime value
- Geographic distribution analysis across 16+ German cities
- City comparison charts with demographic insights
- Regional customer intelligence and market penetration metrics

### 🤖 **Machine Learning Integration**
- Customer Lifetime Value (CLV) prediction with a Random Forest model (`src/models.py`, reports MSE and R²)
- Neural-network CLV model (TensorFlow/Keras) trained in the dashboard's Deep Learning tab, with predicted-vs-actual chart
- K-means customer segmentation (5 clusters)
- Feature importance analysis for business insights

## 🛠️ Tech Stack

- **Python 3.11**
- **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, TensorFlow, Keras, SHAP
- **Visualization:** Plotly, Seaborn, Matplotlib, Folium, Streamlit-Folium
- **Network Analysis:** NetworkX
- **Data Generation & Utilities:** Faker, Geopy, SQLAlchemy, Joblib
- **Frontend / Dashboard:** Streamlit

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

## 📁 Project Structure

```
ml-customer-analytics/
├── .devcontainer/         # Dev container configuration
├── dashboard/
│   └── app.py             # Streamlit dashboard application
├── data/                  # Generated raw and processed datasets (ignored in git)
├── docs/
│   ├── architecture.md    # Architecture and data/control flow
│   └── setup.md           # Local setup and troubleshooting
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis notebook
├── src/
│   ├── data_generation.py # Synthetic customer, transaction & network generator
│   ├── data_preprocessing.py# Feature aggregation and CLV computation
│   ├── models.py          # K-Means clustering and Random Forest CLV predictor
│   └── visualization.py   # Plotly & Folium visualization helpers
├── README.md              # Project overview and instructions
├── requirements.txt       # Python dependencies
└── run_project.py         # End-to-end pipeline orchestration script
```

## 🧪 Testing & Validation

The project includes modular Python scripts and automated pipeline orchestration. Model performance and metrics (such as MSE and $R^2$ scores) are printed during model training (`src/models.py`).

## 📚 Documentation

- [Architecture & System Design](docs/architecture.md)
- [Setup & Installation Guide](docs/setup.md)
