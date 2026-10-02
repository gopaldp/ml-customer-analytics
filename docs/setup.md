# Local Setup & Installation Guide

This guide covers setting up, running, and troubleshooting the `ml-customer-analytics` project locally or within a development container.

## Prerequisites

- Python 3.11 or higher
- pip and virtual environment tooling (`venv`)
- Git

## Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gopaldp/ml-customer-analytics.git
   cd ml-customer-analytics
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note on TensorFlow & Optional Dependencies:** `tensorflow` and `keras` are required because `dashboard/app.py` imports `tensorflow` at startup. `tensorflow` is large and can be slow to install, so ensure sufficient disk space and time. Other packages like `shap` and `mlxtend` are optional extensions.

## Running the Project

You can run the entire end-to-end pipeline (data generation, preprocessing, model training, and dashboard launch) using the orchestration script:

```bash
python run_project.py
```

This will automatically execute:
1. `src/data_generation.py`: Generates synthetic customer demographic records, transactions, web analytics, and relationship networks.
2. `src/data_preprocessing.py`: Aggregates customer features and calculates Customer Lifetime Value (CLV).
3. `src/models.py`: Trains K-Means clustering and Random Forest CLV prediction models, saving artifacts.
4. `dashboard/app.py`: Launches the Streamlit dashboard locally at `http://localhost:8501`.

### Running Components Individually

- **Generate Data:**
  ```bash
  python src/data_generation.py
  ```
- **Preprocess Data:**
  ```bash
  python src/data_preprocessing.py
  ```
- **Train Models:**
  ```bash
  python src/models.py
  ```
- **Launch Dashboard Directly:**
  ```bash
  streamlit run dashboard/app.py
  ```

## Development Container

The repository includes a dev container configuration (`.devcontainer/devcontainer.json`) configured for Python 3.11. Opening this repository in VS Code with the Dev Containers extension will automatically provision the environment and serve the app on port `8501`.

## Configuration

The project operates entirely on generated synthetic data and does not require external database connections or `.env` credential files for local testing.

## Troubleshooting

- **Geocoding Timeouts (`geopy`)**: `src/data_generation.py` uses `Nominatim` to lookup city coordinates for 16 German metropolitan areas. If network access is restricted or rate-limited, fallback coordinates for German metropolitan areas are used automatically.
- **TensorFlow Startup Import Error**: If TensorFlow is missing or failed to install, `dashboard/app.py` will fail at startup (`import tensorflow as tf`). Ensure all packages in `requirements.txt` are successfully installed via `pip install -r requirements.txt`.
- **Port Conflicts**: If port `8501` is already in use, run Streamlit on an alternative port:
  ```bash
  streamlit run dashboard/app.py --server.port 8502
  ```
