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
   > **Note on TensorFlow:** `tensorflow` in `requirements.txt` is large and can be slow to install, so allow enough disk space and time. Do not skip it: `dashboard/app.py` imports TensorFlow at startup, so the dashboard will not run without it.

## Running the Project

You can run the entire end-to-end pipeline (data generation, preprocessing, model training, and dashboard launch) using the orchestration script:

```bash
python run_project.py
```

This will automatically:
1. Generate synthetic customer data, transactions, web logs, and relationship networks.
2. Preprocess and compute feature aggregations and CLV.
3. Train K-means clustering and Random Forest CLV prediction models.
4. Launch the Streamlit dashboard locally at `http://localhost:8501`.

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

## Troubleshooting

- **Geocoding Timeouts (`geopy`)**: `src/data_generation.py` uses `Nominatim` to lookup city coordinates. If network access is restricted or rate-limited, fallback coordinates for German metropolitan areas are used automatically.
- **Port Conflicts**: If port `8501` is already in use, run Streamlit on an alternative port:
  ```bash
  streamlit run dashboard/app.py --server.port 8502
  ```
