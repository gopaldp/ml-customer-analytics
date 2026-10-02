# Copilot instructions

## Project

Customer analytics demo in Python 3.11. The pipeline runs in this order:

1. `src/data_generation.py` creates synthetic customer data.
2. `src/data_preprocessing.py` cleans it and builds features.
3. `src/models.py` trains the CLV prediction and K-means segmentation models.
4. `dashboard/app.py` is the Streamlit dashboard that shows the results.

`run_project.py` runs all four steps. Dependencies are in `requirements.txt`.

## Documentation notes

- Keep the live demo link in `README.md`.
- `notebooks/eda.ipynb` is exploratory analysis. Use it for context only.
- The dev container (`.devcontainer/devcontainer.json`) uses Python 3.11 and
  serves the app on port 8501.
- `tensorflow` in `requirements.txt` is large and slow to install. Mention it
  in setup troubleshooting.

## Conventions

- Keep changes small and focused. One concern per pull request.
- Base documentation on the actual code. Never invent features, metrics or
  commands. Mark anything uncertain with `TODO: confirm …`.
- Do not commit generated data (`data/raw`, `data/processed`), model files
  (`*.joblib`, `*.pkl`, `*.h5`) or virtual environments.
- Use UTF-8 for all text files.
