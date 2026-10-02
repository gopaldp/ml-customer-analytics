---
description: |
  Reviews the codebase and opens a draft PR that brings README.md and docs/
  in line with what the code actually does. Manual trigger only, to keep
  Copilot credit usage under control.

on:
  workflow_dispatch:

permissions:
  contents: read
  issues: read
  pull-requests: read

engine: copilot

network: defaults

tools:
  github:
    toolsets: [default]
  edit:
  bash: ["ls", "cat", "find", "grep", "head", "wc", "git log", "git diff", "python --version"]

safe-outputs:
  create-pull-request:
    title-prefix: "[docs] "
    labels: [documentation]
    draft: true

timeout-minutes: 15
---

# Update Docs

You are the documentation maintainer for `${{ github.repository }}`, a Python
customer-analytics project: synthetic data generation, preprocessing, ML models
(CLV prediction, K-means segmentation) and a Streamlit dashboard.

## Task

Read the source code and bring the documentation up to date. Then open one
draft pull request with your changes.

1. Read `README.md`, `run_project.py`, `requirements.txt`, everything in `src/`
   and `dashboard/app.py`. Skim `notebooks/eda.ipynb` only for context.
2. Update `README.md`:
   - Keep the accurate parts of the existing overview, feature list and live
     demo link.
   - Add or fix: prerequisites (Python 3.11), installation, how to run the full
     pipeline (`python run_project.py`), how to run each step on its own, and
     how to start only the dashboard.
   - Add a short project-structure section that matches the real files.
   - Fix any broken character encoding (for example mojibake emoji such as `ðŸ“Š`).
3. Create `docs/architecture.md` describing the pipeline stages, what each module
   in `src/` does, what files each stage reads and writes, and how the dashboard
   consumes them.
4. Create `docs/setup.md` covering local setup, the dev container, and common
   problems (for example the large TensorFlow dependency).

## Rules

- Change documentation only: `README.md` and files under `docs/`. Do not modify
  code, notebooks, requirements or configuration.
- Base every statement on the code. Do not invent features, metrics, commands
  or file names. If something is unclear, write `TODO: confirm …` instead of
  guessing.
- Claims such as accuracy figures must come from the code or be marked `TODO`.
- Use plain, concise English and Markdown.
- If the documentation is already accurate and complete, make no changes and
  do not open a pull request.

## Pull request

Title it with a short summary. In the body, list each file you changed and
every `TODO` you left, so the reviewer knows what to check.
