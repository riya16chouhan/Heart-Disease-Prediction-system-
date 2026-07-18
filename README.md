# Heart Disease Prediction — Streamlit App

This repository contains a Streamlit app (`app.py`) that predicts heart disease risk using a trained KNN model.

Files included:
- `app.py` — Streamlit app
- `train_model.py` — script to (re)train and create model artifacts
- `KNN_heart.pkl`, `scaler (5).pkl`, `columns.pkl` — model artifacts
- `requirements.txt` — Python dependencies
-'heartdiseases1.ipynb' - code file
Deploy to Streamlit Community Cloud
1. Push this repository to GitHub (if not already). Example:

```bash
git add .
git commit -m "Add Streamlit app and model artifacts"
git push origin main
```

2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select this repository, the `main` branch, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit will install `requirements.txt` and launch the app.

Notes
- Keep model artifacts (`*.pkl`) in the repo so the app can load them on deploy.
- If you prefer not to store large binaries in Git, upload them to cloud storage and modify `app.py` to download at startup.
