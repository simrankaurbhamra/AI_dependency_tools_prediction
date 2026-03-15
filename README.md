# AI Overdependence Predictor

This package contains a Streamlit app and pre-trained XGBoost model to predict **overdependence on AI** (0-100%).

## Contents
- `streamlit_app.py` — main Streamlit app.
- `models/xgb_over_model_noleak_v2.pkl` — trained XGBoost regressor (no-leak, uses encoded demographics).
- `models/scaler_noleak_v2.pkl` — StandardScaler to preprocess inputs.
- `data/features_header_noleak_v2.csv` — feature names and order expected by the model.
- `data/dataset_scores_for_app.csv` — computed scores used in the visualization tab.
- `data/training_report_noleak_v2.json` — training R², RMSE and hyperparameters.
- `requirements.txt` — Python package requirements.

## How to run
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Notes
- The model predicts a percentage; interpretation bins are Low (0-33.33), Medium (33.33-66.66), High (66.66-100).
- The model R² on held-out test set: 0.764
