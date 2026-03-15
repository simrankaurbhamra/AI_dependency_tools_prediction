import joblib
import xgboost as xgb

# load your existing model
model = joblib.load("models/xgb_over_model_noleak_v2.pkl")

# save it in xgboost portable format
model.save_model("models/xgb_model.json")

print("Model converted successfully!")