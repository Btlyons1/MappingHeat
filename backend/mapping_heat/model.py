import pickle
import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, Any, List

try: from lightgbm import LGBMClassifier
except ImportError: LGBMClassifier = Any
try: from sklearn.preprocessing import StandardScaler
except ImportError: StandardScaler = Any

from .db import get_db

MODEL_DIR = "../model/" 
SUFFIX = '2023-03-30_to_2024-9-29'
MODEL_FILENAME = os.path.join(MODEL_DIR, f"pitch_prob_model_{SUFFIX}.sav")
SCALER_FILENAME = os.path.join(MODEL_DIR, f"pitch_scaler_{SUFFIX}.sav")
SPLIT_DATA_FILENAME = os.path.join(MODEL_DIR, f"train_test_split_data_{SUFFIX}.npz")

# --- Feature lists ---
numerical_features = [
    'release_speed', 'balls', 'strikes', 'pfx_x', 'pfx_z', 'release_spin_rate', 'release_extension',
    'plate_x', 'plate_z', 'effective_speed', 'outs_when_up', 'inning', 'sz_top', 'sz_bot', 'spin_axis',
    'release_pos_x', 'release_pos_z', 'arm_angle', 'n_thruorder_pitcher'
]
categorical_features_initial = ['player_name', 'pitch_type', 'zone', 'stand', 'p_throws']
categorical_features_final = ['player_name', 'pitch_type', 'norm_zone', 'stand', 'p_throws']
runner_cols = ['on_1b', 'on_2b', 'on_3b']

cache: Dict[str, Any] = {}

def init_model() -> None:
    """Load all ML artifacts into the cache on application startup."""
    global cache
    print("--- Initializing v5 ML Model Artifacts ---")

    def load_artifact(filename: str, description: str) -> Any:
        print(f"Attempting to load {description} from: {filename}")
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f: return pickle.load(f)
            except Exception as e: print(f"❌ Error loading {description}: {e}")
        else: print(f"❌ {description} file not found.")
        return None

    cache['model'] = load_artifact(MODEL_FILENAME, "Model")
    cache['scaler'] = load_artifact(SCALER_FILENAME, "Scaler")

    if os.path.exists(SPLIT_DATA_FILENAME):
        try:
            with np.load(SPLIT_DATA_FILENAME, allow_pickle=True) as data:
                cache['feature_names'] = data['feature_names'].tolist()
        except Exception as e: print(f"❌ Error loading feature names: {e}")

    if cache.get('model') and cache.get('scaler') and cache.get('feature_names'):
        print("✅ All ML artifacts loaded successfully.")
    else:
        print("❌ CRITICAL: One or more ML artifacts failed to load.")

def predict_hit_prob(raw_input_data: Dict[str, Any]) -> Optional[float]:
    """Preprocesses a single pitch context and returns a hit probability."""
    try:
        model, scaler, feature_names_expected = cache.get('model'), cache.get('scaler'), cache.get('feature_names')
        if not all([model, scaler, feature_names_expected]):
            print("❌ Prediction failed: Model artifacts not loaded.")
            return None

        input_df = pd.DataFrame([raw_input_data])
        for col in numerical_features: input_df[col] = pd.to_numeric(input_df.get(col, 0.0), errors='coerce').fillna(0.0)
        for col in categorical_features_initial: input_df[col] = str(input_df.get(col, 'Unknown'))
        for col in runner_cols: input_df[col] = pd.to_numeric(input_df.get(col, 0.0), errors='coerce').fillna(0.0)

        input_df['norm_zone'] = input_df['zone'].astype(str)
        runner_flag_features = [f'runner_on_{i+1}b_flag' for i in range(len(runner_cols))]
        for flag_col, runner_col in zip(runner_flag_features, runner_cols):
            input_df[flag_col] = np.where(input_df[runner_col] != 0.0, 1, 0).astype(np.int8)

        final_numerical_features = numerical_features + runner_flag_features
        X_pd_df = input_df[final_numerical_features + categorical_features_final]
        X_pd_df_encoded = pd.get_dummies(X_pd_df, columns=categorical_features_final, dtype=np.int8)
        X_pd_df_aligned = X_pd_df_encoded.reindex(columns=feature_names_expected, fill_value=0)

        X_np = X_pd_df_aligned.to_numpy(dtype=np.float32)

        numerical_indices = [X_pd_df_aligned.columns.get_loc(col) for col in final_numerical_features if col in X_pd_df_aligned.columns]
        if numerical_indices:
            X_np[:, numerical_indices] = scaler.transform(X_np[:, numerical_indices])

        probabilities = model.predict_proba(X_np)
        return float(probabilities[0][1])
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR during prediction: {e}")
        return None