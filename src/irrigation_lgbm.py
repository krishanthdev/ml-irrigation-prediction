import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# ================================================================
# CONFIGURATION
# ================================================================
DATA_DIR    = r'd:\Studies\Epita University\3rd Semester\ML 1 - 2'
SEED        = 42
TARGET      = 'Irrigation_Need'

np.random.seed(SEED)

def timer(msg):
    class _Timer:
        def __init__(self, m): self.msg = m
        def __enter__(self):
            self.t0 = time.time()
            print(f'\n{"-"*55}')
            print(f'>> {self.msg}...')
            return self
        def __exit__(self, *_):
            self.elapsed = time.time() - self.t0
            print(f'[OK] Done in {self.elapsed:.1f}s')
    return _Timer(msg)

def engineer_features(df):
    df = df.copy()
    eps = 1e-6
    df['Moisture_x_Temp']        = df['Soil_Moisture'] * df['Temperature_C']
    df['Rainfall_x_Humidity']    = df['Rainfall_mm'] * df['Humidity']
    df['Temp_x_Humidity']        = df['Temperature_C'] * df['Humidity']
    df['Moisture_x_Rainfall']    = df['Soil_Moisture'] * df['Rainfall_mm']
    df['Irrigation_per_Hectare'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + eps)
    df['Rainfall_per_Sunlight']  = df['Rainfall_mm'] / (df['Sunlight_Hours'] + eps)
    df['Moisture_Deficit']       = df['Temperature_C'] - df['Soil_Moisture']
    return df

def main():
    print('=' * 55)
    print('  IRRIGATION NEED PREDICTION - LightGBM (Max Accuracy)')
    print('=' * 55)

    # ----------------------------------------------------------
    # 1. LOAD DATA
    # ----------------------------------------------------------
    with timer('Loading data'):
        train_path = os.path.join(DATA_DIR, 'train.csv')
        test_path  = os.path.join(DATA_DIR, 'test.csv')
        
        train_raw = pd.read_csv(train_path)
        test_raw  = pd.read_csv(test_path)
        print(f'  Train: {train_raw.shape}')
        print(f'  Test : {test_raw.shape}')

    # ----------------------------------------------------------
    # 2. FEATURE ENGINEERING & CATEGORICAL SETUP
    # ----------------------------------------------------------
    with timer('Feature Engineering & Categorical Conversion'):
        train_fe = engineer_features(train_raw)
        test_fe  = engineer_features(test_raw)
        
        cat_cols = [
            'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage',
            'Season', 'Irrigation_Type', 'Water_Source',
            'Mulching_Used', 'Region'
        ]
        
        # LightGBM requires categorical features to be of type 'category'
        for col in cat_cols:
            train_fe[col] = train_fe[col].astype('category')
            test_fe[col]  = test_fe[col].astype('category')
            
        num_cols = [
            'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
            'Electrical_Conductivity', 'Temperature_C', 'Humidity',
            'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh',
            'Field_Area_hectare', 'Previous_Irrigation_mm',
            'Moisture_x_Temp', 'Rainfall_x_Humidity',
            'Temp_x_Humidity', 'Moisture_x_Rainfall',
            'Irrigation_per_Hectare', 'Rainfall_per_Sunlight',
            'Moisture_Deficit'
        ]
        
        feature_cols = num_cols + cat_cols
        
        X_all = train_fe[feature_cols]
        X_test = test_fe[feature_cols]
        
        le_target = LabelEncoder()
        y_all = le_target.fit_transform(train_fe[TARGET])

    # ----------------------------------------------------------
    # 3. TRAIN / VALIDATION SPLIT
    # ----------------------------------------------------------
    # We use a 90/10 split to train and validate
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.10, random_state=SEED, stratify=y_all
    )
    print(f'\n  Train: {len(X_train):,} rows | Validation: {len(X_val):,} rows')

    # ----------------------------------------------------------
    # 4. TRAIN LIGHTGBM
    # ----------------------------------------------------------
    with timer('Training LightGBM'):
        # Define LightGBM parameters for maximum accuracy
        model = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=SEED,
            n_jobs=-1,
            n_estimators=1000,          # Lots of trees
            learning_rate=0.05,         # Slower learning rate for better convergence
            num_leaves=63,              # High number of leaves allows deep interactions
            class_weight=None           # None for max overall accuracy
        )
        
        # Train with early stopping to prevent overfitting
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_cols,
            callbacks=callbacks
        )

    # ----------------------------------------------------------
    # 5. EVALUATE
    # ----------------------------------------------------------
    with timer('Evaluating Model'):
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f'\n>>> FINAL VALIDATION ACCURACY: {val_acc*100:.2f}% <<<')
        
        print('\nClassification Report:')
        print(classification_report(y_val, val_preds, target_names=le_target.classes_, digits=4))

    # ----------------------------------------------------------
    # 6. RETRAIN ON FULL DATA & PREDICT
    # ----------------------------------------------------------
    with timer('Retraining on ALL 630,000 rows (Best Iteration)'):
        best_iter = model.best_iteration_ if model.best_iteration_ else 1000
        
        final_model = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=SEED,
            n_jobs=-1,
            n_estimators=best_iter,     # Use exact number of trees found in early stopping
            learning_rate=0.05,
            num_leaves=63,
            class_weight=None
        )
        final_model.fit(X_all, y_all, categorical_feature=cat_cols)
        
    with timer('Predicting on Test Set'):
        test_preds_encoded = final_model.predict(X_test)
        test_preds_labels = le_target.inverse_transform(test_preds_encoded)
        
    # Save submission
    sub_path = os.path.join(DATA_DIR, 'submission_lgbm.csv')
    submission = pd.DataFrame({
        'id': test_raw['id'],
        'Irrigation_Need': test_preds_labels
    })
    submission.to_csv(sub_path, index=False)
    
    print(f'\n{"="*55}')
    print(f'  [OK] SUBMISSION SAVED')
    print(f'{"="*55}')
    print(f'  File  : {sub_path}')
    print(f'  Score : {val_acc*100:.2f}% Expected Accuracy')
    print('\n  Prediction distribution:')
    print(submission['Irrigation_Need'].value_counts())

if __name__ == '__main__':
    main()
