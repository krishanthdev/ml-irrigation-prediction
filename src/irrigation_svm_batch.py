import numpy as np
import pandas as pd
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, classification_report
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE

# ================================================================
# CONFIGURATION
# ================================================================
DATA_DIR    = r'd:\Studies\Epita University\3rd Semester\ML 1 - 2'
SEED        = 42
TARGET      = 'Irrigation_Need'
VAL_SIZE    = 50_000       # Size of static validation set
CHUNK_SIZE  = 50_000       # Size of training batches
TARGET_ACC  = 0.95         # Early stopping threshold

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
    print('  IRRIGATION NEED PREDICTION - Smart Batch SVM')
    print('=' * 55)
    print(f'  Target Accuracy : {TARGET_ACC*100:.1f}%')
    print(f'  Chunk Size      : {CHUNK_SIZE:,}')
    print('=' * 55)

    train_path = os.path.join(DATA_DIR, 'train.csv')
    
    # ----------------------------------------------------------
    # 1. LOAD VALIDATION SET & INITIAL SAMPLE FOR PREPROCESSOR
    # ----------------------------------------------------------
    with timer('Loading Validation Set & Initial Sample'):
        # We read the first 100k rows to fit the preprocessor (distribution is random enough)
        # We use the LAST 50k rows as the static validation set.
        
        df_init = pd.read_csv(train_path, nrows=100_000)
        df_init_fe = engineer_features(df_init)
        
        # Validation set (skip first 580k rows, read last 50k)
        total_rows = 630_000
        skip_rows = total_rows - VAL_SIZE
        # read header first
        header = pd.read_csv(train_path, nrows=0).columns
        df_val = pd.read_csv(train_path, skiprows=range(1, skip_rows+1), names=header, header=0)
        df_val_fe = engineer_features(df_val)

    # ----------------------------------------------------------
    # 2. DEFINE FEATURES & FIT PREPROCESSOR
    # ----------------------------------------------------------
    cat_cols = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage',
        'Season', 'Irrigation_Type', 'Water_Source',
        'Mulching_Used', 'Region'
    ]
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
    
    le_target = LabelEncoder()
    y_init = le_target.fit_transform(df_init_fe[TARGET])
    y_val  = le_target.transform(df_val_fe[TARGET])
    classes = np.unique(y_init)
    
    with timer('Fitting Preprocessor & Nystroem on initial sample'):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', TargetEncoder(), cat_cols)
            ],
            remainder='drop'
        )
        
        # Fit preprocessor (TargetEncoder requires y)
        X_init_pre = preprocessor.fit_transform(df_init_fe, y_init)
        
        # Compute Gamma dynamically
        gamma_scale = 1.0 / (X_init_pre.shape[1] * X_init_pre.var())
        n_comp = 2000
        
        nystroem = Nystroem(kernel='rbf', gamma=gamma_scale, n_components=n_comp, random_state=SEED)
        nystroem.fit(X_init_pre)
        
        # Transform validation set once to save time during loops
        X_val_pre = preprocessor.transform(df_val_fe)
        X_val_nys = nystroem.transform(X_val_pre)

    # ----------------------------------------------------------
    # 3. HYPERPARAMETER EXPLORATION (BATCH TRAINING LOOP)
    # ----------------------------------------------------------
    print(f'\n{"-"*55}')
    print(f'>> Starting Batch Training (Early Stopping at {TARGET_ACC*100:.1f}%)')
    
    # Best alpha found in previous runs
    alphas = [1e-05]
    best_model = None
    best_acc = 0.0
    
    smote = SMOTE(random_state=SEED)
    
    for alpha in alphas:
        print(f'\n  --- Testing configuration: alpha={alpha} ---')
        
        svm = SGDClassifier(
            loss='hinge', 
            alpha=alpha,
            class_weight=None, # Maximize accuracy
            learning_rate='optimal',
            max_iter=1,  # 1 iteration per chunk
            random_state=SEED, 
            n_jobs=-1
        )
        
        # To avoid reading the same CSV from disk constantly, we read in chunks
        chunk_iter = pd.read_csv(train_path, chunksize=CHUNK_SIZE)
        
        rows_processed = 0
        batch_num = 1
        model_acc = 0
        
        for chunk in chunk_iter:
            # Stop if we reach the validation split
            if rows_processed >= skip_rows:
                break
                
            chunk_fe = engineer_features(chunk)
            X_chunk_pre = preprocessor.transform(chunk_fe)
            y_chunk = le_target.transform(chunk_fe[TARGET])
            
            # Apply SMOTE to balance classes within the chunk
            X_chunk_res, y_chunk_res = smote.fit_resample(X_chunk_pre, y_chunk)
            
            # Nystroem transform
            X_chunk_nys = nystroem.transform(X_chunk_res)
            
            # Partial fit on balanced chunk
            svm.partial_fit(X_chunk_nys, y_chunk_res, classes=classes)
            
            rows_processed += len(chunk)
            
            # Evaluate on static validation set
            val_preds = svm.predict(X_val_nys)
            model_acc = accuracy_score(y_val, val_preds)
            
            print(f'    Batch {batch_num:>2} ({rows_processed:>7,} rows) | Val Acc: {model_acc:.4f}')
            
            if model_acc >= TARGET_ACC:
                print(f'    [!] Target accuracy {TARGET_ACC*100}% reached!')
                break
                
            batch_num += 1
            
        if model_acc > best_acc:
            best_acc = model_acc
            best_model = svm
            print(f'    -> New best model! (Acc: {best_acc:.4f})')
            
        if best_acc >= TARGET_ACC:
            break

    # ----------------------------------------------------------
    # 4. PREDICT ON TEST SET
    # ----------------------------------------------------------
    print(f'\n{"-"*55}')
    print(f'>> Generating predictions using best model (Acc: {best_acc:.4f})...')
    
    t0 = time.time()
    test_path = os.path.join(DATA_DIR, 'test.csv')
    test_raw = pd.read_csv(test_path)
    test_fe = engineer_features(test_raw)
    
    X_test_pre = preprocessor.transform(test_fe)
    X_test_nys = nystroem.transform(X_test_pre)
    
    test_preds_encoded = best_model.predict(X_test_nys)
    test_preds_labels = le_target.inverse_transform(test_preds_encoded)
    
    submission = pd.DataFrame({
        'id': test_raw['id'],
        'Irrigation_Need': test_preds_labels
    })
    
    sub_path = os.path.join(DATA_DIR, 'submission_batch.csv')
    submission.to_csv(sub_path, index=False)
    
    print(f'[OK] Done in {time.time()-t0:.1f}s')
    print(f'\n  Saved to: {sub_path}')
    print(f'  Best Validation Accuracy: {best_acc:.4f}')
    print('\nPrediction distribution:')
    print(submission['Irrigation_Need'].value_counts())

if __name__ == '__main__':
    main()
