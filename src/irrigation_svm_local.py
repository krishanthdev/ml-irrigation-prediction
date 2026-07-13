"""
Irrigation Need Prediction — Fast SVM (Local PC)
=================================================
Trains on partial data for speed, predicts on full test set.
Generates submission.csv for Kaggle.

Usage:
    python irrigation_svm_local.py
    python irrigation_svm_local.py --sample 200000
    python irrigation_svm_local.py --sample 100000 --skip-tune
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

# ================================================================
# CONFIGURATION — edit these if needed
# ================================================================
DATA_DIR   = r'd:\Studies\Epita University\3rd Semester\ML 1 - 2'
SEED       = 42
DEFAULT_SAMPLE = 630_000   # Use ALL rows for max accuracy

np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description='Fast SVM for Irrigation Prediction')
    parser.add_argument('--sample', type=int, default=DEFAULT_SAMPLE,
                        help=f'Number of training rows to use (default: {DEFAULT_SAMPLE:,})')
    parser.add_argument('--skip-tune', action='store_true',
                        help='Skip GridSearchCV — use default hyperparameters for max speed')
    parser.add_argument('--components', type=int, default=2000,
                        help='Nystroem n_components (default: 2000)')
    return parser.parse_args()


def timer(msg):
    """Simple context-manager timer."""
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
    """Create domain-relevant interaction and ratio features."""
    df = df.copy()
    eps = 1e-6
    df['Moisture_x_Temp']       = df['Soil_Moisture'] * df['Temperature_C']
    df['Rainfall_x_Humidity']   = df['Rainfall_mm'] * df['Humidity']
    df['Temp_x_Humidity']       = df['Temperature_C'] * df['Humidity']
    df['Moisture_x_Rainfall']   = df['Soil_Moisture'] * df['Rainfall_mm']
    df['Irrigation_per_Hectare'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + eps)
    df['Rainfall_per_Sunlight'] = df['Rainfall_mm'] / (df['Sunlight_Hours'] + eps)
    df['Moisture_Deficit']      = df['Temperature_C'] - df['Soil_Moisture']
    return df


def main():
    args = parse_args()
    total_start = time.time()

    print('=' * 55)
    print('  IRRIGATION NEED PREDICTION - Fast SVM (Local)')
    print('=' * 55)
    print(f'  Training sample : {args.sample:,} rows')
    print(f'  Nystroem components : {args.components}')
    print(f'  Skip tuning : {args.skip_tune}')
    print(f'  Data dir : {DATA_DIR}')
    print('=' * 55)

    # ----------------------------------------------------------
    # 1. LOAD DATA
    # ----------------------------------------------------------
    with timer('Loading data'):
        train_path = os.path.join(DATA_DIR, 'train.csv')
        test_path  = os.path.join(DATA_DIR, 'test.csv')

        if not os.path.exists(train_path):
            print(f'ERROR: {train_path} not found!')
            sys.exit(1)
        if not os.path.exists(test_path):
            print(f'ERROR: {test_path} not found!')
            sys.exit(1)

        train_raw = pd.read_csv(train_path)
        test_raw  = pd.read_csv(test_path)

        print(f'  Train: {train_raw.shape}')
        print(f'  Test : {test_raw.shape}')

    TARGET = 'Irrigation_Need'
    print(f'\nTarget distribution (full train):')
    for cls, cnt in train_raw[TARGET].value_counts().items():
        print(f'  {cls:8s}: {cnt:>7,} ({cnt/len(train_raw)*100:.1f}%)')

    # ----------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ----------------------------------------------------------
    with timer('Feature engineering'):
        train_fe = engineer_features(train_raw)
        test_fe  = engineer_features(test_raw)

    # ----------------------------------------------------------
    # 3. DEFINE COLUMNS
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
    feature_cols = num_cols + cat_cols

    # Encode target
    le_target = LabelEncoder()
    y_full = le_target.fit_transform(train_fe[TARGET])
    print(f'\nTarget classes: {le_target.classes_}')

    X_full_df  = train_fe[feature_cols]
    X_test_df  = test_fe[feature_cols]
    test_ids   = test_fe['id'].values

    # ----------------------------------------------------------
    # 4. STRATIFIED SUBSAMPLE
    # ----------------------------------------------------------
    sample_size = min(args.sample, len(X_full_df))

    with timer(f'Stratified subsampling ({sample_size:,} rows)'):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=SEED)
        sample_idx, _ = next(sss.split(X_full_df, y_full))

        X_sample_df = X_full_df.iloc[sample_idx]
        y_sample    = y_full[sample_idx]

        for cls, cnt in zip(*np.unique(y_sample, return_counts=True)):
            name = le_target.classes_[cls]
            print(f'  {name:8s}: {cnt:>6,} ({cnt/len(y_sample)*100:.1f}%)')

    # ----------------------------------------------------------
    # 5. TRAIN / VALIDATION SPLIT
    # ----------------------------------------------------------
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_sample_df, y_sample,
        test_size=0.15, random_state=SEED, stratify=y_sample
    )
    print(f'\n  Train: {len(X_train_df):,}  |  Val: {len(X_val_df):,}')

    # ----------------------------------------------------------
    # 6. FEATURE SELECTION (Mutual Information)
    # ----------------------------------------------------------
        # To maximize accuracy, we KEEP ALL FEATURES
        selected_num = num_cols
        selected_cat = cat_cols
        print(f'  Kept ALL {len(feature_cols)} features for maximum accuracy')

    # ----------------------------------------------------------
    # 7. BUILD PREPROCESSOR
    # ----------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), selected_num),
            ('cat', OneHotEncoder(drop='first', sparse_output=False,
                                  handle_unknown='ignore'), selected_cat)
        ],
        remainder='drop'
    )

    # Update feature_cols for downstream
    sel_feature_cols = selected_num + selected_cat

    # ----------------------------------------------------------
    # 8. TRAIN SVM
    # ----------------------------------------------------------
    if args.skip_tune:
        # --- FAST PATH: no tuning ---
        with timer('Training Nystroem + SGD SVM (default params)'):
            # Compute gamma='scale' equivalent: 1 / (n_features * var(X))
            X_pre = preprocessor.fit_transform(X_train_df)
            gamma_scale = 1.0 / (X_pre.shape[1] * X_pre.var())
            preprocessor.fit(X_train_df)  # refit so pipeline is clean

            best_pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('kernel_approx', Nystroem(
                    kernel='rbf', gamma=gamma_scale,
                    n_components=args.components, random_state=SEED
                )),
                ('svm', SGDClassifier(
                    loss='hinge', alpha=1e-4,
                    class_weight=None,
                    max_iter=1000, tol=1e-3,
                    random_state=SEED, n_jobs=-1
                ))
            ])
            best_pipe.fit(X_train_df, y_train)

    else:
        # --- TUNING PATH: GridSearchCV ---
        with timer('GridSearchCV (Nystroem + SGD SVM)'):
            tuning_pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('kernel_approx', Nystroem(
                    kernel='rbf', random_state=SEED
                )),
                ('svm', SGDClassifier(
                    loss='hinge',
                    class_weight=None,
                    max_iter=1000, tol=1e-3,
                    random_state=SEED, n_jobs=-1
                ))
            ])

            # Compute gamma='scale' equivalent for Nystroem
            X_pre = preprocessor.fit_transform(X_train_df)
            gamma_scale = 1.0 / (X_pre.shape[1] * X_pre.var())
            preprocessor.fit(X_train_df)  # refit clean

            param_grid = {
                'kernel_approx__n_components': [500, args.components],
                'kernel_approx__gamma':        [gamma_scale, 0.01],
                'svm__alpha':                  [1e-5, 1e-4, 1e-3]
            }

            n_combos = 2 * 2 * 3
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

            print(f'  {n_combos} combos × 3 folds = {n_combos * 3} fits')

            grid = GridSearchCV(
                tuning_pipe, param_grid,
                cv=cv, scoring='f1_macro',
                refit=True, n_jobs=-1, verbose=1,
                return_train_score=True
            )
            grid.fit(X_train_df, y_train)

            print(f'\n  Best params : {grid.best_params_}')
            print(f'  Best F1     : {grid.best_score_:.4f}')

            best_pipe = grid.best_estimator_

            # Show top 5 results
            cv_df = pd.DataFrame(grid.cv_results_)
            top5 = cv_df.nsmallest(5, 'rank_test_score')[[
                'param_kernel_approx__n_components',
                'param_kernel_approx__gamma',
                'param_svm__alpha',
                'mean_test_score', 'rank_test_score'
            ]]
            top5.columns = ['n_comp', 'gamma', 'alpha', 'F1', 'Rank']
            print(f'\n  Top 5 configurations:')
            print(top5.to_string(index=False))

    # ----------------------------------------------------------
    # 9. EVALUATE ON VALIDATION SET
    # ----------------------------------------------------------
    with timer('Evaluating on validation set'):
        val_preds = best_pipe.predict(X_val_df)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1  = f1_score(y_val, val_preds, average='macro')

    print(f'\n  Validation Accuracy : {val_acc:.4f}')
    print(f'  Validation F1 Macro : {val_f1:.4f}')
    print()
    print(classification_report(
        y_val, val_preds,
        target_names=le_target.classes_, digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    print('Confusion Matrix:')
    cm_df = pd.DataFrame(cm, index=le_target.classes_, columns=le_target.classes_)
    print(cm_df)

    # ----------------------------------------------------------
    # 10. RETRAIN ON FULL SUBSAMPLE & PREDICT TEST
    # ----------------------------------------------------------
    with timer(f'Retraining on full subsample ({sample_size:,} rows)'):
        # Extract params from best pipeline
        ka = best_pipe.named_steps['kernel_approx']
        svm = best_pipe.named_steps['svm']

        final_gamma = ka.gamma if ka.gamma is not None else gamma_scale
        final_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('kernel_approx', Nystroem(
                kernel='rbf',
                gamma=final_gamma,
                n_components=ka.n_components,
                random_state=SEED
            )),
            ('svm', SGDClassifier(
                loss='hinge',
                alpha=svm.alpha,
                class_weight=None,
                max_iter=1000, tol=1e-3,
                random_state=SEED, n_jobs=-1
            ))
        ])
        final_pipe.fit(X_sample_df, y_sample)

    with timer('Predicting test set (270,000 rows)'):
        test_preds_encoded = final_pipe.predict(X_test_df)
        test_preds_labels  = le_target.inverse_transform(test_preds_encoded)

    # ----------------------------------------------------------
    # 11. SAVE SUBMISSION
    # ----------------------------------------------------------
    submission = pd.DataFrame({
        'id': test_ids,
        'Irrigation_Need': test_preds_labels
    })

    output_path = os.path.join(DATA_DIR, 'submission.csv')
    submission.to_csv(output_path, index=False)

    total_time = time.time() - total_start

    print(f'\n{"="*55}')
    print(f'  [OK] SUBMISSION SAVED')
    print(f'{"="*55}')
    print(f'  File  : {output_path}')
    print(f'  Shape : {submission.shape}')
    print(f'  Total time : {total_time:.0f}s ({total_time/60:.1f} min)')
    print()
    print('  Prediction distribution:')
    for cls, cnt in submission['Irrigation_Need'].value_counts().items():
        print(f'    {cls:8s}: {cnt:>7,} ({cnt/len(submission)*100:.1f}%)')
    print()
    print(submission.head(10).to_string(index=False))
    print(f'\n  Ready to submit to Kaggle!')


if __name__ == '__main__':
    main()
