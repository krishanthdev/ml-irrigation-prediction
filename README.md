# Irrigation Need Prediction

A tabular machine learning project that predicts irrigation need from sensor and environmental features. It was built for a class competition.

## Layout

- `notebooks/` holds the modelling notebooks. These cover a gradient boosting model (`irrigation_model.ipynb`), SVM models (`svm_local.ipynb`, `svm_colab.ipynb`), and an optimised competition pipeline (`competition2_pipeline.ipynb`, `competition2_optimized.ipynb`).
- `src/` holds standalone scripts. These are a LightGBM model (`irrigation_lgbm.py`) and SVM training in local and batch variants.

## About

Part of my MSc in Artificial Intelligence Systems coursework at EPITA. Datasets, trained models and large binaries are kept out of version control.
