# GuviProject3DataScience

# Multi-Species Environmental Health Prediction Project

## Overview

This project implements advanced machine learning models for predicting health statuses across multiple species: plants, animals, and humans. It leverages state-of-the-art Convolutional Neural Networks (CNNs) with SMOTE balancing, ensemble learning, and comprehensive model evaluation. The project includes an interactive visualization dashboard for detailed performance and environmental analyses.

---

## Features

- **Advanced Data Preparation**  
  Data cleaning and preprocessing including handling missing values, label encoding, feature scaling, and SMOTE application for class balancing.

- **Superior CNN Models**  
  Separate CNN architectures trained per species with adaptive parameters and normalization techniques.

- **Ensemble Methods**  
  Combining CNN outputs with traditional ML classifiers such as Random Forests, Gradient Boosting, and Logistic Regression for improved accuracy.

- **Comprehensive Evaluation**  
  Model performance assessed through accuracy, F1-score, cross-validation, confusion matrices, and feature importances.

- **Visualization Dashboards**  
  Interactive dashboards built using Plotly for model performance, risk assessment, confidence analysis, and more. Includes 3D risk visualizations and time series environmental trends.

---

## Project Structure

- `data/`  
  Contains all cleaned and preprocessed CSV datasets with environmental and health features.

- `models/`  
  Trained model weights and scalers saved as `.h5` and `.pkl` files.

- `notebooks/`  
  Jupyter notebooks with detailed code for data processing, model training, evaluation, and visualization.

- `dashboards/`  
  Generated HTML and PNG files for interactive dashboards and evaluation visuals.

- `README.md`  
  This documentation file.

---

## Getting Started

### Requirements

- Python 3.x
- TensorFlow
- Scikit-learn
- imbalanced-learn
- Plotly
- Matplotlib, Seaborn
- Kaleido (for Plotly image export)

Install dependencies via:

pip install tensorflow scikit-learn imbalanced-learn plotly matplotlib seaborn kaleido



### Dataset

Place your datasets (`plant_complete_dataset.csv`, `animal_complete_dataset.csv`, `human_complete_dataset.csv`, etc.) inside the `data/` directory before running the notebooks.

---

## Usage Overview

### Data Preparation and Training

- Load datasets and preprocess with SMOTE balancing.
- Train superior CNN models for each species using the provided classes.
- Conduct ensemble training combining CNN and traditional ML classifiers.
- Use provided training pipelines to generate models with optimal parameters.

### Model Evaluation

- Run comprehensive evaluation including cross-validation and detailed classification reports.
- Visualize key metrics with confusion matrices, feature importances, prediction confidence histograms.

### Visualization Dashboards

- Generate interactive Plotly dashboards for multi-species model performance.
- Create 3D risk assessment visualizations and detailed time series analyses of environmental factors.
- Export dashboards as HTML and PNG files for reporting.

---

## Running the Notebooks

1. Open Jupyter or Google Colab environment.
2. Run all cells sequentially starting with data loading and preprocessing.
3. Proceed to model training cells.
4. Execute evaluation and visualization cells.
5. View and export dashboards and reports.

