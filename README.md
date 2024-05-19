# Pycaret-streamlit-app-2.0

This project is a Streamlit application for exploratory data analysis (EDA) and supervised learning model building. It allows users to load datasets, visualize data, set up preprocessing parameters, build supervised learning models, and evaluate model performance.

## Modules
1. [`main.py`](#1-mainpy)
1. [`datavisualizer.py`](#2-datavisualizerpy)
1. [`model_building.py`](#3-model_buildingpy)
1. [`plot.py`](#4-plotpy)
1. [`pages/eda.py`](#5-pages/edapy)
1. [`pages/model_building.py`](#6-pages/model_buildingpy)
### 1. `main.py`

#### Main Page

This Streamlit page serves as the main landing page for the application. It provides an overview of the project and navigation links to other pages.

- **Functionality**: 
    - Page Title
    - load dataset
    - Navigation Links
    - File Upload
    - Data Preview

- **Dependencies**: 
    - Streamlit
    - st-pages

- **Usage**:
    1. Run the Streamlit app.
    2. Navigate to the "Main" page.
    3. Click on the navigation links to explore other pages.
    4. Upload data file (csv, excel, etc.).
    5. View the preview of the dataset.

### 2. `datavisualizer.py`

#### Data Visualizer Module

This module provides functions for visualizing data, including histograms, scatter plots, box plots, pair plots, heatmaps, and covariance matrices.

- **Dependencies**: 
    - matplotlib
    - pandas
    - seaborn
    

- **Functions**:
    - Histogram
    - Scatter Plot
    - Box Plot
    - Pair Plot
    - Heatmap
    - Covariance Matrix
    - Confusion_matrix
    - Classification_report
    - hlines


### 3. `model_building.py`

#### Model Building Module

This module provides classes and methods for setting up, building, and evaluating supervised learning models.

- **Dependencies**: 
    - pycaret
    - pandas
    - numpy
    - scikit-learn
    - joblib
    - category_encoders

- **Classes**:
    - ModelBuilding
    - ModelEvaluation

- **Functions**:
    - estimator_setter
    - clean_data
    - get_problem_type

### 4. `plot.py`

#### Plot Module

This module provides functions for customizing and plotting figures with Streamlit.

- **Dependencies**: 
    - streamlit
    - matplotlib

- **Functions**:
    - plot
    - plot2

### 5. `pages/eda.py`

#### Exploratory Data Analysis (EDA) Page

This Streamlit page allows users to perform exploratory data analysis on uploaded datasets.

- **Dependencies**: 
    - Streamlit
    - matplotlib
    - st_pages

- **Functionality**:
    - Data Visualizations
    - Data Summary
    - Session State Handling

### 6. `pages/model_building.py`

#### Model Building Page

This Streamlit page provides an interactive interface for setting up, building, and evaluating supervised learning models.

- **Dependencies**: 
    - Streamlit
    - st_pages

- **Functionality**:
    - Model Setup
    - Model Building
    - Model Evaluation
