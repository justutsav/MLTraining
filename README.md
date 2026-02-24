# Crop Yield Prediction - Machine Learning Analysis

This repository contains a comprehensive machine learning analysis for predicting agricultural crop yields based on various environmental and agricultural factors. The project utilizes a Random Forest Regressor to model the complex relationships between rainfall, temperature, pesticide usage, and crop yield across different regions.

## Project Files

- `case_study.ipynb`: The main Jupyter Notebook containing the data analysis, preprocessing, and model training.
- `yield_df.csv`: The primary merged dataset used for training and testing.
- `pesticides.csv`, `rainfall.csv`, `temp.csv`, `yield.csv`: Source datasets providing individual factors.
- `README.md`: This documentation file.

---

## Detailed Code Walkthrough

The following sections explain each code block in the `case_study.ipynb` notebook, detailing the rationale and methodology.

### 1. Data Loading and Initial Cleaning
**Why?**
The first step in any data science project is to import the necessary libraries and load the dataset. We also need to remove any artifact columns (like extra indices) that might have been added during dataset exportation or storage.

**How we solved it:**
We used `pandas` to read the CSV file. We specifically looked for an `Unnamed: 0` column, which is a common artifact in datasets downloaded from platforms like Kaggle, and dropped it to ensure our model only trains on relevant features.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('yield_df.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
```

### 2. Exploratory Data Quality Check
**Why?**
Machine learning models often cannot handle missing values (NaN) directly. We need to verify if the dataset is clean or if we need to perform imputation or row deletion.

**How we solved it:**
We ran `df.isnull().sum()` to get a count of null values in each column. The results confirmed that the pre-merged dataset used here is clean and has no missing values.

### 3. Feature Engineering and Preprocessing
**Why?**
The dataset contains both numerical values (Year, Rainfall, Temp) and categorical values (Area, Item). Machine learning algorithms require all inputs to be numerical. Furthermore, numerical features often have different scales (e.g., rainfall in thousands vs. temperature in tens), which can bias the model.

**How we solved it:**
We implemented a `ColumnTransformer` with two distinct pipelines:
- **StandardScaler**: Applied to numerical columns to normalize the data (mean=0, std=1).
- **OneHotEncoder**: Applied to categorical columns (Area, Item) to transform them into a format suitable for the algorithm while handling potential unknown categories during testing.

### 4. Model Selection and Training
**Why?**
Agricultural yield data is non-linear and highly dependent on interactions between various factors. While Linear Regression is a simpler approach, it often fails to capture these complex interactions.

**How we solved it:**
We utilized a `RandomForestRegressor`, an ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction. We combined the preprocessor and the regressor into a single `Pipeline` to ensure that data transformation is consistently applied to both training and test sets.

### 5. Performance Evaluation
**Why?**
We need to quantify how well our model generalizes to unseen data. Two key metrics are used: R-squared ($R^2$) to measure the proportion of variance explained by the model, and Mean Absolute Error (MAE) to understand the average magnitude of prediction errors in actual yield units.

**How we solved it:**
After splitting the data into 80% training and 20% testing sets, the model achieved:
- **Model Accuracy (R2 Score)**: ~0.9877 (indicating a very strong fit).
- **Average Prediction Error (MAE)**: ~3474.24 hg/ha.

### 6. Correlation Analysis
**Why?**
Understanding which factors move together helps in explaining why the model makes certain decisions. For example, the relationship between pesticides and yield might be stronger than the relationship between year and yield.

**How we solved it:**
We generated a heatmap using `seaborn`. This visualization highlights the correlation coefficients between numerical variables, providing a clear view of the underlying data structure.

---

## How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```
2. Open `case_study.ipynb` in your preferred Jupyter environment (JupyterLab, VS Code, etc.).
3. Execute the cells sequentially to reproduce the analysis and model results.
