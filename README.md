# customer-churn-analysis  

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Overview  

Customer churn refers to the phenomenon where customers discontinue their relationship or subscription with a company or service provider. It represents the rate at which customers stop using a company's products or services within a specific period.  

Churn is an important metric for businesses as it directly impacts revenue, growth, and customer retention. Understanding customer churn is crucial for businesses to identify patterns, factors, and indicators that contribute to customer attrition. By analyzing churn behavior and its associated features, companies can develop strategies to retain existing customers, improve customer satisfaction, and reduce customer turnover.  

This project leverages machine learning techniques to analyze and predict customer churn using the **Customer Churn Dataset** sourced from Kaggle:  
[Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data).  

## Data Preprocessing  

1. **Data Cleaning**  
   - Removed the `CustomerID` column as it is not a predictive feature.  
   - Handled missing values by dropping rows with null values.  

2. **Feature Engineering**  
   - Created new features:  
     - `Spend_per_Tenure = Total Spend / (Tenure + 1)`  
     - `SupportCalls_per_Tenure = Support Calls / (Tenure + 1)`  
     - `Usage_per_Tenure = Usage Frequency / (Tenure + 1)`  

3. **Data Transformation**  
   - **Converted data types**:  
     - Numerical columns (`Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction`, `Churn`) were converted to `int64`.  
   - **Categorical Encoding**:  
     - `Gender`, `Subscription Type`, and `Contract Length` were converted into categorical variables for better compatibility with machine learning models.  

4. **Data Splitting & Scaling**  
   - **Splitting**: 80% training, 20% testing.  
   - **Feature Scaling**: Min-Max Normalization applied to standardize feature values.  

## Machine Learning Models  

The project applies **machine learning models** to predict customer churn.  

### **1. Artificial Neural Network (ANN) - Model 1**  
- **Architecture**:  
  - Input layer: `(X_train_scaled.shape[1],)`  
  - Hidden layers: **64-32-16** neurons with `ReLU` activation  
  - Regularization: `Dropout(30%)`  
  - Output layer: **Sigmoid activation**  
- **Training**:  
  - Optimizer: **Adam**  
  - Loss function: `binary_crossentropy`  
  - Epochs: **8**  
- **Performance**:  
  - **Accuracy**: `99.26%`  
  - **AUC-ROC Score**: `0.9997`  
- **Confusion Matrix**: [[38039 24] [ 625 49479]]

### **2. ANN Model 2 - Weighted Binary Cross-Entropy**  
- **Improvements over Model 1**:  
- Introduced **Weighted Binary Cross-Entropy loss**  
- Added **Batch Normalization**  
- Increased **neuron count** and **Dropout (40% & 30%)**  
- **Training Results**:  
- **Accuracy**: `99.58%`  
- **AUC-ROC Score**: `0.9997`  
- **Confusion Matrix**: [[38063 0] [ 369 49735]]
- **Final Decision**: This model was **selected as the best-performing ANN model**.  

### **3. Artificial Neural Network (ANN) - Model 3**  
- **Additional Enhancements**:  
- **More neurons** per hidden layer (256-128-64-32)  
- **LeakyReLU activations** instead of ReLU  
- **L2 Regularization** (`0.01`) added to dense layers  
- **Performance**:  
- **Accuracy**: `98.82%` (slightly lower than Model 2)  
- **AUC-ROC Score**: `0.9984`  
- **Confusion Matrix**: [[38059 4] [ 1038 49066]]

- **Decision**: **Model 2 remains the best ANN model**; Model 3 did not outperform Model 2.  

### **4. XGBoost Model (Best Performing Model)**  
- **Hyperparameter tuning**: Used `GridSearchCV` to optimize:  
- `n_estimators`: [100, 300, 500]  
- `max_depth`: [4, 6, 8]  
- `learning_rate`: [0.01, 0.05, 0.1]  
- `min_child_weight`: [1, 3, 5]  
- `gamma`: [0, 1, 5]  
- `subsample`: [0.8, 1.0]  
- `colsample_bytree`: [0.8, 1.0]  

- **Performance (Best model)**:  
- **Accuracy**: `99.92%`  
- **Precision, Recall, and F1-score: ~1.00**  
- **AUC-ROC Score**: `1.0000`  
- **Confusion Matrix**: [[38063 0] [ 70 50034]]
- **Final Model Selection**: **XGBoost was chosen as the final model due to its superior performance.**  

### **5. Logistic Regression Model (Baseline)**  
- **Performance**:  
- Accuracy: **89.64%**  
- AUC-ROC Score: **0.9596**  
- **Confusion Matrix**: [[34563 3500] [ 5629 44475]]

- This model underperformed compared to the ANN and XGBoost models.  

## Results  

The best-performing model was **XGBoost with hyperparameter tuning**, achieving:  
- **Accuracy: 99.92%**  
- **AUC-ROC Score: 1.0000**  
- **Final Model:** `xgb_tuned.joblib`  

## Project Organization

├── LICENSE            
├── Makefile           
├── README.md          <- Project documentation  
├── data               <- Dataset and processed files  
├── docs               <- Documentation  
├── models             <- Trained models and predictions  
├── notebooks          <- Jupyter notebooks  
├── pyproject.toml     <- Project metadata and configuration  
├── references         <- Data dictionaries, manuals, etc.  
├── reports            <- Generated analysis and figures  
├── requirements.txt   <- Dependencies  
├── setup.cfg          <- Code formatting rules  
└── customer-churn-analysis  
    ├── __init__.py  
    ├── config.py  
    ├── dataset.py  
    ├── features.py  
    ├── modeling  
    │   ├── __init__.py  
    │   ├── predict.py  
    │   └── train.py  
    └── plots.py  

## Conclusion

This project focuses on analyzing and predicting customer churn using different machine learning models, including **artificial neural networks (ANN)** and **XGBoost**. The best-performing model was the **optimized XGBoost model**, achieving an **accuracy of 99.92%** and an **AUC-ROC score of 1.0000**, making it the preferred model for customer churn prediction.

By understanding factors such as tenure, usage frequency, support calls, payment delay, subscription type, and total spend, businesses can implement data-driven strategies to retain customers and enhance customer satisfaction.
