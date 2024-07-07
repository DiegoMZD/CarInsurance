# Car Insurance Claim Prediction
This repository contains a machine learning project focused on predicting car insurance claims based on policy, car, and demographic features.

![Car Image](readme_images/car.png)

## Project Overview
In the insurance industry, accurately predicting whether a policyholder will file a claim is crucial for risk assessment and operational efficiency. This project aims to build and deploy a classification model that can effectively predict insurance claims. By doing so, it helps mitigate the financial impact of false positives and false negatives, where false negatives typically incur higher costs for insurance companies.

## Dataset
The dataset used in this project is sourced from Kaggle, comprising 44 columns and 58,592 rows. Each row represents a policy record, including features such as policy ID, policy tenure, car specifications (make, model, age, etc.), demographic details (age of policyholder, city, population density), and a binary target variable indicating whether the policy resulted in a claim.  
Link to the original dataset: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification/data

## Methodology
### 1. Data Preprocessing
* Checked and handled missing values, duplicates, and ensured data cleanliness.
* Encoded categorical variables and scaled numerical features for modeling.
### 2. Exploratory Data Analysis (EDA)
* Explored distributions of variables through histograms, box plots, and count plots.
* Investigated correlations and relationships between features.
* Conducted hypothesis tests to identify statistically significant differences and associations.
### 3. Baseline Models
* Trained baseline models including KNeighbors Classifier, Decision Tree Classifier, and Logistic Regression.
* Evaluated models using classification reports and confusion matrices.
### 4. Ensemble Methods and Advanced Models
* Implemented Random Forest Classifier and XGBoost Classifier with hyperparameter tuning using Random Search.
* Applied Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance.
* Explored Multilayer Perceptron Classifier (MPC) for neural network-based predictions.
### 5. Model Evaluation
* Optimized models based on business costs associated with false positives and false negatives.
* Selected the best-performing model based on evaluation metrics including ROC AUC and business cost.

## Results
The final model selected for deployment is the XGBoost Classifier after hyperparameter tuning. It demonstrated robust performance with the highest ROC AUC, effectively minimizing the targeted cost function.

## Conclusion and Recommendations
This project highlights the importance of accurate predictive modeling in insurance claims. Future improvements could include acquiring a dataset with reduced class imbalance, performing more exhaustive hyperparameter searches, and refining the business cost function for precise calculations.

## Requirements
* imbalanced-learn==0.12.3
* imblearn==0.0
* numpy==1.26.4
* pandas==2.2.1
* scikit-learn==1.3.0
* scipy==1.12.0
* seaborn==0.13.2
* statsmodels==0.14.2
* xgboost==2.1.0
* matplotlib==3.8.0
* ipython==8.12.3

## License
This project is licensed under the MIT License - see the LICENSE file for details.