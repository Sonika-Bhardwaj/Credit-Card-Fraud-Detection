
Credit-Card-Fraud-Detection-

Credit Card Fraud Detection

This data science project focuses on building a Credit Card Fraud Detection model using machine learning techniques. The goal is to develop a model that can effectively identify and flag potential fraudulent credit card transactions. The project utilizes a dataset containing credit card transactions, with features such as transaction amount, time, and various anonymized features.

Problem Statement

Credit card fraud is a significant concern for financial institutions and consumers alike. Detecting fraudulent transactions is challenging due to the imbalanced nature of the data, where legitimate transactions far outnumber fraudulent ones. This project aims to address this issue by developing a robust machine learning model capable of identifying patterns associated with fraudulent activities.

Installation

clone repository:

  gh repo clone Sonika-Bhardwaj/Credit-Card-Fraud-Detection
  https://github.com/Sonika-Bhardwaj/Credit-Card-Fraud-Detection

Running the code
#prerequisite


Python 3.x Jupyter Notebook (optional)

Launch Jupyter Notebook (if using).

Open the notebook or script containing the code.

Run each cell or execute the script to perform data analysis, train models, and evaluate results.

Deployment
Machine learning model

 https://github.com/Sonika-Bhardwaj/Credit-Card-Fraud-Detection/blob/main/logistic_regression.pkl

 https://github.com/Sonika-Bhardwaj/Credit-Card-Fraud-Detection/blob/main/random_forest_model.pkl

Data Sources

The dataset used in this project is sourced from [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]. It consists of a collection of credit card transactions, each labeled as either legitimate (Class 0) or fraudulent (Class 1). The dataset includes features like transaction amount, time, and anonymized numerical features (V1 to V28).

Project Structure
#Exploratory Data Analysis (EDA):

The project begins with an exploration of the dataset, examining its structure, summary statistics, and visualizing feature distributions. EDA also includes identifying and handling missing values, checking for duplicates, and exploring correlations.

#Data Cleaning:

Duplicate records are removed, and outliers in the 'Amount' and 'Time' features are handled using the Isolation Forest algorithm. The 'Amount' feature is scaled using StandardScaler.

#Data Preprocessing:

Under-sampling is performed to balance the classes. The dataset is split into training and testing sets.

#Model Selection:

Logistic Regression and Random Forest models are trained and evaluated for their performance in fraud detection.

#Model Interpretability:

The project includes an analysis of feature importance for logistic regression and SHAP (SHapley Additive exPlanations) values for Random Forest, providing insights into the model's decision-making process.

#Model Deployment:

The trained models (Logistic Regression and Random Forest) are saved in pickle files for future use




