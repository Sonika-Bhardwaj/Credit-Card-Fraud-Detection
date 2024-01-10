#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir('/Users/sonikabhardwaj/Desktop')


# # Import important Libraries:

# In[4]:


# Importing the important libraries:
import pandas as pd # For data manipulation
import numpy as np # For mathematical and statistical Analysis
import seaborn as sns # For Visualization
import matplotlib.pyplot as plt # For Visualization
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.ensemble import IsolationForest
import shap

import warnings # To ignore warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the Dataset:

# In[5]:


df = pd.read_csv("creditcard.csv") # Reading the csv file


# # Undersatnding the dataset:

# In[6]:


df.info() # To find the data type of columns of the dataset


# In[7]:


df.columns # To know about the columns of the dataset


# In[8]:


len(df.columns) # Number of columns


# In[9]:


df.head() # Get the information of first five rows


# In[10]:


df.tail()# Get the information of last five rows


# In[11]:


df["Class"].value_counts()


# # Checking the duplicate values in the dataset:

# In[12]:


duplicate_rows = df.duplicated().sum()
print('Number of duplicate records:', duplicate_rows)
df = df.drop_duplicates()


# In[13]:


df["Class"].value_counts()


# # Analyzing the valid and fraud transactions:

# In[14]:


fraud = df[df["Class"]==1]
valid = df[df["Class"]==0]
print ("Fraud transaction cases {}".format(len(fraud)))
print ("Valid transaction cases {}".format(len(valid)))


# In[15]:


print("Amount details of fraud transaction")
fraud.Amount.describe()


# In[16]:


print("Amount details of valid transaction")
valid.Amount.describe()


# In[17]:


df["Class"].value_counts()


# # Statistical Analysis:

# In[18]:


# Statistical Analysis of the Dataset
df.describe()


# In[19]:


# Check for null values.
df.isnull().sum()


# # Data Cleaning and Preprocessing:

# In[20]:


# Data Cleaning and Preprocessing

# Handling outliers using Isolation Forest for 'Amount' and 'Time'
outlier_detector = IsolationForest(contamination=0.01, random_state=1)
df['Outlier'] = outlier_detector.fit_predict(df[['Amount', 'Time']])
df = df[df['Outlier'] == 1].drop(columns='Outlier')


# # Exploratory Data Analysis:
# 

# ### Box Plot for Amount by Class:

# In[21]:


plt.figure(figsize=(9,7))
sns.boxplot(x='Class', y='Amount', data=df, showfliers=False)
plt.title('Box Plot of Transaction Amount by Class')
plt.show()


# ### Box plot for Time by Class:

# In[22]:


plt.figure(figsize=(9,7))
sns.boxplot(x='Class', y='Time', data=df, showfliers=False)
plt.title('Box Plot of Transaction Time by Class')
plt.show()


# # Correlation Heatmaps:

# In[23]:


cor = df.corr()
fig = plt.figure(figsize =(12,10))
sns.heatmap(cor, vmax=.8, square = True)
plt.show() # Making the pair plot to understand and visualize relationship between different features.


# # Histograms for feature distributions (V1 to V28)

# In[24]:


plt.figure(figsize=(15,12))
for i in range(1, 29):  # Assuming V1 to V28 are the feature columns
    plt.subplot(7, 4, i)
    sns.histplot(df[f'V{i}'], bins=30, kde=True)
    plt.title(f'Distribution of V{i}')
plt.tight_layout()
plt.show()


# # Pairplots:

# In[25]:


plt.figure(figsize=(12,2))
sns.pairplot(df[['V1','V2','V3','V4','Class']],hue='Class')


# In[26]:


plt.figure(figsize=(12,2))
sns.pairplot(df[['V5','V6','V7','V8','Class']],hue='Class')


# In[27]:


plt.figure(figsize=(12,2))
sns.pairplot(df[['V9','V10','V11','V12','Class']],hue='Class')


# In[28]:


plt.figure(figsize=(12,2))
sns.pairplot(df[['V13','V14','V15','V16','Class']],hue='Class')


# In[29]:


plt.figure(figsize=(12,2))
sns.pairplot(df[['V17','V18','V19','V20','Class']],hue='Class')


# In[30]:


plt.figure(figsize=(12,2))
sns.pairplot(df[['V21','V22','V23','V24','Class']],hue='Class')


# In[31]:


plt.figure(figsize=(12,2))
sns.pairplot(df[['V25','V26','V27','V28','Class']],hue='Class')


# # Separating the X and Y value
# #### Dividing the data into inputs parameters and output value format

# In[33]:


# Dividing the X and Y from the dataset
X = df.drop(['Class'], axis=1)
Y = df['Class']
print(X.shape)
print(Y.shape)
xData = X.values
yData = Y.values


# # Training and Test Data Bifurcation
# ### We will divide the Datset into two main groups. One is for training model and other for testing the trained model's performance

# In[34]:


x_train,x_test,y_train,y_test = train_test_split(xData,yData, test_size = 0.3 , random_state = 42) #Building-a-Random-Forest-Model-using-a-scikit-Learn


# # Building a Logistic Regression Model using Scikit-learn

# In[35]:


# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)


# In[36]:


# Predicting on the test set
y_pred = log_reg.predict(x_test)


# In[37]:


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy*100:.2f}%")


# # Report of the Classification Model:

# In[38]:


# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# # Building a Random Forest Model using a scikit Learn

# In[39]:


cls = RandomForestClassifier()
cls.fit(x_train,y_train)


# In[40]:


y_pred = cls.predict(x_test)


# # Checking the Accuracy of the model:

# In[41]:


from sklearn.metrics import classification_report , accuracy_score


# #### Accuracy Of the Model:

# In[42]:


acc = accuracy_score(y_test, y_pred)
print("Accuracy score : {}".format(acc*100))


# # Report of the model:

# In[43]:


print("Classification report : \n",classification_report(y_test,y_pred))


# In[44]:


# Model Interpretability

# Logistic Regression coefficients
feature_names = X.columns
coefficients = log_reg.coef_[0]
feature_importance_logistic = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance_logistic = feature_importance_logistic.sort_values(by='Coefficient', ascending=False)
print('\nLogistic Regression Coefficients:\n', feature_importance_logistic)


# In[45]:


# SHAP values for Random Forest
# SHAP (SHapley Additive exPlanations) values for model interpretability
# Calculating SHAP values to understand feature importance and model predictions
# Generate SHAP values for the Random Forest model
shap_values_rf = shap.TreeExplainer(cls).shap_values(x_train)
#Create a summary plot of SHAP values to visualize feature importance
shap.summary_plot(shap_values_rf, x_train, plot_type='bar')


# In[46]:


# Save models to pickle files
import pickle

# Save Random Forest model to a file
with open('random_forestmodel.pkl', 'wb') as file:
    pickle.dump(cls.fit, file)
with open('logistic_regression.pkl', 'wb') as file:
    pickle.dump(cls.fit, file)

