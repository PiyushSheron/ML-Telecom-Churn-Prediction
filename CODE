# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from google.colab import files  # Import files for uploading

# Set the style for seaborn
sns.set(style='white')

# Upload the dataset
uploaded = files.upload()  # Upload the file

# Load the dataset, using the filename from the upload
telecom_cust = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Assuming this is the filename after upload

# Data Cleaning
telecom_cust.TotalCharges = pd.to_numeric(telecom_cust.TotalCharges, errors='coerce')
telecom_cust.dropna(inplace=True)

# Convert 'Churn' to binary using map
telecom_cust['Churn'] = telecom_cust['Churn'].map({'Yes': 1, 'No': 0})

# Create dummy variables
df_dummies = pd.get_dummies(telecom_cust.iloc[:, 1:], drop_first=True)

# EDA
# Correlation with Churn
plt.figure(figsize=(15, 8))
df_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.title('Correlation with Churn')
plt.show()

# Gender Distribution
colors = ['#4D3425', '#E4512B']
ax = (telecom_cust['gender'].value_counts() * 100.0 / len(telecom_cust)).plot(kind='bar', stacked=True, rot=0, color=colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers')
ax.set_title('Gender Distribution')

# Senior Citizen Distribution
ax = (telecom_cust['SeniorCitizen'].value_counts() * 100.0 / len(telecom_cust)).plot.pie(autopct='%.1f%%', labels=['No', 'Yes'], figsize=(5, 5), fontsize=12)
ax.set_ylabel('Senior Citizens', fontsize=12)
ax.set_title('% of Senior Citizens', fontsize=12)

# Partner and Dependents
df2 = pd.melt(telecom_cust, id_vars=['customerID'], value_vars=['Dependents', 'Partner'])
df3 = df2.groupby(['variable', 'value']).count().unstack()
df3 = df3 * 100 / len(telecom_cust)
colors = ['#4D3425', '#E4512B']
ax = df3.loc[:, 'customerID'].plot.bar(stacked=True, color=colors, figsize=(8, 6), rot=0, width=0.2)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers', size=14)
ax.set_title('% Customers with dependents and partners', size=14)
ax.legend(loc='center', prop={'size': 14})

# Customer Account Information
plt.figure(figsize=(10, 6))
sns.histplot(telecom_cust['tenure'], bins=30, kde=False)
plt.title('# of Customers by their Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('# of Customers')
plt.show()

# Contract Type Distribution
ax = telecom_cust['Contract'].value_counts().plot(kind='bar', rot=0, width=0.3)
ax.set_ylabel('# of Customers')
ax.set_title('# of Customers by Contract Type')

# Churn Rate
colors = ['#4D3425', '#E4512B']
ax = (telecom_cust['Churn'].value_counts() * 100.0 / len(telecom_cust)).plot(kind='bar', stacked=True, rot=0, color=colors, figsize=(8, 6))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers', size=14)
ax.set_title('Churn Rate', size=14)

# Prepare data for modeling
y = df_dummies['Churn'].values
X = df_dummies.drop(columns=['Churn'])

# Scaling the features
scaler = MinMaxScaler
