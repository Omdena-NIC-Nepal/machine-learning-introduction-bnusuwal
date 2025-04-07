#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\Dell\\Desktop\\Omdena\\ml_assignment\\machine-learning-introduction-bnusuwal\\data\BostonHousing.csv")


# In[8]:


# Compute Q1 (25th percentile) and Q3 (75th percentile)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1  # Interquartile Range

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (data < lower_bound) | (data > upper_bound)
print(outliers.sum())  # Count of outliers per column


# Remove rows with outliers
data_no_outliers = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]


print("Original data shape:", data.shape)
print("Data without outliers shape:", data_no_outliers.shape)


# In[ ]:


plt.figure(figsize=(8, 5))
data.boxplot()
plt.title("Before Removing Outliers")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Visualization of a column outlier handling 

column = 'b'  # replace with a column you're interested in

# Histogram before
plt.figure(figsize=(10, 4))
sns.histplot(data[column], kde=True, bins=30, color='skyblue')
plt.title(f"{column} - Before Removing Outliers")
plt.show()

# Histogram after
plt.figure(figsize=(10, 4))
sns.histplot(data_no_outliers[column], kde=True, bins=30, color='green')
plt.title(f"{column} - After Removing Outliers")
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Check for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Label Encoding for binary categorical features (if any)
for feature in categorical_cols:
    if data[feature].nunique() == 2:  # For binary categorical features like 'chas'
        label_encoder = LabelEncoder()
        data[feature] = label_encoder.fit_transform(data[feature])

# One-Hot Encoding for nominal categorical features (if any)
data_encoded = pd.get_dummies(data, drop_first=True)

# Display the first few rows of the encoded dataset
print(data_encoded.head())


# In[ ]:


#Standardization (Z-score scaling): scales features to have mean = 0 and standard deviation = 1.

from sklearn.preprocessing import StandardScaler

# Select only numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
data_standardized = data.copy()
data_standardized[numeric_cols] = scaler.fit_transform(data[numeric_cols])

print(data_standardized.head())


# In[23]:


from sklearn.model_selection import train_test_split

# Features and target
X = data.drop('medv', axis=1)  
y = data['medv']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

