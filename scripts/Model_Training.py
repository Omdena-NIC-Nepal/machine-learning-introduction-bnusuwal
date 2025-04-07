#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\Dell\\Desktop\\Omdena\\ml_assignment\\machine-learning-introduction-bnusuwal\\data\BostonHousing.csv")


# In[6]:


from sklearn.model_selection import train_test_split

# Features and target
X = data.drop('medv', axis=1)  
y = data['medv']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print(X_train.head)

print(y_train.head)


# In[ ]:


from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train (fit) the model
model.fit(X_train, y_train)


predictions = model.predict(X_test)

# Optional: Check model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# In[4]:


from sklearn.metrics import mean_squared_error, r2_score

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)


# In[ ]:




