#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


data = pd.read_csv("loan_default.csv")


# In[3]:


data.head()


# In[4]:


X = data.drop('Default', axis=1)  # Features
y = data['Default']  # Target variable


# In[5]:


categorical_columns = X.select_dtypes(include=['object']).columns


# In[6]:


# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)


# In[10]:


# Predict and evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# In[11]:


# Build the neural network
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[12]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[13]:


# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)


# In[14]:


# Evaluate the neural network on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", accuracy)


# In[15]:


# Predict and evaluate Neural Network
y_pred_dl = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred_dl))


# In[ ]:




