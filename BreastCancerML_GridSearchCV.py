#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install -U ucimlrepo')


# In[69]:


#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[64]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
   
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 


# In[3]:


#Training Data - Features
X


# In[4]:


#Checking Empty Values
X.info()


# In[5]:


#Training Data - Target Label
y


# In[6]:


#Checking Empty Values
y.info()


# In[7]:


#Scaling the training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[8]:


#80-20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshaping y_train to avoid an error formed as it is not a 1D array
y_train = y_train.iloc[:, 0]


# In[72]:


# Paramterer Grid for each Model

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [1000, 2000, 3000]}
param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
param_grid_svm_linear = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
param_grid_svm_rbf = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}


# In[22]:


# Create a list of models with their respective Parameter grids

models_param_grid = [
    (LogisticRegression(), param_grid_lr),
    (RandomForestClassifier(), param_grid_rf),
    (KNeighborsClassifier(), param_grid_knn),
    (SVC(kernel='linear'), param_grid_svm_linear),
    (SVC(kernel='rbf'), param_grid_svm_rbf)
]


# In[82]:


# Empty list for storing all models and their accuracy

models = []
accuracy = []

# Loop through each model and parameter pair from the models_param_grid List

for model, param_grid in models_param_grid:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its accuracy
    
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    
    # Store the best model and its accuracy
    
    models.append(best_model)
    accuracy.append(best_accuracy)


# In[83]:


# Getting the name of the model along with the kernel for SVM

model_names = [type(model).__name__ + f" ({model.kernel})" if isinstance(model, SVC) else type(model).__name__ for model in models]


# In[84]:


model_names


# In[85]:


# Plotting Accuracy of each model

plt.figure(figsize=(12,8))
bars = plt.bar(model_names, accuracy, color='skyblue')
plt.ylim(0.90, 0.98)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of model')

for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{acc:.4f}', ha='center', va='bottom')

plt.show()


# In[ ]:


#Obtaining the data set as csv file for storage

df = pd.concat([X, y])
df.to_csv("BreastCancerData.csv")


# In[ ]:


df


# In[88]:


score = max(accuracy)

print("The best models for identifying whether the Tumour is Malignant or begign are Logistic Regression and SCV (linear) with an accuracy of {:.2f}%".format(score*100))


# In[ ]:




