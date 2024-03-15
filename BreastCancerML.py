#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install -U ucimlrepo')


# In[55]:


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


# In[20]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 


# In[23]:


#Training Data - Features
X


# In[105]:


#Checking Empty Values
X.info()


# In[24]:


#Training Data - Target Label
y


# In[106]:


#Checking Empty Values
y.info()


# In[81]:


#Scaling the training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[82]:


#80-20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshaping y_train to avoid an error formed as it is not a 1D array
y_train = y_train.iloc[:, 0]


# In[83]:


#Empty list for storing Model Accuracy and Name
result=[]
models=[]


# In[84]:


#Creating a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
models.append(model.__class__.__name__)


# In[85]:


#Prediction of Logistics Regression

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

result.append(accuracy)
print("Accuracy with Logistics Regression:", accuracy)


# In[86]:


#Creating a Random Forest Classifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
models.append(model.__class__.__name__)


# In[87]:


#Prediction of Random Forest Classifier

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

result.append(accuracy)
print("Accuracy with Random Forest:", accuracy)


# In[88]:


#Creating a KNN Classifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
models.append(model.__class__.__name__)


# In[89]:


#Prediction of KNN Classifier

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

result.append(accuracy)
print("Accuracy with KNN (k=5):", accuracy)


# In[90]:


#Creating a Support Vector Machine Model with Linear Kernel

model = SVC(kernel='linear')
model.fit(X_train, y_train)
models.append(model.__class__.__name__+' Linear Kernel')


# In[91]:


#Prediction of Support Vector Machine Model with Linear Kernel

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

result.append(accuracy)
print("Accuracy with SVM (Linear kernel):", accuracy)


# In[92]:


#Creating a Support Vector Machine Model with RBF Kernel

model = SVC(kernel='rbf', C=1.0, gamma=0.1)
model.fit(X_train, y_train)
models.append(model.__class__.__name__+' RBF Kernel')


# In[93]:


#Prediction of Support Vector Machine Model with RBF Kernel

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

result.append(accuracy)
print("Accuracy with SVM (RBF kernel):", accuracy)


# In[107]:


#Plotting Accuracy of each model

plt.figure(figsize=(12,8))
plt.bar(models, result, color='skyblue')
plt.ylim(0.90, 0.98)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of model')
plt.show()


# In[102]:


#Obtaining the data set as csv file for storage

df = pd.concat([X, y])
df.to_csv("BreastCancerData.csv")


# In[103]:


df


# In[ ]:




