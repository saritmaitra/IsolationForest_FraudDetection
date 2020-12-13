#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv('bank_fraud.csv')
df.sample(5)


# In[10]:


print(df.columns)


# In[11]:


print(df.shape)


# In[12]:


df.isnull().values.any() # checking if any null values


# In[13]:


df.describe()


# In[14]:


print(df.info())


# In[15]:


df.hist(figsize= (15,10))
plt.show()


# In[16]:


df.type.value_counts(normalize=True).plot(kind='bar', grid=True, figsize=(10, 6))


# In[17]:


Fraud = df[df['isFraud'] == 1]
Valid = df[df['isFraud'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)

print("Fraud Cases: {}".format(len(Fraud)))
print("Valid Cases: {}".format(len(Valid)))


# In[18]:


import seaborn as sns
corrmat = df.corr()
fig = plt.figure(figsize= (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[19]:


df.head()


# In[20]:


print("Type")
print('*'*15)
df.type.value_counts()


# ### Data cleaning
# We have seen from exploratory data analysis (EDA), that fraud only occurs in 'TRANSFER's and 'CASH_OUT's. So we assemble only the corresponding data in X for analysis.

# In[21]:


X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

randomState = 42
np.random.seed(randomState)

y = X['isFraud']
del X['isFraud']

# Eliminate columns shown to be irrelevant for analysis in the EDA
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

# Binary-encoding of labelled data in 'type'
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) # convert dtype('O') to dtype(int)

print(X.shape)
print(y.shape)


# In[22]:


X.head()


# In[23]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define outlier detection method
classifiers = {
    'Isolation Forest': IsolationForest(max_samples=len(X),
                                       contamination = outlier_fraction,
                                       random_state =randomState),
    'Local Outlier Factor': LocalOutlierFactor(
    n_neighbors = 20, contamination = outlier_fraction)
}


# In[24]:


# fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outlier
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scored_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # reshape the prediction values o for valid, 1 for fraud
    y_pred[y_pred ==1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != y).sum()
    
    # run classification matrics
    print('{}:{}'.format(clf_name, n_errors))
    print('*'*60)
    print(accuracy_score(y, y_pred))
    print('*'*60)
    print(classification_report(y,y_pred))


# - Though over 99% accuracy, but that is because of very few fraud cases in the dataset (Fraud Cases: 8213, Valid Cases:
#    6354407)
# - Isolation forest provides better output compared to local outlier. But, I suppose, there are still rooms for imporvment. We
#    have taken the complete data set and able to identify 34% of anomaly in transaction. 
# - We can try with neural netwrok and check if we can imporve out classification score.

# In[ ]:




