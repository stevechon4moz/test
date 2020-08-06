#!/usr/bin/env python
# coding: utf-8

# Exported from Jupyter Notebook

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import chi2_contingency


# In[2]:


# read in files
df = pd.read_csv("./moz_train.csv",sep="|")
df_test = pd.read_csv("./moz_test.csv",sep="|")


# In[3]:


df.describe(include='all') # overview of full data


# In[4]:


df.dtypes


# In[5]:


df['industry'].value_counts()
df['geography'].value_counts()
df['most_common_browser'].value_counts()
df['prior_month_paid'].isnull().values.any()
df['session_time_prior_month_seconds'].isnull().values.any()


# In[23]:



df['geography'].value_counts()


# In[19]:


df.isna().sum()
df['industry'].fillna(value="None", inplace=True) # fill in fake values for null
df_test.isna().sum()
df_test['industry'].fillna(value="None", inplace=True) # fill in fake values for null


# In[7]:


df.corr() # relatively weak correlation between the 2 continuous variables (if it were strong, we could perhaps do a simple linear regression)


# In[8]:


chi2_contingency(pd.crosstab(df['industry'], df['geography'])) # p > 0.05 so independent (look at relationship b/w categorical)


# In[9]:


chi2_contingency(pd.crosstab(df['most_common_browser'], df['geography'])) # p > 0.05 so independent (look at relationship b/w categorical)


# In[10]:


chi2_contingency(pd.crosstab(df['most_common_browser'], df['industry'])) # p > 0.05 so independent (look at relationship b/w categorical)


# In[11]:


hist = df['session_time_prior_month_seconds'].hist(bins=10) # indicates there are outliers (option: we could trim these)


# In[12]:


hist = df['prior_month_paid'].hist(bins=10) # no major outliers


# In[13]:


# look at visual relationship between session time and prior month paid
plt.scatter(df['session_time_prior_month_seconds'], df['prior_month_paid'])
plt.show() 


# In[14]:


# trimming is an option if there are outliers (this was done purely for investigation purposes)
q = df["session_time_prior_month_seconds"].quantile(0.95)
df_trimmed = df[df["session_time_prior_month_seconds"] < q]
plt.scatter(df_trimmed['session_time_prior_month_seconds'], df_trimmed['prior_month_paid'])
plt.show() 


# In[15]:


# assign data 
X = df['session_time_prior_month_seconds'].to_numpy().reshape(-1,1)
y = df['prior_month_paid'].to_numpy().reshape(-1,1)
X_test = df_test['session_time_prior_month_seconds'].to_numpy().reshape(-1,1)
y_test = df_test['prior_month_paid'].to_numpy().reshape(-1,1)


# In[33]:


# one hot encoding categorical features (required if employing linear regression)
preproc = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(), ['industry', 'geography', 'most_common_browser']),
    ('scale', StandardScaler(), ['session_time_prior_month_seconds'])
], remainder='passthrough') 

# transform variables
X = preproc.fit_transform(df.iloc[:,:-1]) # features
y = df.iloc[:,-1] # last column (prior_month_paid) is what we want to predict/estimate

X_test = preproc.transform(df_test.iloc[:,:-1]) # features
y_test = df_test.iloc[:,-1] # last column (prior_month_paid) is what we want to predict/estimate

# fit the data
lr = LinearRegression()
model = lr.fit(X, y)

# make prediction (estimate prior_month_paid)
X_test = preproc.transform(df_test) # encode test data
y_pred = model.predict(X_test)
r_squared = lr.score(X, y)

print(r_squared) # r^2 value 

# output predictions alongside prior_month_paid
for i in range(len(y_pred)):
	print(y_test[i]," ",y_pred[i])


# In[ ]:




