#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[18]:


# Read Data
train = pd.read_csv('./input/train.csv')
# test = pd.read_csv('./input/test.csv')


train_data = train.iloc[:5000]



# In[19]:


labels = train_data.pop('label')
X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size =.2, random_state =0)


# In[21]:


# Standardize the Data

Std_Scaler = StandardScaler()
Std_Scaler.fit(X_train)
X_train_std = Std_Scaler.transform(X_train)
X_test_std = Std_Scaler.transform(X_test)

y_train = y_train.values




# In[22]:


# Multinomial Logistic Regression
lgRg = LogisticRegression(multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.0001)
lgRg.fit(X_train_std, y_train)
y = lgRg.predict(X_test_std)

# Submission
# submission = pd.DataFrame({"ImageId": range(1,y.shape[0]+1),
#     "label": y})

# submission.to_csv("submission_Logistic.csv", index=False)


# In[25]:


lgRg.score(X_test_std,y_test)


# In[15]:


print(y[:5])


# In[13]:


print(y)


# In[8]:





# In[16]:


print(X)


# In[18]:


print(X.shape)


# In[26]:


lgRg.get_params([])


# In[33]:


lgRg.coef_.shape


# In[4]:


lgRg.coef_[1:]


# In[5]:


test.shape


# In[18]:


get_ipython().run_line_magic('pinfo', 'x_std')


# In[41]:


lgRg.predict(x_std[0:5,:])


# In[46]:


test[1,:]


# In[48]:


train.iloc[1].as_matrix().reshape(28,28)


# In[54]:


lgRg.predict_proba(x_std[0:2,:])


# In[56]:


lgRg.predict(x_std[0:2,:])


# In[59]:


x_std.dtype


# In[62]:


x_std.shape


# In[ ]:




