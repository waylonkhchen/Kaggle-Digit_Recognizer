#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
sb.set_style("dark")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Read Data
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
train.head

print(train.shape)

# save the labels to a Pandas series target
target = train['label']
# Drop the label feature
train = train.drop("label",axis=1)

# Standardize Data
X = train.values
X_std = StandardScaler().fit_transform(X)
Y = target.values


# In[2]:


import matplotlib.pyplot as plt
n, bins, patches = plt.hist(X[4], 50, density=1, facecolor='g', alpha=0.75)
#alpha specifies opaqueness, 0.0 transparent, 1.0 opaque


#plt.axis([0, 255, 0, .2])
plt.grid(True)
plt.show()


# In[3]:


num=4
grid_data = X[num].reshape(28,28)  # reshape from 1d to 2d pixel array
plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
plt.show()
print(target[num])


# In[4]:


print(n,bins,n.sum())


# In[5]:


n, bins, patches = plt.hist(X_std[4], 50, density=1, facecolor='g', alpha=0.75)
print(X_std[4].max())

#plt.axis([0, 6, 0, 2])
plt.grid(True)
plt.show()


# In[6]:


num=4
grid_data = X_std[num].reshape(28,28)  # reshape from 1d to 2d pixel array
plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
plt.show()
print(target[num])


# In[7]:


num=4
grid_data = X_std[num].reshape(28,28)  # reshape from 1d to 2d pixel array
plt.imshow(grid_data, interpolation = "none", cmap = "binary")
plt.show()
print(target[num])


# In[8]:


print(n,bins,n.sum())


# In[9]:


n.sum()


# In[ ]:




