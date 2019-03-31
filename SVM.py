#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm


# In[2]:


#read csv data fromt he path, save it to labeled_images
#../means parent directory
labeled_images = pd.read_csv('./input/train.csv')

#save the pixels to images as a 5000x728 array
images = labeled_images.iloc[0:5000,1:]

#save the labels, which is at the first column of each sample, at 0 position
labels = labeled_images.iloc[0:5000,:1]

#use train_test_split to split the data into train aubsets and test subsets
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


# In[3]:


i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])


# In[4]:


plt.hist(train_images.iloc[i])


# In[5]:


clf = svm.SVC()
clf.fit(train_images, train_labels)
#clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# In[6]:


#make every nonzero pixel becomes 1, i.e. truning gray scale into black and white
test_images[test_images>0]=1
train_images[train_images>0]=1

#print out the black and white image
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i,0])


# In[7]:


plt.hist(train_images.iloc[i])


# In[8]:


#retrain the machine with the black and white image
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())

#show the score 
clf.score(test_images,test_labels)


# In[ ]:




