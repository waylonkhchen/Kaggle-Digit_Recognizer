#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Data preparation
# ## 2.1 Load data

# In[2]:


#load the data and pop the label
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
labels = train.pop('label')

X_train = train.values
X_test = test.values


# ## 2.2 Check for null

# In[3]:


print(train.isnull().any().describe())
print(test.isnull().any().describe())

#check the distribution of training labels
sns.countplot(labels)


# It shows no missing data.
# The label data are uniformly distributed

# ## 2.3 show an image example

# In[4]:



#reshape for showing images
img_X = [X_train[i].reshape((28,28)) for i in range(train.shape[0]) ] #reshape for visualization
#Visulaize the ith training data
i=0
plt.figure()
plt.imshow(img_X[i],cmap='gray')
plt.title(labels[i]);


# ## 2.4 label encoding

# In[5]:


#if we do the one hot vector encoding, we will use 'categorical_crossentropy' for loss function
#otherwise, with integer labels as our targets, we use  'sparse_categorical_crossentropy'

from keras.utils.np_utils import to_categorical
#encode labels to one hot vector
Y_train = to_categorical(labels,num_classes=10);
plt.plot(Y_train[0])
plt.title(labels.iloc[0]);


# encode the label to one hot vector

# ## 2.5 split training and validation

# In[6]:


from sklearn.model_selection import train_test_split

#set therandom seed
random_seed=1;
#split the train and validation sets
x_train, x_val, y_train, y_val = train_test_split(X_train,labels, test_size =.2, random_state = random_seed)


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

scaled_x_train = scaler.transform(x_train)
scaled_x_train = scaled_x_train.reshape(-1,28,28,1)

scaled_x_val = scaler.transform(x_val)
scaled_x_val = scaled_x_val.reshape(-1,28,28,1)


# # 3. CNN
# ## 3.1 Define the model

# In[8]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator #For data augmentation
from keras.callbacks import ReduceLROnPlateau #for annealer


print(tf.__version__)


# In[9]:


#label.unique()#total number of different labels

#build the keras NN model
model =Sequential([
#     Dense(16,activation='relu',input_shape = (28,28,1)),
    
    Conv2D(filters = 64, activation = 'relu',kernel_size=(5,5),padding='Same',input_shape = (28,28,1)),
    Conv2D(filters = 64, activation = 'relu',kernel_size=(5,5),padding='Same'),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    Dropout(0.25),
    
    Conv2D(filters = 64, activation = 'relu',kernel_size=(3,3),padding='Same'),
    Conv2D(filters = 64, activation = 'relu',kernel_size=(3,3),padding='Same'),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
#     BatchNormalization(axis=1),
#     Dropout(0.3),
#     Dense(32, activation='relu'),
# #     BatchNormalization(axis=1),
    Dropout(0.5),
    Dense(10, activation='softmax'), #a normalized exponential functions for probability
])


#compile the model
optimizer = RMSprop(lr=0.001)

model.compile(optimizer = optimizer,
              loss= 'sparse_categorical_crossentropy',
              metrics = ['acc']
             )


# In[10]:


model.summary()


# In[11]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[12]:


model.save_weights('model.init')


# In[13]:


#train the model
EPOCHS = 40;
batch_size = 84;

# # Display training progress by printing a single dot for each completed epoch
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     if epoch % 10 == 0: print('')
#     print('.', end='')
    
# #set earlystop when val_loss is not improving, patienece is the amount of epochs to check for improvement
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(scaled_x_train.reshape(-1,28,28,1), y_train,
                    batch_size=batch_size,
                    epochs=EPOCHS,
                    validation_data = (scaled_x_val.reshape(-1,28,28,1),y_val),
                    verbose=2,
                    callbacks = [learning_rate_reduction]
                   )
#   callbacks=[
#               early_stop,
#               PrintDot()])


# In[14]:


model.save_weights('model.fitted')


# In[15]:


#Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[16]:


from sklearn.metrics import confusion_matrix
Y_preds = model.predict(x_val.reshape(-1,28,28,1))
Y_preds = np.argmax(Y_preds,axis=1)
confusion_mtx = confusion_matrix(y_true=y_val,y_pred=Y_preds)
classes = range(10)


# In[17]:


sns.heatmap(confusion_mtx,annot=True, fmt = 'd', cmap =plt.cm.Blues)
plt.tight_layout();
plt.yticks(rotation=0)
plt.ylabel('True label');
plt.xlabel('Predicted label');


# In[18]:


# import itertools
# def plot_confusion_matrix(cm,classes,normalize=False, 
#                           title ='Confusion Matrix',cmap= plt.cm.Blues):
#     plt.imshow(cm, cmap= cmap)
#     plt.colorbar()

#     tick_marks = np.arange(len(classes));
#     plt.xticks(tick_marks, classes);
#     plt.yticks(tick_marks, classes);
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
        
# plot_confusion_matrix(confusion_mtx,classes)


# In[19]:


# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False  # randomly flip images
        ) 


datagen.fit(scaled_x_train)


# In[20]:


model.load_weights('model.init')
EPOCHS = 150;
# Fit the model
history = model.fit_generator(datagen.flow(scaled_x_train,y_train, batch_size=batch_size),
                              epochs = EPOCHS, 
                              validation_data = (scaled_x_val,y_val),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# In[21]:


model.save_weights('model.augmented_fitted')


# In[22]:


#Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[23]:


Y_preds = model.predict(x_val.reshape(-1,28,28,1))
Y_preds = np.argmax(Y_preds,axis=1)
confusion_mtx = confusion_matrix(y_val,Y_preds)
classes = range(10)


# In[24]:


sns.heatmap(confusion_mtx,annot=True, fmt = 'd', cmap =plt.cm.Blues)
plt.tight_layout();
plt.yticks(rotation=0)
plt.ylabel('True label');
plt.xlabel('Predicted label');


# ## pickle the model

# In[25]:


import pickle
#dump the augmented model
with open('augmented_model.pkl','wb') as f:
    pickle.dump(model, f)

#the inital model
model.load_weights('model.fitted')
with open('CNN_fitted_model.pkl','wb') as f:
    pickle.dump(model, f)
    
# # and later you can load it
# with open('filename.pkl', 'rb') as f:
#     clf = pickle.load(f)


# ## Submission

# In[26]:


#prepare for submission

scaled_x_test = scaler.transform(X_test)
scaled_x_test = scaled_x_test.reshape(-1,28,28,1)


# In[27]:



submission = pd.read_csv('../input/sample_submission.csv')
# print(submission.head())

submission.Label = model.predict_classes(scaled_x_test)
submission.to_csv('submission_augmented_1.csv',index=False)

model.load_weights('model.fitted')
submission.Label = model.predict_classes(scaled_x_test)
submission.to_csv('submission_no_aug.csv',index=False)


# ## Understand Data Augmentation

# In[28]:


gen = ImageDataGenerator(rotation_range=10,width_shift_range=.1,
                         height_shift_range=.1,shear_range=.15,
                        zoom_range = .1, channel_shift_range=10.)


# In[29]:


i=2;
image = scaled_x_train[i];
plt.imshow(image.reshape(28,28),cmap='gray')

#transform the image
image1 = gen.apply_transform(image,transform_parameters={'shear':20})
plt.figure()
plt.imshow(image1.reshape(28,28),cmap='gray')

image1 = gen.apply_transform(image,transform_parameters={'brightness':20})
plt.figure()
plt.imshow(image1.reshape(28,28),cmap='gray')


# In[30]:


image = X_train.reshape(-1,28,28,1)
plt.figure
plt.imshow(image.reshape(-1,28,28)[3],cmap='gray')
i=3;
aug_iter = gen.flow(image[i].reshape(-1,28,28,1))
aug_images = np.asarray([next(aug_iter) for i in range(20)])
print(aug_images.shape)
print(aug_images[0].shape)

for j in range(20):
    plt.subplot(2,10,j+1)
    plt.imshow(aug_images[j][0].reshape(28,28))
    
# for i in range(3):
#     plt.figure
#     plt.imshow(aug_images[i].reshape(28,28))
# # np.asarray(aug_images)


# In[31]:


def convolve(image, kernel ):
    length, width = image.shape;
    k_len = kernel.shape[0];#kernel is square
    
    out_len, out_wid = (length-k_len+1,width-k_len+1)
    convolved = np.zeros((out_len,out_wid))
    for i in range(length - k_len+1):
        for j in range(width-k_len+1):
            convolved[i,j] =np.dot(image[i:i+k_len,j:j+k_len].flatten(),kernel.flatten())
    return convolved
                              
            


# In[32]:


test_img = scaled_x_train[0].reshape(28,28)

plt.imshow(test_img,cmap='gray')
kernel =np.array([[-1]*3, [0]*3, [1]*3])
print(kernel)
conv = convolve(test_img,kernel)
plt.figure()
plt.imshow(conv,cmap='gray')


# In[33]:


test_img = scaled_x_train[0].reshape(28,28)

plt.imshow(test_img,cmap='gray')
kernel =np.array([[-2]*5,[-1]*5, [0]*5, [1]*5,[2]*5]).T
print(kernel)
conv = convolve(test_img,kernel)
plt.figure()
plt.imshow(conv,cmap='gray')


# In[34]:


img = X_train[45].copy()
avg = np.ma.mean(img[img>0])
print(avg)
img = np.where(img < avg,0,1);



plt.imshow(img.reshape((28,28)),cmap ='gray')
img


# In[35]:


mod

