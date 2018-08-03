# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:27:33 2018

@author: Murli Singh Rajpurohit
"""


import keras 
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import normalize 
seed = 1234
np.random.seed(seed)
data=pd.read_csv("C:/Users/murali.nsr/Downloads/Iris.csv")
print("Describing the data: ",data.describe())
print("Info of the data:",data.info())

print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))

print(data["Species"].unique())

data.loc[data["Species"]=="Iris-setosa","Species"]=0
data.loc[data["Species"]=="Iris-versicolor","Species"]=1
data.loc[data["Species"]=="Iris-virginica","Species"]=2
print(data.head())
print(data.tail())

data=data.iloc[np.random.permutation(len(data))]
print(data.head())

X=data.iloc[:,0:4].values
y=data.iloc[:,4].values

print("Shape of X",X.shape)
print("Shape of y",y.shape)

X_normalized=normalize(X,axis=0)

total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X_normalized[:train_length]
X_test=X_normalized[train_length:]
y_train=y[:train_length]
y_test=y[train_length:]

print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])

from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)
print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)

model=Sequential()
model.add(Dense(1000,input_dim=4,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)

prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",accuracy )

















