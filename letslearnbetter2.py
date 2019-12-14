from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import PIL #this is a package that makes data loading easier. You should
#install it separately from tensorflow. Install Pillow

from sklearn.model_selection import StratifiedKFold #allows you to do k-fold validation
from sklearn.metrics import confusion_matrix

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
from time import time
os.chdir(Path(__file__).resolve().parent)
#set the directory to where the this file is



def load_image(px,py,bsize):
    """load num_images of  bitmap images of size px x py into numpy array,
    along with labels whether the image contains monololayers"""
   
    train_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
	    zoom_range=0.05,
	    width_shift_range=0.02,
	    height_shift_range=0.02,
	    shear_range=0.05,
	    fill_mode="nearest")
    
    valid_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_image= train_generator.flow_from_directory("./resized_split/train",
    target_size=(px, py), classes=["nonmono", "mono"], class_mode='binary',
    batch_size=bsize,    shuffle=True)

    valid_image= valid_generator.flow_from_directory("./resized_split/valid",
    target_size=(px, py), classes=["nonmono", "mono"], class_mode='binary',
    batch_size=bsize, shuffle=True)

    test_image= test_generator.flow_from_directory("./resized_split/test",
    target_size=(px, py), classes=["nonmono", "mono"], class_mode='binary',
    batch_size=bsize, shuffle=True)
    
    return train_image, valid_image[0], test_image[0]

def LogisticRegression(px,py,train_set,valid_set,epochnum):

    """Logistic Regression with stochastic gradient descent with validation"""

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(px, py,3)),
        keras.layers.Dense(1, activation='sigmoid') ])
    kernel_regularizer=keras.regularizers.l2(0.001)

    #sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 'FalseNegatives', 'FalsePositives'])

    his=model.fit_generator(train_set,
    epochs=epochnum,
#    class_weight = {0: 0.5, 1: 1.0},
    validation_data =valid_set)

    ypred= model.predict(valid_set[0])
    cfmatrix = confusion_matrix(valid_set[1], np.floor(ypred[:,0]/0.5))

    np.savetxt("./logistic_loss.csv",np.array([his.history['loss'],
    his.history['binary_accuracy'],
    his.history['FalseNegatives'],
    his.history['FalsePositives'],
    his.history['val_loss'],
    his.history['val_binary_accuracy'],
    his.history['val_FalseNegatives'],
    his.history['val_FalsePositives'],]).T, delimiter=",")
    np.savetxt("./logistic_confusion.csv", cfmatrix, delimiter=",")
    
    
def CNNvanilla(px,py,train_set,valid_set,epochnum):

    """convolutional neural network training and validation"""
    # ------Define the neural network------
    model = keras.Sequential([
        keras.layers.Conv2D(16, 11, strides = (3,3), padding='valid', activation='relu', input_shape=(px, py ,3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(24, 5, padding='valid', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='valid', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation = 'relu'),
        keras.layers.Dense(1, activation='sigmoid') ])
    model.summary()
    
    #kernel_regularizer=keras.regularizers.l2(0.001)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5)

    model.compile(optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 'FalseNegatives', 'FalsePositives'])
    
    # ------Train the neural network------
    his = model.fit_generator(train_set,
    verbose=2,
    epochs=epochnum,
    validation_data =valid_set)
    np.savetxt("./CNN1_loss.csv", np.array([his.history['loss'],
    his.history['binary_accuracy'],
    his.history['FalseNegatives'],
    his.history['FalsePositives'],
    his.history['val_loss'],
    his.history['val_binary_accuracy'],
    his.history['val_FalseNegatives'],
    his.history['val_FalsePositives'],]).T, delimiter=",")
    
    # ------Get confusion matrix------
    #True negative : (0,0),  #False Positive : (0,1),
    #False negative : (1,0), #True positive : (1,1), 
    ypred= model.predict(valid_set[0])
    cfmatrix = confusion_matrix(valid_set[1], np.floor(ypred[:,0]/0.5))
    np.savetxt("./CNN1_valid_confusion.csv", cfmatrix, delimiter=",")

    return model

def test_accuracy(model,test_set):
    """test the accuracy of CNN on test set"""
    ypred= model.predict(test_set[0])
    cfmatrix = confusion_matrix(test_set[1], np.floor(ypred[:,0]/0.5))
    np.savetxt("./CNN1_test_confusion.csv", cfmatrix, delimiter=",")
    
if __name__ == '__main__':
    train_set, valid_set, test_set = load_image(375,375,32)
    start=time()
    LogisticRegression(375,375,a,b,30)
    model=CNNvanilla(375, 375, train_set, valid_set, 60)
    test_accuracy(model,test_set)
    end=time()
    print("Used time:", (end-start)/60, " min.")
