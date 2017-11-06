# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 20:01:16 2017

### Lab 3 ###

@author: YANG
"""

### import libraries ###
import tensorflow as tf
import numpy as np
import pandas as pd

## Construct the training set
## Read the loan_status_training.csv as Train
Train = pd.read_csv('./Online_Courses_training.csv')
## Create a float32 array for Train.
Train = np.array(Train).astype(np.float32)
## Extract the training features as x_train
x_train = Train[:, 1:]
## Extract the training labels as y_train
y_train = Train[:, 0]

## Construct the testing set
## Read the loan_status_testing.csv as Test
Test = pd.read_csv('./Online_Courses_testing.csv')
## Create a float32 array for Test.
Test = np.array(Train).astype(np.float32)
## Extract the testing features as x_test
x_test = Test[:, 1:]
## Extract the testing labels as y_test
y_test = Test[:, 0]

## Specify that all features have real-value data
feature_columns = [
    tf.contrib.layers.real_valued_column("", dimension=len(Train[0]) - 1)
]

## Build 3 layer DNN with 20, 20, 10 units respectively using tf.contrib.learn
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[20, 20, 10],
    n_classes=2,
    optimizer=tf.train.AdamOptimizer())

## Fit the DNN model using training data and labels
classifier.fit(x=x_train, y=y_train, steps=200000)

## Prediction
## Use the test data and ground truth labels to test the classifier
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
