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
Train = 'Please fill in the missing code'
## Create a float32 array for Train.
Train = 'Please fill in the missing code'
## Extract the training features as x_train
x_train = 'Please fill in the missing code'
## Extract the training labels as y_train
y_train = 'Please fill in the missing code'

## Construct the testing set
## Read the loan_status_testing.csv as Test
Test = 'Please fill in the missing code'
## Create a float32 array for Test.
Test = 'Please fill in the missing code'
## Extract the testing features as x_test
x_test = 'Please fill in the missing code'
## Extract the testing labels as y_test
y_test = 'Please fill in the missing code'

## Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = len(Train[0]) - 1)]

## Build 3 layer DNN with 20, 20, 10 units respectively using tf.contrib.learn
classifier = 'Please fill in the missing code'

## Fit the DNN model using training data and labels
'Please fill in the missing code'

## Prediction
## Use the test data and ground truth labels to test the classifier
accuracy_score = 'Please fill in the missing code'
print('Accuracy: {0:f}'.format(accuracy_score))


