# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 20:01:16 2017

### Lab 3 ###

@author: YANG
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### import libraries ###
import numpy as np
import pandas as pd
import tensorflow as tf
import uuid

tf.app.flags.DEFINE_string('training_dataset', 'Online_Courses_training.csv',
                           'Training dataset csv')
tf.app.flags.DEFINE_string(
    'testing_dataset', 'Online_Courses_testing.csv', 'testing dataset csv')
tf.app.flags.DEFINE_integer('iteration', 5000, 'iteration')
tf.app.flags.DEFINE_integer('classes', 2, 'How many classess')
tf.app.flags.DEFINE_integer('learning_rate', 0.001, 'Learning rate')

FLAGS = tf.app.flags.FLAGS

# Construct the training set
# Read the loan_status_training.csv as Train
Train = pd.read_csv(FLAGS.training_dataset)
# Create a float32 array for Train.
Train = np.array(Train, dtype=np.float32)
# Extract the training features as x_train
x_train = Train[:, 1:]
# Extract the training labels as y_train
y_train = np.array(Train[:, 0], dtype=np.bool)
# Construct the testing set
# Read the loan_status_testing.csv as Test
Test = pd.read_csv(FLAGS.testing_dataset)
# Create a float32 array for Test.
Test = np.array(Test, dtype=np.float32)
# Extract the testing features as x_test
x_test = Test[:, 1:]
# Extract the testing labels as y_test
y_test = np.array(Test[:, 0], dtype=np.bool)

# Specify that all features have real-value data
feature_columns = [
    tf.contrib.layers.real_valued_column("", dimension=len(Train[0]) - 1)
]

layer = [40,20,20,10]

# Build 3 layer DNN with 20, 20, 10 units respectively using tf.contrib.learn
classifier = tf.estimator.DNNClassifier(
    optimizer='Adam',
    feature_columns=feature_columns,
    hidden_units=layer,
    model_dir='models-{}/'.format(uuid.uuid4()),
    n_classes=2)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={feature_columns[0]: x_train},
    y=y_train,
    num_epochs=FLAGS.iteration,
    shuffle=False)

## tensorflow.estimator.
# Fit the DNN model using training data and labels
classifier.train(input_fn=train_input_fn)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={feature_columns[0]: x_test}, y=y_test, num_epochs=1, shuffle=False)
# Prediction
# Use the test data and ground truth labels to test the classifier
accuracy_score = classifier.evaluate(
    input_fn=test_input_fn)['accuracy']

print(",".join([str(x) for x in layer])+"\nLearning rate: {}".format(FLAGS.learning_rate)+'\niteration: {}'.format(FLAGS.iteration)+'\nAccuracy: {0:f}'.format(accuracy_score))
