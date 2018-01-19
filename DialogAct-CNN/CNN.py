#!/bin/bash

'''
@created: 15.01.2018
@authors: Jens Beck, Fabian Fey, Richard Kollotzek


The script is stored on the server under the following path:
/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources

Usage: python2.7 CNN.py
!!!!!!!!! Server currently only supports TensorFlow python2.7 !!!!!!!!!
'''

### Imports ###
import tensorflow as tf
import cPickle as pickle
from random import shuffle
from copy import deepcopy
import numpy as np

# Server Paths
pathTraining = "NN_Input_Files/trainData_3-5WordContext_prot2.pickle"
pathEvaluation = "NN_Input_Files/devData_3-5WordContext_prot2.pickle"

trainingList = pickle.load(open(pathTraining, "rb"))
#evaluationList = pickle.load(open(pathEvaluation, "rb"))

# ### Functions ###
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def maxPool2x2(x, kernelDepth):
    return tf.nn.max_pool(x, ksize=[1, kernelDepth, 1, 1],
                          strides=[1, 1, 1, 1], padding='VALID')


### Graph definition ###
x = tf.placeholder(tf.float32, shape=[108, 300]) # input vectors

y_ = tf.placeholder(tf.float32, shape=[1, 4]) # gold standard labels; 1hot-vectors

x_4DTensor = tf.reshape(x, shape=[1, 108, 300, 1])

###
# L1 = Layer 1
# 2WC = Two-Word-Context
###
filterNumber = 20
### Two-Word-Context
W_conv_L1_2WC = weightVariable([2, 300, 1, filterNumber])
b_conv_L1_2WC = biasVariable([1])

h_conv_L1_2WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_2WC) + b_conv_L1_2WC)
h_pool_L1_2WC = maxPool2x2(h_conv_L1_2WC, 107)

### Three-Word-Context
W_conv_L1_3WC = weightVariable([3, 300, 1, filterNumber])
b_conv_L1_3WC = biasVariable([1])

h_conv_L1_3WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_3WC) + b_conv_L1_3WC)
h_pool_L1_3WC = maxPool2x2(h_conv_L1_3WC, 106)

### Four-Word-Context
W_conv_L1_4WC = weightVariable([4, 300, 1, filterNumber])
b_conv_L1_4WC = biasVariable([1])

h_conv_L1_4WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_4WC) + b_conv_L1_4WC)
h_pool_L1_4WC = maxPool2x2(h_conv_L1_4WC, 105)

### Concatenate the polling outputs the get the feature vector
outputTensor_L1 = tf.concat([h_pool_L1_2WC, h_pool_L1_3WC, h_pool_L1_4WC], 1)
# Reshape to 2D tensor
outputTensor_L1_2D = tf.reshape(outputTensor_L1, [1, 60])

### First Fully Connected Layer
W_FC_L2 = weightVariable([60, 120])
b_FC_L2 = biasVariable([120])

h_FC_L2 = tf.nn.relu(tf.matmul(outputTensor_L1_2D, W_FC_L2) + b_FC_L2)

## Dropout percentage
keep_Prob = tf.placeholder(tf.float32)
h_FC_L2_drop = tf.nn.dropout(h_FC_L2, keep_Prob)

### Second Fully Connected Layer
W_FC_L3 = weightVariable([120, 4])
b_FC_L3 = biasVariable([4])

y = tf.nn.relu(tf.matmul(h_FC_L2_drop, W_FC_L3) + b_FC_L3)

### Softmax Output
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) # (Goldstandard, Output)

### Training
learningRate = 0.1
train_Step = tf.train.GradientDescentOptimizer(learningRate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### Session ###
numEpoch = 100

config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    random_TrainingList = deepcopy(trainingList)
    for epoch in range(numEpoch):
        shuffle(random_TrainingList)
        epochAccuracyList = []

        for item in random_TrainingList:
            labels = item[1].reshape((1, 4))
            feature_Matrix =  item[0]
            batch = [feature_Matrix, labels]
            # todo Add batch processing for multithreading
            training_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_Prob: 0.5})

            train_Step.run(feed_dict={x: batch[0], y_: batch[1], keep_Prob: 0.5})
            epochAccuracyList.append(training_accuracy)

        if epoch % 10 == 0:
            print('step %d, training accuracy %g' % (epoch, np.mean(epochAccuracyList)))

    #print('test accuracy %g' % accuracy.eval(feed_dict={
    #    x: mnist.test.images, y_: mnist.test.labels, keep_Prob: 1.0}))