#!/bin/bash

"""
@created: 15.01.2018
@authors: Jens Beck, Fabian Fey, Richard Kollotzek


The script is stored on the server under the following path:
/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources

Usage: python2.7 CNN.py
!!!!!!!!! Server currently only supports TensorFlow python2.7 !!!!!!!!!
"""

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
evaluationList = pickle.load(open(pathEvaluation, "rb"))

### Functions ###
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def maxPool100x1(x, kernelDepth):
    return tf.nn.max_pool(x, ksize=[1, kernelDepth, 1, 1],
                          strides=[1, 1, 1, 1], padding='VALID')

### Creates and returns a batchlist with the following format
#   batchList = [ (1Batch-Features, 1Batch-Labels), (2Batch-Features, 2Batch-Labels), ...]
###
def createBatchList(random_TrainingList, batchSize):
    npArrayDepth = 0
    batchList = []
    tmpFeatureList = []
    tmpLabelList = []

    for i in range(1, len(random_TrainingList)+1):
        npArrayDepth += 1
        tmpFeatureList.append(random_TrainingList[i-1][0])
        tmpLabelList.append(random_TrainingList[i-1][1])

        if (i % batchSize == 0) and (i != 0):
            featureBatchArray = np.array(tmpFeatureList)
            labelsBatchArray = np.array(tmpLabelList)

            featureBatchArray = featureBatchArray.reshape(npArrayDepth, 32400)
            labelsBatchArray = labelsBatchArray.reshape(npArrayDepth, 4)

            #print(labelsBatchArray.shape)

            batchList.append((featureBatchArray, labelsBatchArray))
            tmpFeatureList = []
            tmpLabelList = []
            npArrayDepth = 0

        elif (i == len(random_TrainingList)-1):
            featureBatchArray = np.array(np.asarray(tmpFeatureList))
            labelsBatchArray = np.array(np.asarray(tmpLabelList))

            featureBatchArray = featureBatchArray.reshape(npArrayDepth, 32400)
            labelsBatchArray = labelsBatchArray.reshape(npArrayDepth, 4)

            #print(labelsBatchArray.shape)

            batchList.append((featureBatchArray, labelsBatchArray))
            tmpFeatureList = []
            tmpLabelList = []
            npArrayDepth = 0

    return batchList

def createEvalList(rawEvalList):
    npArrayDepth = 0
    evalTuple = ()
    tmpFeatureList = []
    tmpLabelList = []

    for i in range(1, len(rawEvalList) + 1):
        npArrayDepth += 1
        tmpFeatureList.append(rawEvalList[i - 1][0])
        tmpLabelList.append(rawEvalList[i - 1][1])

    featureEvalArray = np.array(tmpFeatureList)
    labelsEvalArray = np.array(tmpLabelList)

    featureEvalArray = featureEvalArray.reshape(npArrayDepth, 32400)
    labelsEvalArray = labelsEvalArray.reshape(npArrayDepth, 4)

    evalTuple = (featureEvalArray, labelsEvalArray)

    return evalTuple



### Graph definition ###
x = tf.placeholder(tf.float32, shape=[None, 108 * 300]) # input vectors
x_4DTensor = tf.reshape(x, shape=[-1, 108, 300, 1]) # input vecotrs as a 4D-Matrix

y_ = tf.placeholder(tf.float32, shape=[None, 4]) # gold standard labels; 1hot-vectors

###
# L1 = Layer 1
# 2WC = Two-Word-Context
###
filterNumber = 20
### Two-Word-Context
W_conv_L1_2WC = weightVariable([2, 300, 1, filterNumber])
b_conv_L1_2WC = biasVariable([1])

h_conv_L1_2WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_2WC) + b_conv_L1_2WC)
h_pool_L1_2WC = maxPool100x1(h_conv_L1_2WC, 107)

### Three-Word-Context
W_conv_L1_3WC = weightVariable([3, 300, 1, filterNumber])
b_conv_L1_3WC = biasVariable([1])

h_conv_L1_3WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_3WC) + b_conv_L1_3WC)
h_pool_L1_3WC = maxPool100x1(h_conv_L1_3WC, 106)

### Four-Word-Context
W_conv_L1_4WC = weightVariable([4, 300, 1, filterNumber])
b_conv_L1_4WC = biasVariable([1])

h_conv_L1_4WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_4WC) + b_conv_L1_4WC)
h_pool_L1_4WC = maxPool100x1(h_conv_L1_4WC, 105)

### Concatenate the pooling outputs to get the feature vector
outputTensor_L1 = tf.concat([h_pool_L1_2WC, h_pool_L1_3WC, h_pool_L1_4WC], 1)
# Reshape to 2D tensor
outputTensor_L1_2D = tf.reshape(outputTensor_L1, [-1, 60])

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

### Softmax Output, loss-function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) # (Goldstandard, Output)

### Training
learningRate = 1e-4 # changed Learning Rate R.K. before 0.1
train_Step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### Session ###
numEpoch = 10

# Configure how many threads are used for batch processing
config = tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    random_TrainingList = deepcopy(trainingList)

    # reshapes the feature-matrix into a vector format
    # reshapes every 1hot-vector (labels) to a 2D shape
    for i in range(len(random_TrainingList)):
        random_TrainingList[i][0] = random_TrainingList[i][0].reshape(1, 32400)
        random_TrainingList[i][1] = random_TrainingList[i][1].reshape((1, 4))

    for epoch in range(numEpoch+1):
        shuffle(random_TrainingList)
        epochAccuracyList = []
        batchList = []

        # Batch processing
        batchList = createBatchList(random_TrainingList, 100)

        for tupleBatch in batchList:
            feature_Matrix = tupleBatch[0]
            labels = tupleBatch[1]
            batch = (feature_Matrix, labels)

            training_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_Prob: 0.75}) # changed dropoutRate R.K. before: 0.5

            train_Step.run(feed_dict={x: batch[0], y_: batch[1], keep_Prob: 0.5})
            epochAccuracyList.append(training_accuracy)

        if epoch % 1 == 0:
            print('step %d, training accuracy %g' % (epoch, np.mean(epochAccuracyList)))

    evaluationTuple = createEvalList(evaluationList)
    print('test accuracy %g' % accuracy.eval(feed_dict={x: evaluationTuple[0], y_: evaluationTuple[1], keep_Prob: 1.0}))

# todo Implement a function to time the duration of training
# todo Save accuarcy, duration etc. to an output file
# todo Implement TensorBoard
