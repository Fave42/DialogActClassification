#!/bin/bash

"""
@created: 15.01.2018
@authors: Jens Beck, Fabian Fey, Richard Kollotzek


The script is stored on the server under the following path:
/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources

Usage: python2.7 CNN.py >>> log.txt
!!!!!!!!! Server currently only supports TensorFlow python2.7 !!!!!!!!!
"""

### Imports ###
import tensorflow as tf
import cPickle as pickle
from random import shuffle
from copy import deepcopy
import numpy as np
import time
import datetime

evalFrequency = 1       # Every odd step
numEpoch = 20           # Number of Epochs for training
numCPUs = 10            # Number of CPU's to be used
filterNumber2WC = 10    # Number of filters for 2-Word-Context
filterNumber3WC = 10    # Number of filters for 3-Word-Context
filterNumber4WC = 10    # Number of filters for 4-Word-Context
activationFunction = "Relu"
lossFunction = "Cross Entropy"
dropout = 0.5
optimizerFunction = "AdamOptimizer"
typeOfCNN = "CNN + 1 Fully-Connected-Layer"

logFileTmp = ""

overallTime = time.time()

# Server Paths
pathTraining = "NN_Input_Files/trainData_3-5WordContext_prot2.pickle"
pathEvaluation = "NN_Input_Files/devData_3-5WordContext_prot2.pickle"

print("### Importing Training and Evaluation Data! ###")
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
### Two-Word-Context
with tf.name_scope("CL1_Weights_2WordContext"):
    W_conv_L1_2WC = weightVariable([2, 300, 1, filterNumber2WC])
with tf.name_scope("CL1_Bias_2WordContext"):
    b_conv_L1_2WC = biasVariable([1])

with tf.name_scope("CL1_HiddenLayer_2WordContext"):
    h_conv_L1_2WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_2WC) + b_conv_L1_2WC)
with tf.name_scope("CL1_MaxPooling_2WordContext"):
    h_pool_L1_2WC = maxPool100x1(h_conv_L1_2WC, 107)

### Three-Word-Context
with tf.name_scope("CL1_Weights_3WordContext"):
    W_conv_L1_3WC = weightVariable([3, 300, 1, filterNumber3WC])
with tf.name_scope("CL1_Bias_3WordContext"):
    b_conv_L1_3WC = biasVariable([1])

with tf.name_scope("CL1_HiddenLayer_3WordContext"):
    h_conv_L1_3WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_3WC) + b_conv_L1_3WC)
with tf.name_scope("CL1_MaxPooling_3WordContext"):
    h_pool_L1_3WC = maxPool100x1(h_conv_L1_3WC, 106)

### Four-Word-Context
with tf.name_scope("CL1_Weights_4WordContext"):
    W_conv_L1_4WC = weightVariable([4, 300, 1, filterNumber4WC])
with tf.name_scope("CL1_Bias_4WordContext"):
    b_conv_L1_4WC = biasVariable([1])

with tf.name_scope("CL1_HiddenLayer_4WordContext"):
    h_conv_L1_4WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_4WC) + b_conv_L1_4WC)
with tf.name_scope("CL1_MaxPooling_4WordContext"):
    h_pool_L1_4WC = maxPool100x1(h_conv_L1_4WC, 105)

# Concatenate the pooling outputs to get the feature vector
with tf.name_scope("L1_OutputTensor"):
    outputTensor_L1 = tf.concat([h_pool_L1_2WC, h_pool_L1_3WC, h_pool_L1_4WC], 1)
# Reshape to 2D tensor
numOutputConcat = filterNumber2WC + filterNumber3WC + filterNumber4WC
with tf.name_scope("L1_OutputTensor_2D"):
    outputTensor_L1_2D = tf.reshape(outputTensor_L1, [-1, numOutputConcat])

# First Fully Connected Layer
# with tf.name_scope("FCL2_Weights"):
#     W_FC_L2 = weightVariable([numOutputConcat, numOutputConcat * 2])
# with tf.name_scope("FCL2_Bias"):
#     b_FC_L2 = biasVariable([numOutputConcat * 2])
#
# with tf.name_scope("FCL2_HiddenLayer"):
#     h_FC_L2 = tf.nn.relu(tf.matmul(outputTensor_L1_2D, W_FC_L2) + b_FC_L2)

# Dropout percentage
keep_Prob = tf.placeholder(tf.float32)
h_FC_L2_drop = tf.nn.dropout(outputTensor_L1_2D, keep_Prob)

# Second Fully Connected Layer
with tf.name_scope("FCL3_Weights"):
    W_FC_L2 = weightVariable([numOutputConcat, 4])
with tf.name_scope("FCL3_Bias"):
    b_FC_L2 = biasVariable([4])

y = tf.nn.relu(tf.matmul(h_FC_L2_drop, W_FC_L2) + b_FC_L2)

# Softmax Output, loss-function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) # (Goldstandard, Output)

# Training
learningRate = 1e-4 # changed Learning Rate R.K. before 0.1
# Optimizer
#train_Step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start the session
# Configure how many threads are used for batch processing
config = tf.ConfigProto(intra_op_parallelism_threads=numCPUs, inter_op_parallelism_threads=numCPUs)

# Dump for logfile
logFileTmp += "##########\n"
logFileTmp += "Type of CNN: " + str(typeOfCNN) + "\n"
logFileTmp += "Number of epochs: " + str(numEpoch) + "\n"
logFileTmp += "Number of filters: " + str(filterNumber4WC) + "\n"
logFileTmp += "Learning Rate: " + str(learningRate) + "\n"
logFileTmp += "Activation Function: " + str(activationFunction) + "\n"
logFileTmp += "Loss Function: " + str(lossFunction) + "\n"
logFileTmp += "Dropout: " + str(dropout) + "\n"
logFileTmp += "Optimizer: " + str(optimizerFunction) + "\n"
logFileTmp += "##########\n"


print("Starting Training...")
start_time = time.time()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    random_TrainingList = deepcopy(trainingList)

    # reshapes the feature-matrix into a vector format
    # reshapes every 1hot-vector (labels) to a 2D shape
    for i in range(len(random_TrainingList)):
        random_TrainingList[i][0] = random_TrainingList[i][0].reshape(1, 32400)
        random_TrainingList[i][1] = random_TrainingList[i][1].reshape((1, 4))

    # Tensorboard integration
    training_accuracy = 0
    #tf.summary.scalar("loss_function", loss)
    tf.summary.scalar("learning_rate", learningRate)
    #tf.summary.histogram("Accuracy", training_accuracy)
    #tf.summary.histogram("train_prediction", train_prediction)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources/log", sess.graph)

    for epoch in range(numEpoch):
        print("Current epoch " + str(epoch))
        # Shuffle the traininglist
        shuffle(random_TrainingList)
        epochAccuracyList = []
        batchList = []

        # Batch processing
        batchList = createBatchList(random_TrainingList, 100)

        for tupleBatch in batchList:
            feature_Matrix = tupleBatch[0]
            labels = tupleBatch[1]
            batch = (feature_Matrix, labels)

            # metadata for tensorboard
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            training_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_Prob: dropout}) # changed dropoutRate R.K. before: 0.5

            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict={x: batch[0], y_: batch[1], keep_Prob: dropout})

            #summary, lr = train_Step.run([merged, learningRate], feed_dict={x: batch[0], y_: batch[1], keep_Prob: 0.75}, options=run_options, run_metadata=run_metadata)

            elapsed_time = time.time() - start_time
            start_time = time.time()
            #epochAccuracyList.append(training_accuracy)

        # Evaluation Frequency
        if epoch % evalFrequency == 0:
            summary = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_Prob: dropout},
                                         options=run_options, run_metadata=run_metadata)

            print('step %d, training accuracy %g, learning rate %f, %f ms' % (epoch, training_accuracy,
                                                                              learningRate, 1000 * elapsed_time))
            writer.add_run_metadata(run_metadata, 'step %d' % epoch)
            writer.add_summary(summary, epoch)
            print("Adding run metadata for epoch " + str(epoch))

            logFileTmp += 'step %d, training accuracy %g, learning rate %f, %f ms\n' % (epoch, training_accuracy,
                                                                              learningRate, 1000 * elapsed_time)
            logFileTmp += "####\n"

    evaluationTuple = createEvalList(evaluationList)
    overallEndTime = (time.time() - overallTime)/60

    testAccuracy = accuracy.eval(feed_dict={x: evaluationTuple[0], y_: evaluationTuple[1], keep_Prob: 1.0})
    print('test accuracy %g' % testAccuracy)
    print("The program was executed in " + str(overallEndTime) + " minutes")

    logFileTmp += "########\n"
    logFileTmp += "Test Accuracy: " + str(testAccuracy) + "\n"
    logFileTmp += "The program was executed in " + str(overallEndTime) + " minutes\n"

    savePath = "log/terminal_logfiles/" + str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M")) + ".txt"
    with open(savePath,'wb') as saveFile:
        saveFile.write(logFileTmp)

# todo More TensorBoard