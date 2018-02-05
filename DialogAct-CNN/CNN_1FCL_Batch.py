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
import time
import datetime
import os


batchSize = 100         # Batchsize for training
evalFrequency = 1       # Evaluation frequency (epoch % evalFrequency == 0)
numEpoch = 3            # Number of Epochs for training
numCPUs = 10            # Number of CPU's to be used
filterNumber2WC = 10    # Number of filters for 2-Word-Context
filterNumber3WC = 10    # Number of filters for 3-Word-Context
filterNumber4WC = 10    # Number of filters for 4-Word-Context
trainableEmbeddings = False
activationFunction = "CNN = tanh + FCL = Relu"
lossFunction = "Cross Entropy"
learningRate = 0.01
dropout = 1.0
optimizerFunction = "Stochastic Gradient Descent"
typeOfCNN = "CNN + 1 Fully-Connected-Layer"

logFileTmp = ""

overallTime = time.time()

# Server Paths
# Without stopwords
pathTraining = "NN_Input_Files/trainData_Embeddings.pickle"
pathEvaluation = "NN_Input_Files/devData_Embeddings.pickle"
pathEmbeddings = "dict/embeddingMatrix_np.pickle"
# With stopwords
#pathTraining = "NN_Input_Files/trainData_4_100_fsw.pickle"
#pathEvaluation = "NN_Input_Files/devData_4_100_fsw.pickle"

print("### Importing Training, Evaluation and Embedding Data! ###")
trainingList = pickle.load(open(pathTraining, "rb"))
evaluationList = pickle.load(open(pathEvaluation, "rb"))
embeddingInputs = pickle.load(open(pathEmbeddings, "rb"))
print("\t---> Done with importing!")


### Functions ###
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)


def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 300, 1], padding='VALID')


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

    for i in range(1, len(random_TrainingList) + 1):
        npArrayDepth += 1
        tmpFeatureList.append(random_TrainingList[i - 1][0])
        tmpLabelList.append(random_TrainingList[i - 1][1])

        if (i % batchSize == 0) and (i != 0):
            featureBatchArray = np.array(tmpFeatureList)
            labelsBatchArray = np.array(tmpLabelList)

            featureBatchArray = featureBatchArray.reshape(npArrayDepth, 100)
            labelsBatchArray = labelsBatchArray.reshape(npArrayDepth, 4)

            # print(labelsBatchArray.shape)

            batchList.append((featureBatchArray, labelsBatchArray))
            tmpFeatureList = []
            tmpLabelList = []
            npArrayDepth = 0

        elif (i == len(random_TrainingList) - 1):
            featureBatchArray = np.array(np.asarray(tmpFeatureList))
            labelsBatchArray = np.array(np.asarray(tmpLabelList))

            featureBatchArray = featureBatchArray.reshape(npArrayDepth, 100)
            labelsBatchArray = labelsBatchArray.reshape(npArrayDepth, 4)

            # print(labelsBatchArray.shape)

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
x = tf.placeholder(tf.float32, shape=[None, 100])  # input vectors
y_ = tf.placeholder(tf.float32, shape=[None, 4])  # gold standard labels; 1hot-vectors

###
# L1 = Layer 1
# 2WC = Two-Word-Context
###

### Layer 0
vocab_size = 10017
embedding_dim = 300
with tf.name_scope("Embedding_Layer"):
    with tf.name_scope("Embedding_Matrix"):
        embedding_Matrix = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                        trainable=trainableEmbeddings)
    with tf.name_scope("Embedding_Placeholder"):
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    with tf.name_scope("Embedding_init"):
        embedding_init = embedding_Matrix.assign(embedding_placeholder)
    with tf.name_scope("Embedded_Input"):
        x_embedded = tf.nn.embedding_lookup(embedding_Matrix, tf.cast(x, dtype=tf.int32))
    with tf.name_scope("Embedded_4D_Tensor"):
        x_4DTensor = tf.reshape(x_embedded, shape=[-1, 100, 300, 1])  # input vecotrs as a 4D-Matrix
print("Shape of x_4DTensor: " + str(x_4DTensor.shape))

### Layer 1
### Two-Word-Context
with tf.name_scope("Two_Word_Context"):
    with tf.name_scope("CL1_Weights"):
        W_conv_L1_2WC = weightVariable([2, 300, 1, filterNumber2WC])
    with tf.name_scope("CL1_Bias"):
        b_conv_L1_2WC = biasVariable([1])

    with tf.name_scope("CL1_HiddenLayer"):
    #    h_conv_L1_2WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_2WC) + b_conv_L1_2WC) ### activation function ReLu
        h_conv_L1_2WC = tf.tanh(conv2d(x_4DTensor, W_conv_L1_2WC) + b_conv_L1_2WC) ### activation function TanH
        print("Shape of h_conv_L1_2WC: " + str(h_conv_L1_2WC.shape))
    #	 h_conv_L1_2WC = tf.nn.sigmoid(conv2d(x_4DTensor, W_conv_L1_2WC) + b_conv_L1_2WC) ### activation function sigmoid
    with tf.name_scope("CL1_MaxPooling"):
        h_pool_L1_2WC = maxPool100x1(h_conv_L1_2WC, 99)

### Three-Word-Context
with tf.name_scope("Three_Word_Context"):
    with tf.name_scope("CL1_Weights"):
        W_conv_L1_3WC = weightVariable([3, 300, 1, filterNumber3WC])
    with tf.name_scope("CL1_Bias"):
        b_conv_L1_3WC = biasVariable([1])

    with tf.name_scope("CL1_HiddenLayer"):
    #    h_conv_L1_3WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_3WC) + b_conv_L1_3WC) ### activation function ReLu
        h_conv_L1_3WC = tf.tanh(conv2d(x_4DTensor, W_conv_L1_3WC) + b_conv_L1_3WC) ### activation function TanH
    #    h_conv_L1_3WC = tf.nn.sigmoid(conv2d(x_4DTensor, W_conv_L1_3WC) + b_conv_L1_3WC) ### activation function sigmoid
    with tf.name_scope("CL1_MaxPooling"):
        h_pool_L1_3WC = maxPool100x1(h_conv_L1_3WC, 98)

### Four-Word-Context
with tf.name_scope("Four_Word_Context"):
    with tf.name_scope("CL1_Weights"):
        W_conv_L1_4WC = weightVariable([4, 300, 1, filterNumber4WC])
    with tf.name_scope("CL1_Bias"):
        b_conv_L1_4WC = biasVariable([1])

    with tf.name_scope("CL1_HiddenLayer"):
    #    h_conv_L1_4WC = tf.nn.relu(conv2d(x_4DTensor, W_conv_L1_4WC) + b_conv_L1_4WC) ### activation function ReLu
        h_conv_L1_4WC = tf.tanh(conv2d(x_4DTensor, W_conv_L1_4WC) + b_conv_L1_4WC) ### activation function TanH
    #    h_conv_L1_4WC = tf.nn.sigmoid(conv2d(x_4DTensor, W_conv_L1_4WC) + b_conv_L1_4WC) ### activation function sigmoid
    with tf.name_scope("CL1_MaxPooling"):
        h_pool_L1_4WC = maxPool100x1(h_conv_L1_4WC, 97)

# Concatenate the pooling outputs to get the feature vector
with tf.name_scope("L1_OutputTensor"):
    outputTensor_L1 = tf.concat([h_pool_L1_2WC, h_pool_L1_3WC, h_pool_L1_4WC], 1)
# Reshape to 2D tensor
with tf.name_scope("Concatination_Dimensions"):
    numOutputConcat = filterNumber2WC + filterNumber3WC + filterNumber4WC
with tf.name_scope("L1_OutputTensor_2D"):
    outputTensor_L1_2D = tf.reshape(outputTensor_L1, [-1, numOutputConcat])

# Dropout percentage
keep_Prob = tf.placeholder(tf.float32)
h_FC_L2_drop = tf.nn.dropout(outputTensor_L1_2D, keep_Prob)

# Second Fully Connected Layer
with tf.name_scope("FCL2_Weights"):
    W_FC_L2 = weightVariable([numOutputConcat, 4])
with tf.name_scope("FCL2_Bias"):
    b_FC_L2 = biasVariable([4])

with tf.name_scope("Activation_Function"):
    y = tf.nn.relu(tf.matmul(h_FC_L2_drop, W_FC_L2) + b_FC_L2)  ### activation function ReLu
#y = tf.tanh(tf.matmul(h_FC_L2_drop, W_FC_L2) + b_FC_L2) ### activation function TanH
#y = tf.nn.sigmoid(tf.matmul(h_FC_L2_drop, W_FC_L2) + b_FC_L2)  ### activation function sigmoid

# Softmax Output, loss-function
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # (Goldstandard, Output); Cross Entropy; reduce_mean
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # (Goldstandard, Output); Cross Entropy; reduce_sum
#loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_, logits=y, pos_weight=0.5))  # (Goldstandard, Output); Weighted Cross Entropy


# Training
# Optimizer
# train_Step = tf.train.AdamOptimizer(learningRate).minimize(loss)
train_Step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start the session
# Configure how many threads are used for batch processing
config = tf.ConfigProto(intra_op_parallelism_threads=numCPUs, inter_op_parallelism_threads=numCPUs)

# Dump for logfile
logFileTmp += "##########\n"
logFileTmp += "Used Training Data: " + str(pathTraining) + "\n"
logFileTmp += "Used Dev Data: " + str(pathEvaluation) + "\n"
logFileTmp += "Type of CNN: " + str(typeOfCNN) + "\n"
logFileTmp += "Number of epochs: " + str(numEpoch) + "\n"
logFileTmp += "Number of filters: " + str(filterNumber4WC) + "\n"
logFileTmp += "Trainable Embeddings: " + str(trainableEmbeddings) + "\n"
logFileTmp += "Batchsize: " + str(batchSize) + "\n"
logFileTmp += "Learning Rate: " + str(learningRate) + "\n"
logFileTmp += "Activation Function: " + str(activationFunction) + "\n"
logFileTmp += "Loss Function: " + str(lossFunction) + "\n"
logFileTmp += "Dropout: " + str(dropout) + "\n"
logFileTmp += "Optimizer: " + str(optimizerFunction) + "\n"
logFileTmp += "##########\n"


print("Starting Training...")
programStartTime = str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")) # stores the time when the computation starts.
logPath = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources/log/"
logPath += programStartTime

# Creates a folder in which to store all logfiles
if not os.path.exists(logPath):
    os.makedirs(logPath)

start_time = time.time()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(embedding_init, feed_dict={embedding_placeholder: embeddingInputs})

    random_TrainingList = deepcopy(trainingList)

    # reshapes the feature-matrix into a vector format
    # reshapes every 1hot-vector (labels) to a 2D shape
    for i in range(len(random_TrainingList)):
        random_TrainingList[i][0] = random_TrainingList[i][0].reshape(1, 100)
        random_TrainingList[i][1] = random_TrainingList[i][1].reshape((1, 4))

    # Tensorboard integration
    training_accuracy = 0
    tf.summary.scalar("learning_rate", learningRate)
    tf.summary.histogram("training_accuracy", training_accuracy)
    tf.summary.histogram("loss_function", loss)
    tf.summary.histogram("accuracy", accuracy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logPath, sess.graph)

    for epoch in range(numEpoch):
        print("Current epoch " + str(epoch))
        # Shuffle the traininglist
        shuffle(random_TrainingList)
        epochAccuracyList = []
        epochLossList = []
        batchList = []

        # Batch processing
        batchList = createBatchList(random_TrainingList, batchSize)

        for tupleBatch in batchList:
            # dividing data into the actual features and the gold standard one hot vectors.
            feature_Matrix = tupleBatch[0]
            labels = tupleBatch[1]
            batch = (feature_Matrix, labels)

            # metadata for tensorboard
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # computes the actual accuracy for the current batch.
            training_accuracy = accuracy.eval(
                feed_dict={x: batch[0], y_: batch[1], keep_Prob: 1.0})

            # Run the optimizer to update weights.
            summary, l, train = sess.run([merged, loss, train_Step], feed_dict={x: batch[0], y_: batch[1], keep_Prob: dropout})

            elapsed_time = time.time() - start_time
            start_time = time.time()
            epochAccuracyList.append(training_accuracy)
            epochLossList.append(l)

        # Evaluation output for direct user controll.
        if epoch % evalFrequency == 0:
            epochAvgAccuracy = np.mean(epochAccuracyList)
            epochAvgLoss = np.mean(epochLossList)

            print('step %d, epoch accuracy %g, learning rate %f, loss %f, %f s' % (epoch, epochAvgAccuracy, learningRate,
                                                                                   epochAvgLoss, 1000 * elapsed_time))
            writer.add_run_metadata(run_metadata, 'step %d' % epoch)
            writer.add_summary(summary, epoch)
            print("Adding run metadata for epoch " + str(epoch))

            logFileTmp += 'step %d, epoch accuracy %g, loss %f, learning rate %f, %f s\n' % (epoch, epochAvgAccuracy,
                                                                                             epochAvgLoss, learningRate,
                                                                                             1000 * elapsed_time)
            logFileTmp += "####\n"

    evaluationTuple = createEvalList(evaluationList)
    overallEndTime = (time.time() - overallTime) / 60

    testAccuracy = accuracy.eval(feed_dict={x: evaluationTuple[0], y_: evaluationTuple[1], keep_Prob: 1.0})

    print('test accuracy %g' % testAccuracy)
    print("The program was executed in " + str(overallEndTime) + " minutes")

    logFileTmp += "########\n"
    logFileTmp += "Test Accuracy: " + str(testAccuracy) + "\n"
    logFileTmp += "The program was executed in " + str(overallEndTime) + " minutes\n"

    savePath = logPath + "/" + programStartTime +".txt"
    with open(savePath, 'wb') as saveFile:
        saveFile.write(logFileTmp)

writer.close()

# todo More TensorBoard
