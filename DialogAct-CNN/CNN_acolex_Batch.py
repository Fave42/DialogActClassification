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

### Tunable Variables
numEpoch = 20                    # Number of Epochs for training
trainableEmbeddings = True
activationFunction = "TanH"     #"CNN = tanh + FCL = Relu"
lossFunction = "Hinge-Loss"
learningRate = 0.05
dropout = 0.50
optimizerFunction = "Stochastic Gradient Descent"
filterNumberMFCC = 20    # Number of filters for the MFCC features
mfccFilterSize = 100

### Static Variables
batchSize = 100         # Batchsize for training
evalFrequency = 1       # Evaluation frequency (epoch % evalFrequency == 0)
numCPUs = 10            # Number of CPU's to be used
filterNumber2WC = 20    # Number of filters for 2-Word-Context
filterNumber3WC = 20    # Number of filters for 3-Word-Context
filterNumber4WC = 20    # Number of filters for 4-Word-Context
numberMFCCFeatures = 2000
typeOfCNN = "CNN + 1 Fully-Connected-Layer"


logFileTmp = ""
overallTime = time.time()

# Server Paths
# Without stopwords
pathTraining = "NN_Input_Files/trainData_acolex_Embeddings.pickle"
pathEvaluation = "NN_Input_Files/devData_acolex_Embeddings.pickle"
pathTest = "NN_Input_Files/testData_acolex_Embeddings.pickle"
pathEmbeddings = "dict/embeddingMatrix_np_acolex_full.pickle"
### Testing purposes only
# pathTraining = "NN_Input_Files/sanityTestFiles/trainData_acolex_Embeddings_short.pickle"
# pathEvaluation = "NN_Input_Files/sanityTestFiles/devData_acolex_Embeddings_short.pickle"

# With stopwords
#pathTraining = "NN_Input_Files/trainData_4_100_fsw.pickle"
#pathEvaluation = "NN_Input_Files/devData_4_100_fsw.pickle"

print("### Importing Training, Evaluation and Embedding Data! ###")
trainingList = pickle.load(open(pathTraining, "rb"))
evaluationList = pickle.load(open(pathEvaluation, "rb"))
testList = pickle.load(open(pathTest, "rb"))
embeddingInputs = pickle.load(open(pathEmbeddings, "rb"))
print("\t---> Done with importing!")


### Functions ###
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0, seed=1)
    return tf.Variable(initial)


def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 300, 1], padding='VALID')


def conv2d_MFCC(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 13, 1, 1], padding='VALID')


def maxPool100x1(x, kernelDepth):
    return tf.nn.max_pool(x, ksize=[1, kernelDepth, 1, 1],
                          strides=[1, 1, 1, 1], padding='VALID')


def maxPool_MFCC(x, kernelDepth):
    return tf.nn.max_pool(x, ksize=[1, 1, kernelDepth, 1],
                          strides=[1, 1, 100, 1], padding='VALID')


### Creates and returns a batchlist with the following format
#   batchList = [ (1Batch-Features, 1Batch-MFCC, 1Batch-Labels), (2Batch-Features, 2Batch-MFCC, 2Batch-Labels), ...]
###
def createBatchList(random_TrainingList, batchSize):
    npArrayDepth = 0
    batchList = []
    tmpFeatureList = []
    tmpMFCCList = []
    tmpLabelList = []

    for i in range(1, len(random_TrainingList) + 1):
        npArrayDepth += 1
        tmpFeatureList.append(random_TrainingList[i - 1][0])
        tmpMFCCList.append(random_TrainingList[i - 1][1])
        tmpLabelList.append(random_TrainingList[i - 1][2])

        if (i % batchSize == 0) and (i != 0):
            featureBatchArray = np.array(tmpFeatureList)
            mfccBatchArray = np.array(tmpMFCCList)
            labelsBatchArray = np.array(tmpLabelList)

            featureBatchArray = featureBatchArray.reshape(npArrayDepth, 100)
            mfccBatchArray = mfccBatchArray.reshape(npArrayDepth, numberMFCCFeatures*13)
            labelsBatchArray = labelsBatchArray.reshape(npArrayDepth, 4)

            # print(labelsBatchArray.shape)

            batchList.append((featureBatchArray, mfccBatchArray, labelsBatchArray))
            tmpFeatureList = []
            tmpMFCCList = []
            tmpLabelList = []
            npArrayDepth = 0

        elif (i == len(random_TrainingList) - 1):
            featureBatchArray = np.array(np.asarray(tmpFeatureList))
            mfccBatchArray = np.array(np.asarray(tmpMFCCList))
            labelsBatchArray = np.array(np.asarray(tmpLabelList))

            featureBatchArray = featureBatchArray.reshape(npArrayDepth, 100)
            mfccBatchArray = mfccBatchArray.reshape(npArrayDepth, numberMFCCFeatures*13)
            labelsBatchArray = labelsBatchArray.reshape(npArrayDepth, 4)

            # print(labelsBatchArray.shape)

            batchList.append((featureBatchArray, mfccBatchArray, labelsBatchArray))
            tmpFeatureList = []
            tmpMFCCList = []
            tmpLabelList = []
            npArrayDepth = 0

    return batchList


def createEvalList(rawEvalList):
    npArrayDepth = 0
    evalTriple = ()
    tmpFeatureList = []
    tmpMFCCList = []
    tmpLabelList = []

    for i in range(1, len(rawEvalList) + 1):
        npArrayDepth += 1
        tmpFeatureList.append(rawEvalList[i - 1][0])
        tmpMFCCList.append(rawEvalList[i - 1][1])
        tmpLabelList.append(rawEvalList[i - 1][2])

    featureEvalArray = np.array(tmpFeatureList)
    mfccEvalArray = np.array(tmpMFCCList)
    labelsEvalArray = np.array(tmpLabelList)

    featureEvalArray = featureEvalArray.reshape(npArrayDepth, 100)
    mfccEvalArray = mfccEvalArray.reshape(npArrayDepth, numberMFCCFeatures*13)
    labelsEvalArray = labelsEvalArray.reshape(npArrayDepth, 4)

    evalTriple = (featureEvalArray, mfccEvalArray ,labelsEvalArray)

    return evalTriple


### Graph definition ###
x = tf.placeholder(tf.float32, shape=[None, 100])  # lexical input vectors
x_mfcc = tf.placeholder(tf.float32, shape=[None, numberMFCCFeatures*13])  # mfcc input vectors
y_ = tf.placeholder(tf.float32, shape=[None, 4])  # gold standard labels; 1hot-vectors

x_MFCC_4DTensor = tf.reshape(x_mfcc, shape=[-1, 13, numberMFCCFeatures, 1])

###
# L1 = Layer 1
# 2WC = Two-Word-Context
###

### Layer 0
# vocab_size = 10017 # For normal dataset
vocab_size = 11825 # For "full" dataset
embedding_dim = 300
with tf.name_scope("Embedding_Layer"):
    with tf.name_scope("Embedding_Matrix"):
        embedding_Matrix = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_dim], minval=-1, maxval=1),
                        trainable=trainableEmbeddings)
    with tf.name_scope("Embedding_Placeholder"):
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    with tf.name_scope("Embedding_init"):
        embedding_init = embedding_Matrix.assign(embedding_placeholder)
    with tf.name_scope("Embedded_Input"):
        x_embedded = tf.nn.embedding_lookup(embedding_Matrix, tf.cast(x, dtype=tf.int32))
    with tf.name_scope("Embedded_4D_Tensor"):
        x_4DTensor = tf.reshape(x_embedded, shape=[-1, 100, 300, 1])  # input vecotrs as a 4D-Matrix


with tf.name_scope("generate_padded_matrices"):
    paddings = tf.constant([[0,0],[3, 3],[0,0],[0, 0]])
    x4DTensor_padded = tf.pad(x_4DTensor, paddings, "CONSTANT")


### MFCC-Layer
with tf.name_scope("MFCC_Layer"):
    with tf.name_scope("MFCC_Weights"):
        W_conv_MFCC_L0 = weightVariable([13, mfccFilterSize, 1, filterNumberMFCC])
    with tf.name_scope("MFCC_Bias"):
        b_conv_MFCC_L0 = biasVariable([1])

    with tf.name_scope("MFCC_CL1_HiddenLayer"):
        # h_conv_MFCC_L0 = tf.nn.relu(conv2d_MFCC(x_MFCC_4DTensor, W_conv_MFCC_L0) + b_conv_MFCC_L0) ### activation function ReLu
        h_conv_MFCC_L0 = tf.tanh(conv2d_MFCC(x_MFCC_4DTensor, W_conv_MFCC_L0) + b_conv_MFCC_L0) ### activation function TanH
        # h_conv_MFCC_L0 = tf.nn.sigmoid(conv2d_MFCC(x_MFCC_4DTensor, W_conv_MFCC_L0) + b_conv_MFCC_L0) ### activation function sigmoid
    with tf.name_scope("MFCC_CL1_MaxPooling"):
        poolingWindow = numberMFCCFeatures - (mfccFilterSize - 1)
        h_pool_MFCC_L0 = maxPool_MFCC(h_conv_MFCC_L0, poolingWindow)  # with 1951 input --> 40 anoutput
        h_pool_MFCC_L0_3D = tf.reshape(h_pool_MFCC_L0, shape=[1, filterNumberMFCC, -1])


### Layer 1
### Two-Word-Context
with tf.name_scope("Two_Word_Context"):
    with tf.name_scope("CL1_Weights"):
        W_conv_L1_2WC = weightVariable([2, 300, 1, filterNumber2WC])
    with tf.name_scope("CL1_Bias"):
        b_conv_L1_2WC = biasVariable([1])

    with tf.name_scope("CL1_HiddenLayer"):
        # h_conv_L1_2WC = tf.nn.relu(conv2d(x4DTensor_padded, W_conv_L1_2WC) + b_conv_L1_2WC) ### activation function ReLu
        h_conv_L1_2WC = tf.tanh(conv2d(x4DTensor_padded, W_conv_L1_2WC) + b_conv_L1_2WC) ### activation function TanH
        # h_conv_L1_2WC = tf.nn.sigmoid(conv2d(x4DTensor_padded, W_conv_L1_2WC) + b_conv_L1_2WC) ### activation function sigmoid
    with tf.name_scope("CL1_MaxPooling"):
        h_pool_L1_2WC = maxPool100x1(h_conv_L1_2WC, 105)

### Three-Word-Context
with tf.name_scope("Three_Word_Context"):
    with tf.name_scope("CL1_Weights"):
        W_conv_L1_3WC = weightVariable([3, 300, 1, filterNumber3WC])
    with tf.name_scope("CL1_Bias"):
        b_conv_L1_3WC = biasVariable([1])

    with tf.name_scope("CL1_HiddenLayer"):
        # h_conv_L1_3WC = tf.nn.relu(conv2d(x4DTensor_padded, W_conv_L1_3WC) + b_conv_L1_3WC) ### activation function ReLu
        h_conv_L1_3WC = tf.tanh(conv2d(x4DTensor_padded, W_conv_L1_3WC) + b_conv_L1_3WC) ### activation function TanH
        # h_conv_L1_3WC = tf.nn.sigmoid(conv2d(x4DTensor_padded, W_conv_L1_3WC) + b_conv_L1_3WC) ### activation function sigmoid
    with tf.name_scope("CL1_MaxPooling"):
        h_pool_L1_3WC = maxPool100x1(h_conv_L1_3WC, 104)

### Four-Word-Context
with tf.name_scope("Four_Word_Context"):
    with tf.name_scope("CL1_Weights"):
        W_conv_L1_4WC = weightVariable([4, 300, 1, filterNumber4WC])
    with tf.name_scope("CL1_Bias"):
        b_conv_L1_4WC = biasVariable([1])

    with tf.name_scope("CL1_HiddenLayer"):
        # h_conv_L1_4WC = tf.nn.relu(conv2d(x4DTensor_padded, W_conv_L1_4WC) + b_conv_L1_4WC) ### activation function ReLu
        h_conv_L1_4WC = tf.tanh(conv2d(x4DTensor_padded, W_conv_L1_4WC) + b_conv_L1_4WC) ### activation function TanH
        # h_conv_L1_4WC = tf.nn.sigmoid(conv2d(x4DTensor_padded, W_conv_L1_4WC) + b_conv_L1_4WC) ### activation function sigmoid
    with tf.name_scope("CL1_MaxPooling"):
        h_pool_L1_4WC = maxPool100x1(h_conv_L1_4WC, 103)

# with tf.name_scope("reshape_tensors_into_2D"):
#     h_pool_L1_2D_2WC = tf.reshape(h_pool_L1_2WC, shape=[1,filterNumber2WC,-1])
#     h_pool_L1_2D_3WC = tf.reshape(h_pool_L1_3WC, shape=[1,filterNumber3WC,-1])
#     h_pool_L1_2D_4WC = tf.reshape(h_pool_L1_4WC, shape=[1,filterNumber4WC,-1])

# Concatenate the pooling outputs to get the feature vector
with tf.name_scope("L1_OutputTensor"):
    outputTensor_L1 = tf.concat([h_pool_L1_2WC, h_pool_L1_3WC, h_pool_L1_4WC, h_pool_MFCC_L0], 1)
    # outputTensor_L1 = tf.concat([h_pool_L1_2D_2WC, h_pool_L1_2D_3WC, h_pool_L1_2D_4WC, h_pool_MFCC_L0_3D], 1)
# Reshape to 2D tensor
with tf.name_scope("Concatination_Dimensions"):
    # mfccPoolingOutput_1F = math.ceil((numberMFCCFeatures - (mfccFilterSize - 1)) / float(poolingWindow))
    mfccPoolingOutput_1F = (numberMFCCFeatures - (mfccFilterSize - 1)) / poolingWindow
    numOutputConcat = filterNumber2WC + filterNumber3WC + filterNumber4WC + (filterNumberMFCC * mfccPoolingOutput_1F)
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

with tf.name_scope("Final_Linear_Function"):
    y = tf.matmul(h_FC_L2_drop, W_FC_L2) + b_FC_L2

# Softmax Output, loss-function
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # (Goldstandard, Output); Cross Entropy; reduce_mean
# loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # (Goldstandard, Output); Cross Entropy; reduce_sum
loss = tf.losses.hinge_loss(labels=y_, logits=y, weights=1.0)

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
logFileTmp += "Number of filters 2WC: " + str(filterNumber2WC) + "\n"
logFileTmp += "Number of filters 3WC: " + str(filterNumber3WC) + "\n"
logFileTmp += "Number of filters 4WC: " + str(filterNumber4WC) + "\n"
logFileTmp += "Number of filters MFCC: " + str(filterNumberMFCC) + "\n"
logFileTmp += "Filter size MFCC: " + str(mfccFilterSize) + "\n"
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

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(embedding_init, feed_dict={embedding_placeholder: embeddingInputs})
    stepCount = 0


#    values2 = sess.run(W_conv_L1_2WC)
#    print("pool_Layer_2WC:")
#    print(values2)
#    print(tf.shape(values2))
#    print("Weight_3WC:")


    random_TrainingList = deepcopy(trainingList)

    # print(random_TrainingList[0][1])
    # print("\trandomTrainingList shape: " + str(random_TrainingList[0][1].shape))
    # reshapes the feature-matrix into a vector format
    # reshapes every 1hot-vector (labels) to a 2D shape
    for i in range(len(random_TrainingList)):
        random_TrainingList[i][0] = random_TrainingList[i][0].reshape((1, 100))   # text
        random_TrainingList[i][1] = random_TrainingList[i][1].flatten() #reshape((1, numberMFCCFeatures))   # mfcc
        random_TrainingList[i][2] = random_TrainingList[i][2].reshape((1, 4))   # gold-standard
    # print(random_TrainingList[0][1].shape)

    # Tensorboard integration
    training_accuracy = 0
    tf.summary.scalar("learning_rate", learningRate)
    tf.summary.histogram("training_accuracy", training_accuracy)
    tf.summary.scalar("loss_function", loss)
    tf.summary.histogram("accuracy", accuracy)
    # tf.summary.histogram("bias_FCL2", b_FC_L2)
    # tf.summary.histogram("bias_FCL1_2WC", b_conv_L1_2WC)
    # tf.summary.histogram("bias_FCL1_3WC", b_conv_L1_3WC)
    # tf.summary.histogram("bias_FCL1_4WC", b_conv_L1_4WC)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logPath, sess.graph)

    for epoch in range(numEpoch):
        start_time = time.time()
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
            mfcc_feature_Matrix = tupleBatch[1]
            labels = tupleBatch[2]
            batch = (feature_Matrix, mfcc_feature_Matrix, labels)

            # metadata for tensorboard
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # computes the actual accuracy for the current batch.
            training_accuracy = accuracy.eval(
                feed_dict={x: batch[0], x_mfcc: batch[1], y_: batch[2], keep_Prob: 1.0})

            # Run the optimizer to update weights.
            summary, l, train = sess.run([merged, loss, train_Step], feed_dict={x: batch[0], x_mfcc: batch[1], y_: batch[2], keep_Prob: dropout})

            writer.add_run_metadata(run_metadata, 'step %d' % stepCount)
            writer.add_summary(summary, stepCount)
            # print("Adding run metadata for epoch " + str(stepCount))

            stepCount += 1

        elapsed_time = time.time() - start_time
            # epochAccuracyList.append(training_accuracy)
            # epochLossList.append(l)

        # Evaluation output for direct user controll.
        if epoch % evalFrequency == 0:
            # epochAvgAccuracy = np.mean(epochAccuracyList)
            # epochAvgLoss = np.mean(epochLossList)

            print('\t- step %d, training accuracy %g, learning rate %f, loss %f, %f s' % (epoch, training_accuracy,
                                                                                          learningRate, l,
                                                                                          elapsed_time))
            # writer.add_run_metadata(run_metadata, 'step %d' % epoch)
            # writer.add_summary(summary, epoch)
            # print("Adding run metadata for epoch " + str(epoch))

            logFileTmp += 'step %d, training accuracy %g, loss %f, learning rate %f, %f s\n' % (epoch, training_accuracy,
                                                                                             l, learningRate,
                                                                                             elapsed_time)
            logFileTmp += "####\n"

    overallEndTime = (time.time() - overallTime) / 60

    evaluationTriple = createEvalList(evaluationList)
    devAccuracy = accuracy.eval(feed_dict={x: evaluationTriple[0], x_mfcc: evaluationTriple[1], y_: evaluationTriple[2], keep_Prob: 1.0})

    testTriple = createEvalList(testList)
    testAccuracy = accuracy.eval(feed_dict={x: testTriple[0], x_mfcc: testTriple[1], y_: testTriple[2], keep_Prob: 1.0})

    print('dev accuracy %g' % devAccuracy)
    print('test accuracy %g' % testAccuracy)
    print("The program was executed in " + str(overallEndTime) + " minutes")

    logFileTmp += "########\n"
    logFileTmp += "Dev Accuracy: " + str(devAccuracy) + "\n"
    logFileTmp += "Test Accuracy: " + str(testAccuracy) + "\n"
    logFileTmp += "The program was executed in " + str(overallEndTime) + " minutes\n"

    savePath = logPath + "/" + programStartTime +".txt"
    with open(savePath, 'wb') as saveFile:
        saveFile.write(logFileTmp)

writer.close()

# todo Add adaptive learningrate
# todo Add config file integration