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
numEpoch = 21                    # Number of Epochs for training
activationFunction_AM = "TanH"     # TanH or Relu or Sigmoid
lossFunction = "Cross-Entropy"      # Cross-Entropy or Hinge-Loss or Mean-Squared-Error
learningRate = 0.01
dropout = 0.50
optimizerFunction = "Stochastic Gradient Descent"
filterNumberMFCC = 100    # Number of filters for the MFCC features
mfccFilterSize = 100
AM_Feature_Output_Number = 100  # Output size of AM FCL
weightSeed = 1

### Static Variables
batchSize = 100         # Batchsize for training
evalFrequency = 1       # Evaluation frequency (epoch % evalFrequency == 0)
numCPUs = 10            # Number of CPU's to be used
numberMFCCFrames = 2000
typeOfCNN = "CNN + 1 Fully-Connected-Layer"

logFileTmp = ""
overallTime = time.time()

### Checking  Hyperparameters
# Lossfunction
lossFunctionList = ["Cross-Entropy", "Hinge-Loss", "Mean-Squared-Error"]
if (lossFunction not in lossFunctionList):
    print("Lossfunction not defined!!!")
    exit()

# Activationfunction
activationFunctionList = ["TanH", "Relu", "Sigmoid"]
if (activationFunction_AM not in activationFunctionList):
    print("Activationfunction not defined!!!")
    exit()

# Server Paths
# Without stopwords
pathTraining = "NN_Input_Files/trainData_acolex_Embeddings_final.pickle"
pathEvaluation = "NN_Input_Files/devData_acolex_Embeddings_final.pickle"
pathTest = "NN_Input_Files/Test_data/testData_acolex_Embeddings_final.pickle"
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
print("\t---> Done with importing!")


### Functions ###
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0, seed=weightSeed)
    return tf.Variable(initial)


def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d_MFCC(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 13, 50, 1], padding='VALID')

def maxPool_MFCC(x, kernelDepth):
    return tf.nn.max_pool(x, ksize=[1, 1, kernelDepth, 1],
                          strides=[1, 1, 1, 1], padding='VALID')  #old config: strides=[1, 1, 100, 1]


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
            mfccBatchArray = mfccBatchArray.reshape(npArrayDepth, numberMFCCFrames * 13)
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
            mfccBatchArray = mfccBatchArray.reshape(npArrayDepth, numberMFCCFrames * 13)
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
    mfccEvalArray = mfccEvalArray.reshape(npArrayDepth, numberMFCCFrames * 13)
    labelsEvalArray = labelsEvalArray.reshape(npArrayDepth, 4)

    evalTriple = (featureEvalArray, mfccEvalArray ,labelsEvalArray)

    return evalTriple


### Graph definition ###
x_mfcc = tf.placeholder(tf.float32, shape=[None, numberMFCCFrames * 13])  # mfcc input vectors
y_ = tf.placeholder(tf.float32, shape=[None, 4])  # gold standard labels; 1hot-vectors

x_MFCC_4DTensor = tf.reshape(x_mfcc, shape=[-1, 13, numberMFCCFrames, 1])

### Acoustic Model
with tf.name_scope("Acoustic_Model"):
    with tf.name_scope("CNN_Layer"):
        with tf.name_scope("MFCC_Weights"):
            W_conv_MFCC_L0 = weightVariable([13, mfccFilterSize, 1, filterNumberMFCC])
        with tf.name_scope("MFCC_Bias"):
            b_conv_MFCC_L0 = biasVariable([1])

        with tf.name_scope("MFCC_CL1_HiddenLayer"):
            if (activationFunction_AM == "Relu"):
                h_conv_MFCC_L0 = tf.nn.relu(
                    conv2d_MFCC(x_MFCC_4DTensor, W_conv_MFCC_L0) + b_conv_MFCC_L0)  ### activation function ReLu
            elif (activationFunction_AM == "TanH"):
                h_conv_MFCC_L0 = tf.tanh(
                    conv2d_MFCC(x_MFCC_4DTensor, W_conv_MFCC_L0) + b_conv_MFCC_L0)  ### activation function TanH
            elif (activationFunction_AM == "Sigmoid"):
                h_conv_MFCC_L0 = tf.nn.sigmoid(
                    conv2d_MFCC(x_MFCC_4DTensor, W_conv_MFCC_L0) + b_conv_MFCC_L0)  ### activation function sigmoid
            else:
                print("Activationfunction not defined!!!")
                exit()

        with tf.name_scope("MFCC_CL1_MaxPooling"):
            outputdepthMFCCConv = (numberMFCCFrames / 50) - 1  #Computes the
            h_pool_MFCC_L0 = maxPool_MFCC(h_conv_MFCC_L0, outputdepthMFCCConv)

    with tf.name_scope("Fully_Connected_Layer"):
        with tf.name_scope("AM_FCL_Weights"):
            W_AM_FC = weightVariable([filterNumberMFCC, AM_Feature_Output_Number])
        with tf.name_scope("AM_FCL_Bias"):
            b_AM_FC = biasVariable([AM_Feature_Output_Number])
        with tf.name_scope("AM_Output_2D"):
            h_pool_MFCC_L0_2D = tf.reshape(h_pool_MFCC_L0, [-1, filterNumberMFCC])
        with tf.name_scope("FCL"):
            if (activationFunction_AM == "Relu"):
                AM_Output = tf.nn.relu(tf.matmul(h_pool_MFCC_L0_2D, W_AM_FC) + b_AM_FC)  ### activation function ReLu
            elif (activationFunction_AM == "TanH"):
                AM_Output = tf.nn.tanh(tf.matmul(h_pool_MFCC_L0_2D, W_AM_FC) + b_AM_FC)  ### activation function TanH
            elif (activationFunction_AM == "Sigmoid"):
                AM_Output = tf.nn.sigmoid(tf.matmul(h_pool_MFCC_L0_2D, W_AM_FC) + b_AM_FC)  ### activation function sigmoid
            else:
                print("Activationfunction AM FCL not defined!!!")
                exit()

            AM_Output_4D = tf.reshape(AM_Output, [-1, 1, 1, AM_Feature_Output_Number])

# Reshape to 2D tensor
with tf.name_scope("Concatination_Dimensions"):
    numOutputConcat = filterNumberMFCC
with tf.name_scope("L1_OutputTensor_2D"):
    outputTensor_L1_2D = tf.reshape(AM_Output_4D, [-1, numOutputConcat])

# Dropout percentage
keep_Prob = tf.placeholder(tf.float32)
h_FC_L2_drop = tf.nn.dropout(outputTensor_L1_2D, keep_Prob)

# Output
with tf.name_scope("FCL2_Weights"):
    W_FC_L2 = weightVariable([numOutputConcat, 4])
with tf.name_scope("FCL2_Bias"):
    b_FC_L2 = biasVariable([4])

with tf.name_scope("Final_Linear_Function"):
    y = tf.matmul(h_FC_L2_drop, W_FC_L2) + b_FC_L2

# Softmax Output, loss-function
if (lossFunction == "Cross-Entropy"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                  logits=y))  # (Goldstandard, Output); Cross Entropy; reduce_mean
    # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # (Goldstandard, Output); Cross Entropy; reduce_sum
elif (lossFunction == "Hinge-Loss"):
    loss = tf.losses.hinge_loss(labels=y_, logits=y, weights=1.0)
elif (lossFunction == "Mean-Squared-Error"):
    loss = tf.losses.mean_squared_error(labels=y_, predictions=tf.nn.softmax(y))    #Introduced softmax before y
else:
    exit()

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
logFileTmp += "Number of filters MFCC: " + str(filterNumberMFCC) + "\n"
logFileTmp += "Filter size MFCC: " + str(mfccFilterSize) + "\n"
logFileTmp += "Output size AM FCL: " + str(AM_Feature_Output_Number) + "\n"
logFileTmp += "Batchsize: " + str(batchSize) + "\n"
logFileTmp += "Learning Rate: " + str(learningRate) + "\n"
logFileTmp += "Activation Function AM: " + str(activationFunction_AM) + "\n"
logFileTmp += "Loss Function: " + str(lossFunction) + "\n"
logFileTmp += "Dropout: " + str(dropout) + "\n"
logFileTmp += "Optimizer: " + str(optimizerFunction) + "\n"
logFileTmp += "Seed: " + str(weightSeed) + "\n"
logFileTmp += "##########\n"


print("Starting Training...")
programStartTime = str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")) # stores the time when the computation starts.
logPath = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources/log/"
logPath += programStartTime

# Creates a folder in which to store all logfiles
if not os.path.exists(logPath):
    os.makedirs(logPath)

with tf.Session(config=config) as sess:
# with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
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
                feed_dict={x_mfcc: batch[1], y_: batch[2], keep_Prob: 1.0})

            # Run the optimizer to update weights.
            summary, l, train = sess.run([merged, loss, train_Step], feed_dict={x_mfcc: batch[1], y_: batch[2], keep_Prob: dropout})

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
    devAccuracy = accuracy.eval(feed_dict={x_mfcc: evaluationTriple[1], y_: evaluationTriple[2], keep_Prob: 1.0})

    testTriple = createEvalList(testList)
    testAccuracy = accuracy.eval(feed_dict={x_mfcc: testTriple[1], y_: testTriple[2], keep_Prob: 1.0})

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