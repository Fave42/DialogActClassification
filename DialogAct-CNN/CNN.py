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
import numpy as np
import cPickle as pickle

# Server Paths
pathTraining = "NN_Input_Files/trainData_3-5WordContext_prot2.pickle"
pathEvaluation = "NN_Input_Files/devData_3-5WordContext_prot2.pickle"

# PC Paths
#pathTraining = "D:/NN_Projekt/trainData_3-5WordContext.pickle"
#pathEvaluation = "D:/NN_Projekt/devData_3-5WordContext.pickle"

#trainingList = pickle.load(open(pathTraining, "rb"))
evaluationList = pickle.load(open(pathEvaluation, "rb"))

print len(evaluationList[0][0])
print evaluationList[0][0].shape
print evaluationList[0][1].shape

#trainingArray = np.asarray(pathTraining)
#evaluationArray = np.asarray(evaluationList)

#print evaluationList[0]

# print(evaluationList[0, 0].shape)
# print(evaluationList[0])
#
evalDataTF = np.array(evaluationList[0][0]).reshape(1, 108, 300, 1)
#evalDataTF = tf.convert_to_tensor(evaluationList[0][0], np.float32)
print evaluationList[0][0].shape
print "-----------------------"
print evalDataTF




#
# print(evalDataTF[0, 0].shape)
# print(evalDataTF[0])
#
features = []
labels = []
# for item in evalDataTF
features.append(tf.convert_to_tensor(evaluationList[0][0], np.float32))
labels.append(tf.convert_to_tensor(evaluationList[0][1], np.float32))
#
# features = np.asarray(features)
# labels = np.asarray(labels)
#
# # print(features)
# # print(labels)
#
#assert features.shape[0] == labels.shape[0]
#
#features_placeholder = tf.placeholder(features.dtype, features.shape)
#labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
#
# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# iterator = dataset.make_initializable_iterator()
#
# ### Functions ###
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
#
#
# ### Graph definition ###
#
x = tf.placeholder(tf.float32, shape=[108, 300]) # input vectors
#print(x.shape)
y_ = tf.placeholder(tf.float32, shape=[1, 4]) # gold standard labels; 1hot-vectors

x_ = tf.reshape(x, shape=[1, 108, 300, 1])
#
W_conv1 = weightVariable([2, 108, 1, 32])
b_conv1 = biasVariable([20])
#
tmp = conv2d(x_, W_conv1)
# #todo Use GlobalAveragePooling
#
# ### Session ###
# # numEpoch = 1
# #
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(numEpoch):
#         for i in range(0, len(trainingArray)):
# #todo Add randemization of training examples
#
# Creating the session object
sess = tf.Session()

# Initializing the variables
sess.run(tf.global_variables_initializer())

# Assigning values to placeholders and running the graph
input = features[0]

# we want to get (fetch) the value of a, feeding the input vector for the placeholders
output = sess.run([tmp], {x: evaluationList[0][0]})
#
print(output[0].shape)
print(output)