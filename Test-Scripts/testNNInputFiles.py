#!/bin/bash

'''
This script is used to inspect and check the data generated  by generateTrainingData_Embeddings_acolex.py

Usage: python3 testNNInputFiles.py

Filestructure must be:
/NN_Input_Files/devData_acolex_Embeddings.pickle
/NN_Input_Files/testData_acolex_Embeddings.pickle
/NN_Input_Files/trainData_acolex_Embeddings.pickle
'''

import pickle
import numpy

# Dev
devData = pickle.load(open('/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources/NN_Input_Files/devData_acolex_Embeddings.pickle', "rb"))
# for item in devData:
#     print(item)

#print(devData[0])
devData = numpy.asarray(devData)
print(devData[0][1].shape)
print(devData[0][1][0].shape)
# # Test
# testData = pickle.load(open('/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources/NN_Input_Files/devData_acolex_Embeddings.pickle', "rb"))
#
# # Training
# trainData = pickle.load(open('/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Processing_Resources/NN_Input_Files/devData_acolex_Embeddings.pickle', "rb"))