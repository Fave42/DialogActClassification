#!/bin/bash

'''
This script loads a transcription file, extracts the word2vec features for each word of the uttered sentence
and stores those vectors in a 2D matrix. Furthermore a one hot vector is generated for each training examble.
Finally the script stores the training examples together with their one-hot vectors in a pickle file.

Usage: python3 generateTrainingData_onlyLexical.py <PathToTranscriptionFile>
'''

import sys
import gensim

fileName = sys.argv[1]

# Load Google's pre-trained Word2Vec model.
# Richie Laptop
model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/richard/Projekte/Project_Deep_Learners/Processing_Resources/dict/GoogleNews-vectors-negative300.bin',
    binary=True)
# Fabian Laptop
model = gensim.models.KeyedVectors.load_word2vec_format(
    'D:/word2vec_Test/dict/GoogleNews-vectors-negative300.bin',
    binary=True)

with open(fileName, "r") as inputFile:
    for line in inputFile:

        splittedLine = line.split()
        fileID = splittedLine[0]
        diagClass = splittedLine[1]

        trainingTuple = [[], []]  # first list stores the actual featureMatrix, second tuple stores the one-hot vector.
        for i in range(2, len(splittedLine)):
            trainingTuple[0].append(model[splittedLine[i])

            if diagClass == "backchannel":
                trainingTuple[1] = [1, 0, 0, 0]
            elif diagClass == "statement":
                trainingTuple[1] = [0, 1, 0, 0]
            elif diagClass == "question":
                trainingTuple[1] = [0, 0, 1, 0]
            elif diagClass == "opinion":
                trainingTuple[1] = [0, 0, 1, 0]

            print('OK')
            # store the tuple in file.
