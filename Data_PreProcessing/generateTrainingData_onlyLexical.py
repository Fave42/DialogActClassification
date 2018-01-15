#!/bin/bash

'''
This script loads a transcription file, extracts the word2vec features for each word of the uttered sentence
and stores those vectors in a 2D matrix. Furthermore a one hot vector is generated for each training examble.
Finally the script stores the training examples together with their one-hot vectors in a pickle file.

Usage: python3 generateTrainingData_onlyLexical.py <PathToTranscriptionFile>
'''

import sys
import gensim
import time # To time the import and excecution
import operator # For sorting the dictionary that is storing the unknown words
import random as rand
import pickle

#fileName = sys.argv[1]
# Fabian-Laptop Path
#fileName = "D:/Mega/Uni-Master/Deep Learning/Projekt/Data/Train/train.txt"
# Serverpath
#fileName = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/Train/train.txt"
#fileName = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/Dev/dev.txt"
fileName = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/Test/test.txt"

# Select the padding depth of the matrix (1=one line with 0's, 2=two lines with 0's
paddingDepth = 1

# Loads "unknownWordsDict.txt that all every unknown words has unique 300 random float numbers
unknownWordsDict = {}
# Fabian-Laptop Path
#unknownWordsDict = pickle.load(open("unknownWordsDict.pickle", "rb"))
# Serverpath
unknownWordsDict = pickle.load(open("dict/unknownWordsDict.pickle", "rb"))

# Stores every sentence to be dumped with pickle
outputList = []

# Generates 300 random numbers between -1 and 1 for unknown words
def getRandomDimensions():
    randomList = []
    for i in range(0,300):
        randomList.append(rand.uniform(-1.0, 1.0))
    return randomList

# Start timing the importing of the dictionary
startTime = time.time()

# Load Google's pre-trained Word2Vec model.
# Laptop/PC Verions
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     'dict/GoogleNews-vectors-negative300.bin', binary=True)
# Server Version
model = gensim.models.Word2Vec.load_word2vec_format(
    'dict/GoogleNews-vectors-negative300.bin', binary=True)
# Path Richie: '/home/richard/Projekte/Project_Deep_Learners/Processing_Resources/dict/GoogleNews-vectors-negative300.bin'

endTime = time.time()

print("--- Dictionary was imported successfully! ---")

# Start timing the creation of the training set
startTime2 = time.time()

with open(fileName, "r") as dataFile:
    for line in dataFile:

        splittedLine = line.split()
        fileID = splittedLine[0]
        diagClass = splittedLine[1]

        # first list stores the actual featureMatrix, second tuple stores the one-hot vector.
        trainingTuple = [[], []]
        for i in range(0, paddingDepth):
            zeroPadding = [0] * 300
            trainingTuple[0].append(zeroPadding)

        #print(trainingTuple[0])

        for i in range(2, len(splittedLine)):
            word = splittedLine[i]
            # Check if the word is in the model
            # If not fill the list with 300 0's
            if (splittedLine[i] in model.vocab):
                word300 = model[word]
                word300 = word300.tolist()
            elif (splittedLine[i] in unknownWordsDict):
                word300 = unknownWordsDict[word]
            else:
                word300 = getRandomDimensions()
                unknownWordsDict[word] = word300

            trainingTuple[0].append(word300)

        for i in range(0, paddingDepth):
            zeroPadding = [0] * 300
            trainingTuple[0].append(zeroPadding)

        if diagClass == "backchannel":
            trainingTuple[1] = [1, 0, 0, 0]
        elif diagClass == "statement":
            trainingTuple[1] = [0, 1, 0, 0]
        elif diagClass == "question":
            trainingTuple[1] = [0, 0, 1, 0]
        elif diagClass == "opinion":
            trainingTuple[1] = [0, 0, 0, 1]

        #print("################")
        #print(trainingTuple)

        outputList.append(trainingTuple)

# Saves the unknown words with their respective 300 random numbers
with open('dict/unknownWordsDict.pickle', 'wb') as handle:
    pickle.dump(unknownWordsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Saves the the complete list with every sentence
with open('Output_Files/testExamples_2WordContext.pickle', 'wb') as handle:
    pickle.dump(outputList, handle, protocol=pickle.HIGHEST_PROTOCOL)

endTime2 = time.time()
duration = endTime - startTime
duration2 = endTime2 - startTime2

print("Import duration of dictionary was " + str(duration) + " seconds!")
print("The rest of the program was completed in " + str(duration2) + " seconds!")