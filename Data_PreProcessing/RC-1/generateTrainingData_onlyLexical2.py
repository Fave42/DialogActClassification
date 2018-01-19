#!/bin/bash

'''
This script loads a transcription file, extracts the word2vec features for each word of the uttered sentence
and stores those vectors in a 2D matrix. Furthermore a one hot vector is generated for each training examble.
Finally the script stores the training examples together with their one-hot vectors in a pickle file.

Usage: python3 generateTrainingData_onlyLexical.py <PathToTranscriptionFile>

Filestructure must be:
/dict/GoogleNews-vectors-negative300.bin
/main.py
/generateTrainingData_onlyLexical2.py
'''

import gensim
import time  # To time the import and excecution
import random as rand
import pickle
import numpy as np

# Generates 300 random numbers between -1 and 1 for unknown words
def getRandomDimensions():
    randomList = []
    for i in range(0, 300):
        randomList.append(rand.uniform(-1.0, 1.0))
    return randomList

def work(paddingDepth, maxSentenceLength, filePath, dataType):
    # Loads "unknownWordsDict.txt that all every unknown words has unique 300 random float numbers
    unknownWordsDict = {}
    # Serverpath
    unknownWordsDict = pickle.load(open("dict/unknownWordsDict.pickle", "rb"))

    # Stores every sentence to be dumped with pickle
    outputList = []

    # Start timing the importing of the dictionary
    startTime = time.time()

    # Load Google's pre-trained Word2Vec model.
    # Laptop/PC Verions
    # model = gensim.models.KeyedVectors.load_word2vec_format(
    #     'dict/GoogleNews-vectors-negative300.bin', binary=True)
    # Server Version
    model = gensim.models.Word2Vec.load_word2vec_format(
        'dict/GoogleNews-vectors-negative300.bin', binary=True)

    endTime = time.time()

    print("--- Dictionary was imported successfully! ---")

    # Start timing the creation of the training set
    startTime2 = time.time()

    with open(filePath, "r") as dataFile:
        for line in dataFile:

            splittedLine = line.split()
            fileID = splittedLine[0]
            diagClass = splittedLine[1]

            # first list stores the actual featureMatrix, second tuple stores the one-hot vector.
            trainingTuple = [np.zeros((maxSentenceLength + 2 * paddingDepth, 300)), None]

            # print(trainingTuple[0])

            for i in range(paddingDepth, len(splittedLine) - 2 + paddingDepth):  # for filling np.array correctly.

                if i <= maxSentenceLength + paddingDepth:
                    word = splittedLine[i + 2 - paddingDepth]  # transforms the i to fit to the splittedLine index.
                    # Check if the word is in the model
                    # If not fill the list with 300 random numbers between -1 and 1
                    if (word in model.vocab):
                        word300 = model[word]
                        # word300 = word300.tolist()
                    elif (word in unknownWordsDict):
                        word300 = unknownWordsDict[word]
                    else:
                        word300 = getRandomDimensions()
                        unknownWordsDict[word] = word300

                    # fills the corresponding row and elements of the trainingMatrix with the values of word300.
                    for j in range(0, 300):
                        trainingTuple[0][i][j] = word300[j]
                else:
                    print("Sentence is longer than " + str(maxSentenceLength) + "words!")
                    print(line)
                    break

            if diagClass == "backchannel":
                trainingTuple[1] = np.array([1, 0, 0, 0])
            elif diagClass == "statement":
                trainingTuple[1] = np.array([0, 1, 0, 0])
            elif diagClass == "question":
                trainingTuple[1] = np.array([0, 0, 1, 0])
            elif diagClass == "opinion":
                trainingTuple[1] = np.array([0, 0, 0, 1])

            trainingTuple = np.asarray(trainingTuple)
            outputList.append(trainingTuple)

        outputList = np.asarray(outputList)

    # Saves the unknown words with their respective 300 random numbers
    with open('dict/unknownWordsDict.pickle', 'wb') as handle:
        pickle.dump(unknownWordsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Saves the the complete list with every sentence
    if ("train" in dataType):
        with open('NN_Input_Files/trainOutput_3-5WordContext.pickle', 'wb') as handle:
            pickle.dump(outputList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if ("test" in dataType):
        with open('NN_Input_Files/testOutput_3-5WordContext.pickle', 'wb') as handle:
            pickle.dump(outputList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if ("dev" in dataType):
        with open('NN_Input_Files/devOutput_3-5WordContext.pickle', 'wb') as handle:
            pickle.dump(outputList, handle, protocol=pickle.HIGHEST_PROTOCOL)

    endTime2 = time.time()
    duration = endTime - startTime
    duration2 = endTime2 - startTime2

    print("Import duration of dictionary was " + str(duration) + " seconds!")
    print("The rest of the program was completed in " + str(duration2) + " seconds!")

