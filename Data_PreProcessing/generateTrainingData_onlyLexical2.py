#!/bin/bash

'''
This script loads a transcription file, extracts the word2vec features for each word of the uttered sentence
and stores those vectors in a 2D matrix. Furthermore a one hot vector is generated for each training examble.
Finally the script stores the training examples together with their one-hot vectors in a pickle file.

Usage: python3 generateTrainingData_onlyLexical.py

Filestructure must be:
/dict/GoogleNews-vectors-negative300.bin
/generateTrainingData_onlyLexical2.py
'''

import gensim
import time  # To time the import and excecution
import random as rand
import pickle
import numpy as np
import datetime
import os
import argparse

def main():
    fileNameList = ["Train/train.txt", "Dev/dev.txt", "Test/test.txt"]
    stopWordSave = ""

    # Store the arguments given at the start of teh script
    paddingDepth, maxSentenceLength, stopWordPath = getArgs()

    #print("A paddingdepth of %d and a maximum sentence length of %d will be used.") % (paddingDepth, maxSentenceLength)
    print("A paddingdepth of", paddingDepth, "and a maximum sentence length of", maxSentenceLength, "will be used.")
    if (stopWordPath != ""):
        print("The stopwordsfile", stopWordPath, "will be used.")
        stopWordSave = "_fsw"
    else:
        print("No stopwordsfile wax specified.")

    # Loads "unknownWordsDict.txt that all every unknown words has unique 300 random float numbers
    # Serverpath
    unknownWordsDict = pickle.load(open("dict/unknownWordsDict.pickle", "rb"))

    # Load the stopword file if it is present
    # stopWordPath = "stopwords.txt"
    stopWordsDict = {}
    if (stopWordPath != ""):
        if os.path.isfile(stopWordPath):
            print("Stopwordsfile found!\nImporting it now...")
            stopWordsDict = buildStopWordDict(stopWordPath)
        else:
            print("Stopwordsfile not found!\nPlease rerun the script!")
            quit()

    # Generate a timestamp
    timeStamp = generateTimesstamp()
    # Start timing the importing of the dictionary
    startTime = time.time()

    # Load Google's pre-trained Word2Vec model.
    # Laptop/PC Verions
    # model = gensim.models.KeyedVectors.load_word2vec_format(
    #     'dict/GoogleNews-vectors-negative300.bin', binary=True)
    # Server Version
    print("Importing word2vec binary now...")
    model = gensim.models.Word2Vec.load_word2vec_format(
        'dict/GoogleNews-vectors-negative300.bin', binary=True)

    endTime = time.time()
    duration = endTime - startTime

    print("--- Dictionary was imported successfully! ---")
    print("Import duration of dictionary was " + str(duration) + " seconds!")

    for type in fileNameList:
        path = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/"
        path += type
        work(paddingDepth, maxSentenceLength, path, type, unknownWordsDict, timeStamp, model, stopWordsDict, stopWordSave)

# This function parses and returns arguments passed in
def getArgs():
    # Add description
    parser = argparse.ArgumentParser(description='This script generates training, development and test data for a CNN.')
    # Add arguments
    parser.add_argument("-p", "--paddingDepth", type=int,
                        help="Specifies the padding depth of the input Matrix. If left empty the standard value 5 will be used.",
                        required=False, default=4)
    parser.add_argument("-m", "--maxSentenceLength", type=int,
                        help="Specifies the maximum sentence length. If left empty the standard value 100 will be used.",
                        required=False, default=100)
    parser.add_argument("-s", "--stopWords", type=str,
                        help="Specifies a stopwordfile. If left empty no stopword filtering will be done.",
                        required=False, default="")
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    pd = args.paddingDepth
    msl = args.maxSentenceLength
    swf = args.stopWords
    # Return all variables
    return pd, msl, swf


# Generates a stopword dictionary based on a given "stopwords.txt" file in the folder
def buildStopWordDict(stopWordPath):
    tmpDict = {}
    with open(stopWordPath, "r") as stopWordsFile:
        for word in stopWordsFile:
            if (word not in tmpDict):
                tmpDict[word] = word
    print("Stopwordsfile imported!")
    return tmpDict


# Generates a timestamp for the current day and time
def generateTimesstamp():
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    timestamp = "%s-%s-%s-%d-%d" % (year, month, day, hour, minute)
    return timestamp


# Generates 300 random numbers between -1 and 1 for unknown words
def getRandomDimensions():
    randomList = []
    for i in range(0, 300):
        randomList.append(rand.uniform(-1.0, 1.0))
    return randomList


def work(paddingDepth, maxSentenceLength, filePath, dataType, unknownWordsDict, timeStamp, model, stopWordsDict, stopWordSave):
    # Cast the strings to int for computation
    paddingDepth = int(paddingDepth)
    maxSentenceLength = int(maxSentenceLength)

    print("Currently working on %s" % (dataType))

    # Stores every sentence to be dumped with pickle
    outputList = []

    # Start timing the creation of the data
    startTime2 = time.time()
    with open(filePath, "r") as dataFile:
        for line in dataFile:

            splittedLineUncleaned = line.split()
            splittedLineCleaned = []
            for item in splittedLineUncleaned:
                if item not in stopWordsDict:
                    splittedLineCleaned.append(item)

            if (len(splittedLineCleaned) == 0):
                print("Line is empty!")

            fileID = splittedLineCleaned[0]
            diagClass = splittedLineCleaned[1]

            # first list stores the actual featureMatrix, second tuple stores the one-hot vector.
            trainingTuple = [np.zeros((maxSentenceLength + 2 * paddingDepth, 300)), None]

            # print(trainingTuple[0])

            for i in range(paddingDepth, len(splittedLineCleaned) - 2 + paddingDepth):  # for filling np.array correctly.

                if i <= maxSentenceLength + paddingDepth:
                    word = splittedLineCleaned[i + 2 - paddingDepth]  # transforms the i to fit to the splittedLine index.
                    # Check if the word is in the model, if it is in stopWordsDict it will not be added
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


    # Saves the the complete list with every sentence and a timestamp
    if ("train" in dataType):
        savePath ='NN_Input_Files/trainData_'+str(paddingDepth)+'_'+str(maxSentenceLength)+''+str(stopWordSave)+'.pickle'
        with open(savePath, 'wb') as handle:
            pickle.dump(outputList, handle, protocol=2)
        print("### Saving training data! ###")
    if ("test" in dataType):
        savePath = 'NN_Input_Files/testData_'+str(paddingDepth)+'_'+str(maxSentenceLength)+''+str(stopWordSave)+'.pickle'
        with open(savePath, 'wb') as handle:
            pickle.dump(outputList, handle, protocol=2)
        print("### Saving test data! ###")
    if ("dev" in dataType):
        savePath = 'NN_Input_Files/devData_'+str(paddingDepth)+'_'+str(maxSentenceLength)+''+str(stopWordSave)+'.pickle'
        with open(savePath, 'wb') as handle:
            pickle.dump(outputList, handle, protocol=2)
        print("### Saving development data! ###")

    endTime2 = time.time()
    duration2 = endTime2 - startTime2

    print("The rest of the program was completed in " + str(duration2) + " seconds!")


### Run ###
main()