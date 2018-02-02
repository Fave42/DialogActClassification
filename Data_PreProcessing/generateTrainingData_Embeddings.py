#!/bin/bash

'''
This script generates an input for a CNN compatible with embedding.

Usage: python3 generateTrainingData_Embedding.py

Filestructure must be:
/dict/GoogleNews-vectors-negative300.bin
/generateTrainingData_onlyLexical2.py
'''

import gensim
import pickle
import time
import numpy as np
import random

path = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/"

def getTypes(pathList):
    typeList = []
    # iterate over every given file
    for fileName in pathList:
        tempPath = path
        tempPath += fileName

        with open(tempPath, 'r') as file:
            for sentence in file:
                sentenceSplit = sentence.split()
                sentenceSplit.pop(0)    # remove .wav filename
                sentenceSplit.pop(0)    # remove sentence class

                for word in sentenceSplit:
                    word = word.lower()     # every word to lowercase
                    if word not in typeList:
                        typeList.append(word)

    return typeList


def generateEmbeddingMatrix(model, typeList, unknownWordsDict):
    embeddingMatrix_np = np.zeros((len(typeList)+2, 300))

    for i, type in enumerate(typeList):
        if (type in model.vocab):
            typeVector = model[type]
        elif (type in unknownWordsDict):
            typeVector = unknownWordsDict[type]
        else:
            typeVector = getRandomDimensions()
            unknownWordsDict[type] = typeVector

        for j in range(0, 300):
            embeddingMatrix_np[i][j] = typeVector[j]

    randomFilling = getRandomDimensions()
    for j in range(0, 300):
        embeddingMatrix_np[len(typeList)][j] = randomFilling[j]

    return unknownWordsDict, embeddingMatrix_np


def generateFeatureVector(typeList, fileNameList):
    for fileName in fileNameList:
        tempPath = path
        tempPath += fileName
        outputList = []

        with open(tempPath, 'r') as file:
            for sentence in file:
                sentenceSplit = sentence.split()
                fileID = sentenceSplit.pop(0)           # pop .wav filename
                dialogClass = sentenceSplit.pop(0)      # pop sentence class
                tmpFeatureVec = np.full([106], len(typeList)+1)  # [sentence] + [unknown word Vector] + [padding vector]                tmpTrainingTuple = ()

                for i, word in enumerate(sentenceSplit):
                    word = word.lower()     # every word to lowercase
                    if (word in typeList) and (i <= 100):   # adds every word up to a sentence length of 100
                        tmpFeatureVec[i + 3] = typeList.index(word)
                    elif (word not in typeList) and (i <= 100):
                        tmpFeatureVec[i + 3] = len(typeList)  # every unknown word gets the index of the random vector

                if dialogClass == "backchannel":
                    tmpClassVec = np.array([1, 0, 0, 0])
                elif dialogClass == "statement":
                    tmpClassVec = np.array([0, 1, 0, 0])
                elif dialogClass == "question":
                    tmpClassVec = np.array([0, 0, 1, 0])
                elif dialogClass == "opinion":
                    tmpClassVec = np.array([0, 0, 0, 1])

                tmpTrainingTuple = (tmpFeatureVec, tmpClassVec)
                tmpTrainingTuple = np.asarray(tmpTrainingTuple)

                outputList.append(tmpTrainingTuple)

        outputList = np.asarray(outputList)

        # saves the outputList for every file
        if ("train" in fileName):
            print("Dumping the training ouputList as pickle file...")
            savePath = 'NN_Input_Files/trainData_Embeddings.pickle'
        if ("test" in fileName):
            print("Dumping the test ouputList as pickle file...")
            savePath = 'NN_Input_Files/testData_Embeddings.pickle'
        if ("dev" in fileName):
            print("Dumping the development ouputList as pickle file...")
            savePath = 'NN_Input_Files/devData_Embeddings.pickle'

        with open(savePath, 'wb') as handle:
            pickle.dump(outputList, handle, protocol=2)


# Generates 300 random numbers between -1 and 1 for unknown words
def getRandomDimensions():
    randomList = []
    for i in range(0, 300):
        randomList.append(random.uniform(-1.0, 1.0))
    return randomList


### Run ###
if __name__ == "__main__":
    fileNameList = ["Train/train.txt", "Dev/dev.txt", "Test/test.txt"]
    knownWordFiles = ["Train/train.txt"]

    # Load Google's pre-trained Word2Vec model and time the import
    startTime = time.time()
    print("Importing word2vec binary file...")
    model = gensim.models.Word2Vec.load_word2vec_format(
        'dict/GoogleNews-vectors-negative300.bin', binary=True)
    durationDictImport = time.time() - startTime

    print("--- Dictionary was imported successfully! ---")
    print("Import duration of dictionary was", durationDictImport, "seconds!")

    # Loads "unknownWordsDict.txt that all every unknown words has unique 300 random float numbers
    print("Importing unknownWordsDict file...")
    unknownWordsDict = {}
    with open('dict/unknownWordsDict.pickle', 'wb') as handle:
        pickle.dump(unknownWordsDict, handle, protocol=2)

    unknownWordsDict = pickle.load(open("dict/unknownWordsDict.pickle", "rb"))

    # generate the typelist
    print("Generating typeList...")
    typeList = getTypes(knownWordFiles)

    # generate the embedding matrix
    print("Generating embedding matrix...")
    unknownWordsDict, embeddingMatrix_np = generateEmbeddingMatrix(model, typeList, unknownWordsDict)

    # Saves the unknown words with their respective 300 random numbers
    print("Dumping the unknownWordsDict as pickle file...")
    with open('dict/unknownWordsDict.pickle', 'wb') as handle:
        pickle.dump(unknownWordsDict, handle, protocol=2)

    # Saves the embedding matrix
    print("Dumping the embeddingMatrix_np as pickle file...")
    with open('dict/embeddingMatrix_np.pickle', 'wb') as handle:
        pickle.dump(embeddingMatrix_np, handle, protocol=2)

    # generate the feature vector
    generateFeatureVector(typeList, fileNameList)

    print("Done!!!!")
