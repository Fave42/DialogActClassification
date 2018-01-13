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

#fileName = sys.argv[1]
# For testing provide the file "TestDict.txt" wich is provided in the parent folder of this script
fileName = "F:/Mega/Uni-Master/Deep Learning/Projekt/Data/Train/train.txt"

# Start timing the importing of the dictionary
startTime = time.time()

# Load Google's pre-trained Word2Vec model.
# Richie Laptop
#model = gensim.models.KeyedVectors.load_word2vec_format(
#    '/home/richard/Projekte/Project_Deep_Learners/Processing_Resources/dict/GoogleNews-vectors-negative300.bin',
#    binary=True)

# Fabian Laptop
model = gensim.models.KeyedVectors.load_word2vec_format(
    'D:/word2vec_Test/dict/GoogleNews-vectors-negative300.bin', binary=True)
endTime = time.time()

print("--- Dictionary was imported successfully! ---")

# Generate the outputfilfe wich is stored in the root folder
outputFile = open("testOutput.txt", "w")

# Count and keep the unknown words for later review
countKnown = 0
countUnknown = 0
unknownWordsDict = {}


# Start timing the creation of the training set
startTime2 = time.time()

with open(fileName, "r") as dictionary:
    for line in dictionary:

        splittedLine = line.split()
        fileID = splittedLine[0]
        diagClass = splittedLine[1]

        for i in range(2, len(splittedLine)):
            # first list stores the actual featureMatrix, second tuple stores the one-hot vector.
            trainingTuple = [[], []]
            word = splittedLine[i]
            # Check if the word is in the model
            # If not fill the list with 300 0's
            if (splittedLine[i] in model.vocab):
                word300 = model[word]
                countKnown += 1
            else:
                word300 = [0] * 300
                countUnknown += 1
                if (word not in unknownWordsDict):
                    unknownWordsDict[word] = 1
                else:
                    unknownWordsDict[word] += 1
            trainingTuple[0].append(word300)

            if diagClass == "backchannel":
                trainingTuple[1] = [1, 0, 0, 0]
            elif diagClass == "statement":
                trainingTuple[1] = [0, 1, 0, 0]
            elif diagClass == "question":
                trainingTuple[1] = [0, 0, 1, 0]
            elif diagClass == "opinion":
                trainingTuple[1] = [0, 0, 1, 0]

            #print('OK')
            # Store the tuple in an output file
            # Each tuple/word is currently in one line
            outputFile.write((word) + " ")
            for item in trainingTuple[0]:
                for subitem in item:
                    outputFile.write(str(subitem) + " ")
            for item in trainingTuple[1]:
                outputFile.write(str(item) + " ")
            outputFile.write("\n")
outputFile.close()

endTime2 = time.time()
duration = endTime - startTime
duration2 = endTime2 - startTime2

print("Import duration of dictionary was " + str(duration) + " seconds!")
print("The rest of the program was completed in " + str(duration2) + " seconds!")
print(str(countKnown) + " words where found in the dictionary, " + str(countUnknown) + " weren't.")
print("The unknown words are stored in the file Unknown_Words.txt.")

# Create a sorted file with all unknown words their frequency
sortedUnknownWordsList = sorted(unknownWordsDict.items(), key=operator.itemgetter(1)) # Is saved as a list of Tuples
with open("Unknown_Words.txt", "w") as unknownWordsFile:
    for item in reversed(sortedUnknownWordsList):
        unknownWordsFile.write(str(item[0]) + " " + str(item[1]) + "\n")