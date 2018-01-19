#!/bin/bash

'''
This script loads a transcription file, extracts the word2vec features for each word of the uttered sentence
and stores those vectors in a 2D matrix. Furthermore a one hot vector is generated for each training examble.
Finally the script stores the training examples together with their one-hot vectors in a pickle file.

Usage: python3 generateTrainingData_onlyLexical.py <PathToTranscriptionFile>
'''

import generateTrainingData_onlyLexical2 as generateScript

def main():

    fileNameList = ["Train/train.txt", "Dev/dev.txt", "Test/test.txt"]

    # Select the padding depth of the matrix (1=one line with 0's, 2=two lines with 0's
    paddingDepth = input("Which vertical padding depth is used?\n")
    maxSentenceLength = input("What is the maximum sentence length?\n")

    for type in fileNameList:
        path = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/"
        path += type
        generateScript.work(paddingDepth, maxSentenceLength, path, type)

### Run ###
main()