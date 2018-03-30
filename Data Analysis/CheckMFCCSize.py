import numpy as np

def generateMFCCDict():
    mfccDict = {}  # Should be declared out here, otherwise it won't contain all data dumbass -.-
    with open("F:/DeepLearners/dev_MFCC_features.arff", 'r') as mfccFile:
        for line in mfccFile:
            if ("'" in line):
                lineSplitted = line.split(",")
                fileID = lineSplitted.pop(0)  # remove the ID
                fileID = fileID.replace("'", "")  # Checked, it works
                lineSplitted.pop(-1)  # remove the class

                if (fileID not in mfccDict):
                    mfccDict[fileID] = 1
                elif (fileID in mfccDict):
                    mfccDict[fileID] += 1
    return mfccDict


mfccDict = generateMFCCDict()

sum = 0
min = 10000
max = 0
for key in mfccDict:
    print(key, ":", mfccDict[key])
    sum += mfccDict[key]
    if (min > mfccDict[key]):
        min = mfccDict[key]
    if (max < mfccDict[key]):
        max = mfccDict[key]

print("Durchschnitt:", sum/len(mfccDict))
print("Maximum:", max)
print("Minimum:", min)