path1 = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/Test/test.txt"
path2 = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/Dev/dev.txt"
path3 = "/mount/arbeitsdaten31/studenten1/deeplearning/2017/Deep_Learners/Data/Train/train.txt"

testList = []
with open(path1) as testFile:
    for sentence in testFile:
        sentenceSplit = sentence.split()
        sentenceSplit.pop(0)  # remove .wav filename
        sentenceSplit.pop(0)  # remove sentence class

        for word in sentenceSplit:
            word = word.lower()  # every word to lowercase
            if word not in testList:
                testList.append(word)

devTrainList = []
with open(path2) as testFile:
    for sentence in testFile:
        sentenceSplit = sentence.split()
        sentenceSplit.pop(0)  # remove .wav filename
        sentenceSplit.pop(0)  # remove sentence class

        for word in sentenceSplit:
            word = word.lower()  # every word to lowercase
            if word not in devTrainList:
                devTrainList.append(word)

with open(path3) as testFile:
    for sentence in testFile:
        sentenceSplit = sentence.split()
        sentenceSplit.pop(0)  # remove .wav filename
        sentenceSplit.pop(0)  # remove sentence class

        for word in sentenceSplit:
            word = word.lower()  # every word to lowercase
            if word not in devTrainList:
                devTrainList.append(word)
countN = 0
count = 0
for word in testList:
    count+=1
    if word not in devTrainList:
        print(word)
        countN+=1
print(countN)
print(count)