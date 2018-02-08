path = "/home/fabian/Dokumente/Projekt_NN/train.txt"
path2 = "/home/fabian/Dokumente/Projekt_NN/dev.txt"
path3 = "/home/fabian/Dokumente/Projekt_NN/test.txt"

print("train.txt")
with open(path, "r") as file:
    channelDict = {}
    countGesamt = 0
    for line in file:
        line = line.split()
        channel = line[1]
        if channel in channelDict:
            channelDict[channel] += 1
        else:
            channelDict[channel] = 1

        countGesamt += 1


    # print(channelDict)
    for key in channelDict:
        print("Key:", key, "Channel:", channelDict[key], "%:", float(channelDict[key])/countGesamt)
print("\ndev.txt")
with open(path2, "r") as file:
    channelDict = {}
    countGesamt = 0
    for line in file:
        line = line.split()
        channel = line[1]
        if channel in channelDict:
            channelDict[channel] += 1
        else:
            channelDict[channel] = 1

        countGesamt += 1

    # print(channelDict)
    for key in channelDict:
        print("Key:", key, "Channel:", channelDict[key], "%:", float(channelDict[key]) / countGesamt)
print("\ntest.txt")
with open(path3, "r") as file:
    channelDict = {}
    countGesamt = 0
    for line in file:
        line = line.split()
        channel = line[1]
        if channel in channelDict:
            channelDict[channel] += 1
        else:
            channelDict[channel] = 1

        countGesamt += 1

    # print(channelDict)
    for key in channelDict:
        print("Key:", key, "Channel:", channelDict[key], "%:", float(channelDict[key]) / countGesamt)