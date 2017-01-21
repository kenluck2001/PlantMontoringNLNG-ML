import os
if __name__ == "__main__":

    f = open('result/data.txt', 'r')
    lineList = [n for n in f.readlines() if not n.startswith('current error:')]


    with open('result/output.txt', 'a') as fileObj:
        for line in lineList:
            fileObj.write(line + '\n')
