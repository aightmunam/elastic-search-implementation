import os
import sys
import time
import codecs
import csv


def index_exists(ls, i):
    return (0 <= i < len(ls)) or (-len(ls) <= i < 0)



def partition(alist):
    # return [alist[i] for i in range(1, len(alist))]
    alist[0], alist[1] = alist[1], alist[0]
    list = alist
    return list

def deltaEncode(list):
	for i in reversed(range(3, len(list))):
		if index_exists(list, i):
			list[i] = str(int(list[i]) - int(list[i-1]))
	return list


invertedList = []


with open('doc_index.txt') as inf:
    reader = csv.reader(inf, delimiter="\t")
    for line in reader:
    	encodedLine = deltaEncode(line)
    	invertedList.append(partition(encodedLine))

# print(invertedList[15])
checkTerms = {}
keeptabsonDocs = {}
termFrequency = {}
docFrequency = {}
byteOffsets = {}



for j in range(0, len(invertedList)):
	if(invertedList[j][0] not in checkTerms):
		
		t = invertedList[j][1]
		checkDocs =  "\t" + t + ":" + ("\t"+ "0" +":").join(invertedList[j][2:len(invertedList[j])])
		checkTerms[invertedList[j][0]] = checkDocs
		keeptabsonDocs[invertedList[j][0]] = invertedList[j][1]
		termFrequency[invertedList[j][0]] = len(invertedList[j][2:len(invertedList[j])])
		docFrequency[invertedList[j][0]] = 1

	else:
		tj = str(int(invertedList[j][1]) - int(keeptabsonDocs[invertedList[j][0]]))
		concatTemp = checkTerms[invertedList[j][0]]
		checkTerms[invertedList[j][0]] = concatTemp + ("\t" + tj + ":" + ("\t" + "0" + ":").join(invertedList[j][2:len(invertedList[j])]))

		incrementCount = termFrequency[invertedList[j][0]]
		docCount = docFrequency[invertedList[j][0]]
		docCount = docCount+1
		termFrequency[invertedList[j][0]] = incrementCount + len(invertedList[j][2:len(invertedList[j])])
		docFrequency[invertedList[j][0]] = docCount
		
del invertedList
del keeptabsonDocs
del reader

termIndex = codecs.open('term_index.txt', 'w', encoding= 'utf8')
for key, value in checkTerms.items():
	termIndex.write(key + value + "\r\n")

del checkTerms
del termIndex


termNum = 1

with open("term_index.txt") as f:
	byteOffsets[str(termNum)] = f.tell()
	termNum = termNum + 1
	line = f.readline()
	while line:
		byteOffsets[str(termNum)] = f.tell()
		termNum = termNum + 1
		line = f.readline()

open("term_info.txt", 'w').close()
with codecs.open("term_info.txt", 'w', encoding='utf8') as termInfo:
	for key, value in termFrequency.items():
		termInfo.write(key + "\t" + str(byteOffsets[key]) + "\t" + str(value) + "\t" + str(docFrequency[key]) + "\r\n")

