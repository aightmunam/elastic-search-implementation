import os
import sys
import time
import codecs
import csv
from nltk.stem import PorterStemmer


def stemQuery(queryString):
	stemmer = PorterStemmer()
	return stemmer.stem(queryString)

def getTermID(termName):
	t = termName
	term = ""
	with open("termids.txt", "r",  encoding="utf8") as termIDs:
		for line in termIDs:
			if t in line: 
				datas = line.split()
				if t == datas[1]:
					term = datas[0]

	del termIDs
	if term == "":
		print("Term was not found.\n")
	return term

def getDocID(docTitle):
	d = docTitle
	doc = ""
	with open("docids.txt", "r") as docIDs:
		for line in docIDs:
		    if d in line: 
		    	datas = line.split()
		    	if d == datas[1]:
		    		doc = datas[0]

	del docIDs
	if doc == "":
		print("Document was not found.\n")
	return doc


def getListingsforDoc(docTitle):
	d = docTitle
	doc = getDocID(d)
	print ("Listing for doc:\t" + d)
	distinctTerms = 0
	totalTerms = 0
	with open("doc_index.txt") as docINFO:
		for line in docINFO:
			if doc in line:
				datas = line.split()
				if doc == datas[0]:
					distinctTerms = distinctTerms + 1
					totalTerms = totalTerms + len(datas[2:len(datas)])
	if doc != "":
		print("DOCID:\t"  + doc + "\n")
		print("Distinct Terms:\t" + str(distinctTerms) +"\n")
		print("total Terms:\t" + str(totalTerms) + "\n")

	return


def getListingsforTerm(termName):
	t = stemQuery(termName)
	term = getTermID(t)
	print ("Listing for term:\t" + t)

	inv = ""
	with open("term_info.txt", "r") as termINFO:
		for line in termINFO:
			if term in line:
				datas = line.split()
				if term == datas[0]:
					print("TERMID:\t" + datas[0] + "\n")
					print("Number of documents containing " + t + ":\t" + datas[3] +"\n")
					print("Term frequency in corpus:\t" + datas[2] + "\n")
					print("Inverted List Offset:\t" + datas[1] + "\n")
					inv = datas[1]
					break;

	return


def getDualListings(termName, docTitle):
	t = stemQuery(termName)
	d = docTitle
	print("Inverted list for term:\t" + t)
	print("In document:\t" + d)

	term = getTermID(t)
	doc = getDocID(d)
	print("TERMID:\t" + term)
	print("DOCID:\t" + doc)

	termFreqinDoc = 0
	Positions = 0
	with open("doc_index.txt") as docINFO:
		for line in docINFO:
			if doc in line:
				datas = line.split()
				if doc == datas[0] and term == datas[1]:
					termFreqinDoc = len(datas[2:len(datas)])
					Positions = ", ".join(datas[2:len(datas)]) 

	print("Term frequency in Document:\t" + str(termFreqinDoc))
	print("Positions:\t" + str(Positions))


if len(sys.argv) >= 4:
	if (sys.argv[1] == '--term') and (sys.argv[3] =='--doc'):
		getDualListings(sys.argv[2], sys.argv[4])
	elif (sys.argv[1] == '--doc') and (sys.argv[3] =='--term'):
		getDualListings(sys.argv[4], sys.argv[2])
	else:
		print("ERROR! Please enter the query in correct format: [--doc/--term][Document Title/Term Name]")
		exit()
elif len(sys.argv) >= 2:
    if(sys.argv[1] == '--doc'):
    	getListingsforDoc(sys.argv[2])
    elif(sys.argv[1] == '--term'):
    	getListingsforTerm(sys.argv[2])
    else:
    	print("ERROR! Please enter the query in correct format: [--doc/--term][Document Title/Term Name]")
    	exit()
else:
	print("ERROR! Please enter the query in correct format: [--doc/--term][Document Title/Term Name]")
	exit()


