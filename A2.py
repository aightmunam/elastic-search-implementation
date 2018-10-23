import os
import sys
import nltk
import time
import math
import codecs
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup, Comment
import csv
import itertools


TOTALDOCS = 3496
# invertedIndex = {}
# docLengths = {}


def loadLengths():
	lengthdict = dict()
	with open("doc_lengths.txt") as file:
		while(True):
			l = file.readline()
			if l is "":
				break
			else:
				p = l.split("\t")
				p[len(p) - 1] = p[len(p) - 1].rstrip()
				lengthdict[p[0]] = p[1]

	return lengthdict


def loadInvertedIndex():
	invertedIndex = dict()
	with open("term_index.txt") as ii:
		while (True):
			l = ii.readline()
			if l is "":
				break
			else:
				p = l.split("\t")
				p[len(p)-1] = p[len(p)-1].rstrip()

				invertedIndex[p[0]] = p[1:]
	return invertedIndex


def textNormalize(text):
	token = tokenizer.tokenize(text)
	token = [t.lower() for t in token]
	token = [stemmer.stem(i) for i in token if i not in stopWords]
	return token


def parseThisShit(html):
	soup = BeautifulSoup(html, 'html.parser')
	soup = soup.find_all('query')
	t1 = []
	for i in range(len(soup)):
		text = soup[i].get_text()
		lines = (line.strip() for line in text.splitlines())
		chunks = (phrase.strip()
		for line in lines for phrase in line.split(" "))
		text = ' '.join(chunk for chunk in chunks if chunk)
		t1.append(text)
	return t1


def getTermID(term):
	t = term
	term = ""
	with open("termids.txt", "r",  encoding="utf8") as termIDs:
		for line in termIDs:
			if t in line:
				data = line.split()
				if t == data[1]:
					term = data[0]

	del termIDs
	return term

def getDocID(doc):
	d = doc
	doc = ""
	with open("docids.txt", "r") as docIDs:
		for line in docIDs:
			if d in line:
				data = line.split()
				if d == data[1]:
					doc = data[0]

	del docIDs
	return doc



def deltaDecodeDocs(postings):
	docs = list()
	positions = dict()
	newD = 0
	newT = 0
	term = []
	for _ in range(0,len(postings)):
		tmp = postings[_]
		doc, n, position = tmp.partition(":")
		doc = int(doc)
		position = int(position)
		if doc is 0:
			newT = position + newT
			term.append(newT)
		elif doc is not 0:
			if term:
				positions[str(newD)] = term
			term = []
			newD = doc + newD
			docs.append(newD)
			newT = newT + position
			term.append(newT)
			p
	return docs, positions


def tf(term, document):
	freq = 0
	doc = str(document)
	if doc is "":
		return freq
	with open("doc_index.txt") as docINFO:
		for line in docINFO:
			if doc in line:
				data = line.split()
				if doc == data[0] and term == data[1]:
					freq = len(data[2:len(data)])
					return freq

	return freq


def tf_mem(term, document):
	freq = 0
	doc = str(document)
	if doc not in invertedIndex.keys():
		for i in range(1,len(document)):
			if(term in document[i]):
				freq = freq + 1
		return freq

	_, positions = deltaDecodeDocs(invertedIndex[term])
	print(term)
	return len(positions[doc])


def length(document):
	if str(document) in docLengths:
		return docLengths[str(document)]
	return len(document)


def getDocTerms(doc):
	# DocDict = {}
	DocList = []
	totalTerms = 0
	doc = str(doc)
	with open("doc_index.txt") as docINFO:
		for line in docINFO:
			if doc in line:
				data = line.split()
				if doc == data[0]:
					# DocDict[str(data[1])] = data[2:len(data)]
					DocList.append(data[1])
					totalTerms = totalTerms + len(data[2:len(data)])

	return DocList, totalTerms



def df(id):
	newD = 0
	docs = []
	postings = invertedIndex[id]
	docs, _ = deltaDecodeDocs(postings)
	return len(docs)


def avglength(totalDocs):
	sum = 0
	for i in range(1, totalDocs):
		sum = sum + int(length(i))
	return sum/totalDocs


def oktf(term, document):
	avglen = avglength(TOTALDOCS)
	return float(tf_mem(term, document) / (tf_mem(term, document) + 0.5 + 1.5 * (int(length(document)) / avglen)))


def otaki_tf(document, query):
	Dterms, _ = getDocTerms(document)
	tfD = dict()
	tfQ = dict()
	Qterms = query
	for i in range(len(Dterms)):
		tfD[Dterms[i]] = oktf(Dterms[i], document)
	for j in range(len(Qterms)):
		tfQ[Qterms[j]] = oktf(Qterms[j], query)

	dlen = vectorlength(list(tfD.values()))
	qlen = vectorlength(list(tfQ.values()))
	print(dlen)
	print(qlen)

	dxq = 0
	keylist = list(tfQ.keys())
	k2 = list(tfD.keys())
	print(set(keylist).intersection(k2))

	for key in tfQ.keys():
		if key in tfD.keys():
			print(key)
			dxq = dxq + tfQ[key] * tfD[key]

	return dxq / (dlen * qlen)


def td_idf(term, document):
	D = TOTALDOCS
	return int(oktf(term, document) + math.log(D/df(term)))


def K(document, k1, b):
	return int(k1 * ((1 - b) + b * (length(document) / avglen)))


def okapi_BM25(term, document, query):
	D = TOTALDOCS
	k1 = 1.2
	k2 = 500
	b = 0.75
	return int(math.log((D+0.5)/(df(term) + 0.5)))\
		* (((1+k1)*tf(term, document))/(K(document, k1, b) + tf(term, document))) \
		* (((1+k2)*tf(term, query))/(k2 + tf(term, query)))


def vectorlength(tfs):
	s = 0
	for i in range(1,len(tfs)):
		s = s + (tfs[i] * tfs[i])

	return math.sqrt(s)

def getAllDocs(term):
	docs, _ = deltaDecodeDocs(invertedIndex[term])
	return docs

def getAllDocsOfaQuery(query):
	d = []
	for i in range(len(query)):
		d.append(getAllDocs(getTermID(query[i])))
	d = list(itertools.chain.from_iterable(d))
	return d





tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*')
stemmer = PorterStemmer()
stopWords = open(os.getcwd()+r"\\stoplist.txt").read()

readfile = open(os.getcwd() + "\\topics.xml", encoding='utf-8', errors = 'ignore').read()

parsed = parseThisShit(readfile)
tokens = textNormalize(parsed[0])

invertedIndex = loadInvertedIndex()
docLengths = loadLengths()


# documents = getAllDocsOfaQuery(tokens)
#
# query = [getTermID(tokens[i]) for i in range(1,len(tokens))]
# query.append('153')
# s = otaki_tf(documents[3], query)

a,b = deltaDecodeDocs(invertedIndex['11809'])
print(a)
print(b)


print(s)
