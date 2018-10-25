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

# loads the lengths of the documents
def loadLengths():
	lengthdict = dict()
	with open("doc_lengths.txt") as file:
		while (True):
			l = file.readline()
			if l is "":
				break
			else:
				p = l.split("\t")
				p[len(p) - 1] = p[len(p) - 1].rstrip()
				lengthdict[p[0]] = p[1]

	return lengthdict

# loads our inverted index
def loadInvertedIndex():
	invertedIndex = dict()
	with open("term_index.txt") as ii:
		while (True):
			l = ii.readline()
			if l is "":
				break
			else:
				p = l.split("\t")
				p[len(p) - 1] = p[len(p) - 1].rstrip()
				invertedIndex[p[0]] = p[1:]
	return invertedIndex


# normalizes the queries in the same way our inverted index is
def textNormalize(text):
	token = tokenizer.tokenize(text)
	token = [t.lower() for t in token]
	token = [stemmer.stem(i) for i in token if i not in stopWords]
	return token


# parses our queries
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


# returns the term id
def getTermID(term):
	t = term
	term = ""
	with open("termids.txt", "r", encoding="utf8") as termIDs:
		for line in termIDs:
			if t in line:
				data = line.split()
				if t == data[1]:
					term = data[0]

	del termIDs
	return term


# returns the document id of the document
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


# Returns the docs in the postings list of given term
# It also returns a dictionary which stores the list of positions of the given term in every document it appears in
# dict[document in which the term appears] = [list of all the positions it appears in the document]

def deltaDecodeDocs(postings):
	docs = list()
	positions = dict()
	newD = 0
	newT = 0
	term = []
	for _ in range(0, len(postings)):
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
				newT = 0
			term = []
			newD = doc + newD
			docs.append(newD)
			newT = newT + position
			term.append(newT)
			positions[str(newD)] = term
	return docs, positions


# Uses doc_index.txt for the same functionality of tf_mem
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


# returns the number of times a term appears in a document (TF)
# if it is a new document (i.e query) the tf is calculated using the query document
# if a corpus document is found, it just returns the number of times that term appears in it
def tf_mem(term, document):
	freq = 0
	doc = str(document)
	if term in invertedIndex.keys():
		_, positions = deltaDecodeDocs(invertedIndex[term])
		if type(document) is list:
			for i in range(len(document)):
				if (str(term) in document[i]):
					freq = freq + 1
			return freq
		if doc not in positions:
			return 0
		return len(positions[doc])


# returns the length of a document. (This includes the count of all the terms' positions in the document)
def length(document):
	if str(document) in docLengths:
		return int(docLengths[str(document)])
	return len(document)


# Returns all the terms of a document
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


# returns all the documents, a term appears in
def df(term):
	if str(term) not in invertedIndex.keys():
		return len(set(term))
	postings = invertedIndex[str(term)]
	docs, _ = deltaDecodeDocs(postings)
	return len(docs)


# returns the average document length in the corpus
def getAvgFieldLength():
	sum = 0
	for i in range(1, TOTALDOCS):
		sum = sum + int(length(i))
	return sum / TOTALDOCS


# determines the cosine similarity between the query and doc
# both the queryTerms and docTerms are dictionary having query/doc as keys and their tf/oktf/tf-idf as the value
# queryTerms [query term number 1] = oktf/tf/tf-idf of the query term number 1
def sim(queryTerms, docTerms):
	dlen = vectorlength(list(docTerms.values()))
	qlen = vectorlength(list(queryTerms.values()))
	dxq = 0
	for key in queryTerms.keys():
		if key in docTerms.keys():
			dxq = dxq + queryTerms[key] * docTerms[key]

	return dxq / (dlen * qlen)


# this returns the norm or the vector length. i.e By pathagoras theorem, square root of squared sum of all components
def vectorlength(tfs):
	s = 0
	for i in range(len(tfs)):
		s = s + (tfs[i] * tfs[i])
	return math.sqrt(s)


# IMPLEMENTATION FUNCTIONS OF OKAPI-TF FROM HERE ONWARDS


# returns the oktf score given a document and the term
def oktf(term, document):
	avglen = getAvgFieldLength()
	return float(tf_mem(term, document) / (tf_mem(term, document) + 0.5 + 1.5 * (int(length(document)) / avglen)))


# gets the document and query and computes their similarity (scores them) based on the okapi_tf method
def okapi_tf(document, query):
	Dterms, _ = getDocTerms(document)
	tfD = dict()
	tfQ = dict()
	Qterms = query
	for i in range(len(Dterms)):
		tfD[Dterms[i]] = oktf(Dterms[i], document)
	for j in range(len(Qterms)):
		tfQ[Qterms[j]] = oktf(Qterms[j], query)

	return sim(tfQ, tfD)


# IMPLEMENTATION FUNCTIONS FOR TF-IDF SCORING FROM HERE ONWARDS

# returns the tf-idf, given a term and document
def tf_idf(term, document):
	D = TOTALDOCS
	return float(oktf(term, document) * math.log(D / df(term)))


# returns the similarity between document and query (scores them) based on tf-idf method
def TF_IDF(document, query):
	Dterms, _ = getDocTerms(document)
	tfD = dict()
	tfQ = dict()
	Qterms = query
	for i in range(len(Dterms)):
		tfD[Dterms[i]] = tf_idf(Dterms[i], document)
	for j in range(len(Qterms)):
		tfQ[Qterms[j]] = tf_idf(Qterms[j], query)

	return sim(tfQ, tfD)


# IMPLEMENTATION FUNCTIONS FOR OKAPI BM25 FROM HERE ONWARDS

# returns K as required by the BM25 formula
def K(document, k1, b):
	avglen = getAvgFieldLength()
	return float(k1 * ((1 - b) + (b * (length(document) / avglen))))


# returns the BM25 score for a single query term, given the term, document vector, query vector
def BM25(term, document, query):
	D = TOTALDOCS
	k1 = 1.2
	k2 = 500
	b = 0.75
	return float(math.log((D + 0.5) / (df(term) + 0.5))) \
		* (((1 + k1) * tf_mem(term, document)) / (K(document, k1, b) + tf_mem(term, document))) \
		* (((1 + k2) * tf_mem(term, query)) / (k2 + tf_mem(term, query)))


# scores the document and query pair by taking a summation of the BM25 scores of each query term
def okapi_BM25(document, query):
	s = 0
	s = s + sum([BM25(query[i], document, query) for i in range(len(query))])
	return s


# IMPLEMENTATION OF JELINEK-MERCER SMOOTHING
def totalLength():
	sum = 0
	for i in range(1, TOTALDOCS):
		sum = sum + int(length(i))
	return sum

def Jelinek_Mercer_smoothing(term, document):
	collectionLength = totalLength()
	lmbda = 0.6
	return lmbda*(tf_mem(term, document)/length(document)) + ((1-lmbda)*(getCumulativeFrequency(term)/collectionLength))


def JM(document, query):
	Dterms, _ = getDocTerms(document)
	tfD = dict()
	tfQ = dict()
	Qterms = query
	for i in range(len(Dterms)):
		tfD[Dterms[i]] = Jelinek_Mercer_smoothing(Dterms[i], document)
	for j in range(len(Qterms)):
		tfQ[Qterms[j]] = Jelinek_Mercer_smoothing(Qterms[j], query)
	return sim(tfQ, tfD)


# returns the cumulative frequency of a term
def getCumulativeFrequency(term):
	with open("term_info.txt", "r", encoding="utf8") as term_info:
		for line in term_info:
			if term in line:
				data = line.split()
				if term == data[0]:
					return int(data[2])

	del term_info
	return 0


# returns all the documents of a term using the deltaDecodeDocs function
def getAllDocs(term):
	docs, _ = deltaDecodeDocs(invertedIndex[term])
	return docs


# return all the documents of multiple terms in a single list. For convenient retrieval of all related docs of the query
def getAllDocsOfaQuery(query):
	d = []
	for i in range(len(query)):
		d.append(getAllDocs(getTermID(query[i])))

	d = list(set(list(itertools.chain.from_iterable(d))))

	return d


tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*')
stemmer = PorterStemmer()
stopWords = open(os.getcwd() + r"\\stoplist.txt").read()

readfile = open(os.getcwd() + "\\topics.xml", encoding='utf-8', errors='ignore').read()
parsed = parseThisShit(readfile)
tokens = textNormalize(parsed[0])

invertedIndex = loadInvertedIndex()
docLengths = loadLengths()

documents = getAllDocsOfaQuery(tokens)
query = [getTermID(tokens[i]) for i in range(len(tokens))]
scores
# print(query)
# i = 400
# print(okapi_tf(documents[i], query))
# print(TF_IDF(documents[i], query))
# print(okapi_BM25(documents[i], query))
# print(JM(documents[i], query))


print(len(documents))
# for i in range(1, len(documents)):
# 	print(str(documents[i]) + " : " + str(okapi_tf(documents[i], query)))