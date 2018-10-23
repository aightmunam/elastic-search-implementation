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

def getListingsforDoc(docTitle):
	d = docTitle
	doc = getDocID(d)
	distinctTerms = 0
	totalTerms = 0
	with open("doc_index.txt") as docINFO:
		for line in docINFO:
			if doc in line:
				data = line.split()
				if doc == data[0]:
					distinctTerms = distinctTerms + 1
					totalTerms = totalTerms + len(data[2:len(data)])


	return distinctTerms, totalTerms



def getAllTerms(document):
	freq = []
	flag = False
	doc = str(document)
	with open("doc_index.txt") as docINFO:
		for line in docINFO:
			if doc in line:
				data = line.split()
				if doc == data[0]:
					l = (data[2:len(data)])
					freq.append(l)
					flag = True
				if doc != data[0] and flag is True:
					break
	freq = list(itertools.chain.from_iterable(freq))
	return freq

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

def deltaDecodeDocs(postings):
	docs = list()
	positions = dict()
	newD = 0
	newT = 0
	term = []
	for _ in range(1,len(postings)):
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


	return docs, positions



def tf_mem(term, document):
	freq = 0
	doc = str(document)
	if doc not in invertedIndex.keys():
		return 0
	postings = invertedIndex[document]
	_, positions = deltaDecodeDocs(postings)
	return positions




def length(document):
	count = 0
	doc = str(document)
	with open("doc_lengths.txt") as docINFO:
		for line in docINFO:
			if doc in line:
				data = line.split()
				if doc == data[0]:
					return data[1]



def df(term):
	with open("term_info.txt", "r") as termINFO:
		for line in termINFO:
			if term in line:
				data = line.split()
				if term == data[0]:
					return int(data[3])



def df_mem(id):
	newD = 0
	docs = []
	postings = invertedIndex[id]
	docs = deltaDecodeDocs(postings)
	return docs


def avglength(totalDocs):
	sum = 0
	for i in range(1, totalDocs):
		sum = sum + int(length(i))
	return sum/totalDocs


def oktf(term, document):
	avglen = avglength(TOTALDOCS)
	return int(tf(term, document) / (tf(term, document) + 0.5 + 1.5 * (int(length(document)) / avglen)))


def otaki_tf(document, query):
	Dterms = getAllTerms(document)
	tfD = dict()
	tfQ = dict()
	Qterms = query
	for i in range(len(Dterms)):
		tfD[Dterms[i]] = oktf(Dterms[i], document)
	for j in range(len(Qterms)):
		tfQ[Qterms[j]] = oktf(Qterms[j], query)

	dlen = vectorlength(tfD.values())
	qlen = vectorlength(tfQ.values())

	dxq = 0
	for key in tfQ.keys():
		if key in tfD.keys():
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
	for i in range(len(tfs)):
		s = s + (tfs[i] * tfs[i])

	return math.sqrt(s)



def getAllDocsOfaQuery(query):
	d = []
	for i in range(len(query)):
		d.append(getAllDocs(getTermID(tokens[i])))
	d = list(itertools.chain.from_iterable(d))
	return d



tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*')
stemmer = PorterStemmer()
stopWords = open(os.getcwd()+r"\\stoplist.txt").read()


readfile = open(os.getcwd() + "\\topics.xml", encoding='utf-8', errors = 'ignore').read()

parsed = parseThisShit(readfile)
tokens = textNormalize(parsed[0])


invertedIndex = dict()
with open("term_index.txt") as ii:
	while(True):
		l = ii.readline()
		if l is "":
			break
		else:
			p = l.split("\t")

			invertedIndex[p[0]] = p[1:]

print(df('1'))
print(len(df_mem('1')))

print(invertedIndex['1'])
docs, terms = deltaDecodeDocs(invertedIndex['1'])
print(docs)
print(terms)
# documents = getAllDocsOfaQuery(tokens)
# query = []
# for i in range(len(tokens)):
# 	query.append(getTermID(tokens[i]))
#
# s = otaki_tf(documents[0], query)



# print((documents))


# print(oktf(id, documents[0]))

# print(tf(id,documents[0]))
# print(documents)


