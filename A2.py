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


class scoring:
	invertedIndex = {}
	docLengths = {}
	queries = {}
	tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*')
	stemmer = PorterStemmer()
	stopWords = open(os.getcwd() + r"\\stoplist.txt").read()

	def __init__(self):
		self.invertedIndex = self.loadInvertedIndex()
		self.docLengths = self.loadLengths()
		self.queries = self.loadQueries()

	# loads the lengths of the documents
	def loadLengths(self):
		self.docLengths = dict()
		with open("doc_lengths.txt") as file:
			while (True):
				l = file.readline()
				if l is "":
					break
				else:
					p = l.split("\t")
					p[len(p) - 1] = p[len(p) - 1].rstrip()
					self.docLengths[p[0]] = p[1]

		return self.docLengths

	# loads our inverted index
	def loadInvertedIndex(self):
		self.invertedIndex = dict()
		with open("term_index.txt") as ii:
			while (True):
				l = ii.readline()
				if l is "":
					break
				else:
					p = l.split("\t")
					p[len(p) - 1] = p[len(p) - 1].rstrip()
					self.invertedIndex[p[0]] = p[1:]
		return self.invertedIndex

	def loadQueries(self):
		readfile = open(os.getcwd() + "\\topics.xml", encoding='utf-8', errors='ignore').read()
		self.queries = self.parseThisShit(readfile)
		return self.queries


	# normalizes the queries in the same way our inverted index is
	def textNormalize(self, text):
		token = self.tokenizer.tokenize(text)
		token = [t.lower() for t in token]
		token = [self.stemmer.stem(i) for i in token if i not in self.stopWords]
		return token

	# parses our queries
	def parseThisShit(self, html):
		queries = []
		soup1 = BeautifulSoup(html, 'html.parser')
		soup1 = soup1.find_all('topic')
		for i in range(len(soup1)):
			num = soup1[i]['number']
			soup = soup1[i].query
			text = soup.get_text()
			lines = (line.strip() for line in text.splitlines())
			chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
			text = ' '.join(chunk for chunk in chunks if chunk)
			queries.append([num, text])
		return queries


	# returns the term id
	def getTermID(self, term):
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
	def getDocID(self, doc):
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

	def getDocTitle(self, doc):
		d = str(doc)
		doc = ""
		with open("docids.txt", "r") as docIDs:
			for line in docIDs:
				if d in line:
					data = line.split()
					if d == data[0]:
						doc = data[1]
		del docIDs
		return doc

	# Returns the docs in the postings list of given term
	# It also returns a dictionary which stores the list of positions of the given term in every document it appears in
	# dict[document in which the term appears] = [list of all the positions it appears in the document]

	def deltaDecodeDocs(self, postings):
		docs = list()
		positions = dict()
		newD = 0
		newT = 0
		term = []
		for _ in range(len(postings)):
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
	def tf(self, term, document):
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
	def tf_mem(self, term, document):
		freq = 0
		doc = str(document)
		if term in self.invertedIndex.keys():
			_, positions = self.deltaDecodeDocs(self.invertedIndex[term])
			if type(document) is list:
				for i in range(len(document)):
					if str(term) in document[i]:
						freq = freq + 1
				return freq
			if doc not in positions:
				return 0
			return len(positions[doc])

	# returns the length of a document. (This includes the count of all the terms' positions in the document)
	def length(self, document):
		if str(document) in self.docLengths:
			return int(self.docLengths[str(document)])
		return len(document)

	# Returns all the terms of a document
	def getDocTerms(self, doc):
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


	# IMPLEMENTATION FUNCTIONS OF OKAPI-TF FROM HERE ONWARDS



	# returns all the documents, a term appears in
	def df(self, term):
		if str(term) not in self.invertedIndex.keys():
			return len(set(term))
		postings = self.invertedIndex[str(term)]
		docs, _ = self.deltaDecodeDocs(postings)
		return len(docs)


	# returns the average document length in the corpus
	def getAvgFieldLength(self):
		sum = 0
		for i in range(1, TOTALDOCS):
			sum = sum + int(self.length(i))
		return sum / TOTALDOCS



	# determines the cosine similarity between the query and doc
	# both the queryTerms and docTerms are dictionary having query/doc as keys and their tf/oktf/tf-idf as the value
	# queryTerms [query term number 1] = oktf/tf/tf-idf of the query term number 1
	def sim(self, queryTerms, docTerms):
		dlen = self.vectorlength(list(docTerms.values()))
		qlen = self.vectorlength(list(queryTerms.values()))
		dxq = 0
		for key in queryTerms.keys():
			if key in docTerms.keys():
				dxq = dxq + queryTerms[key] * docTerms[key]

		return dxq / (dlen * qlen)


	# this returns the norm or the vector length. i.e By pathagoras theorem, square root of squared sum of all components
	def vectorlength(self, tfs):
		s = 0
		for i in range(len(tfs)):
			s = s + (tfs[i] * tfs[i])
		return math.sqrt(s)


	# IMPLEMENTATION FUNCTIONS OF OKAPI-TF FROM HERE ONWARDS

	# returns the oktf score given a document and the term
	def oktf(self, term, document):
		avglen = self.getAvgFieldLength()
		freq = self.tf_mem(term, document)
		return float(self.tf_mem(term, document) / (self.tf_mem(term, document) + 0.5 + 1.5 * (int(self.length(document)) / avglen)))

	# gets the document and query and computes their similarity (scores them) based on the okapi_tf method
	def okapi_tf(self, document, query):
		Dterms, _ = self.getDocTerms(document)
		tfD = dict()
		tfQ = dict()
		Qterms = query
		for i in range(len(Dterms)):
			tfD[Dterms[i]] = self.oktf(Dterms[i], document)
		for j in range(len(Qterms)):
			tfQ[Qterms[j]] = self.oktf(Qterms[j], query)

		return self.sim(tfQ, tfD)


	# IMPLEMENTATION FUNCTIONS FOR TF-IDF SCORING FROM HERE ONWARDS

	# returns the tf-idf, given a term and document
	def tf_idf(self, term, document):
		D = TOTALDOCS
		return float(self.oktf(term, document) * math.log(D / self.df(term)))


	# returns the similarity between document and query (scores them) based on tf-idf method
	def TF_IDF(self, document, query):
		Dterms, _ = self.getDocTerms(document)
		tfD = dict()
		tfQ = dict()
		Qterms = query
		for i in range(len(Dterms)):
			tfD[Dterms[i]] = self.tf_idf(Dterms[i], document)
		for j in range(len(Qterms)):
			tfQ[Qterms[j]] = self.tf_idf(Qterms[j], query)

		return self.sim(tfQ, tfD)


	# IMPLEMENTATION FUNCTIONS FOR OKAPI BM25 FROM HERE ONWARDS

	# returns K as required by the BM25 formula
	def K(self, document, k1, b):
		avglen = self.getAvgFieldLength()
		return float(k1 * ((1 - b) + (b * (self.length(document) / avglen))))


	# returns the BM25 score for a single query term, given the term, document vector, query vector
	def BM25(self, term, document, query):
		D = TOTALDOCS
		k1 = 1.2
		k2 = 500
		b = 0.75
		return float(math.log((D + 0.5) / (self.df(term) + 0.5))) \
			* (((1 + k1) * self.tf_mem(term, document)) / (self.K(document, k1, b) + self.tf_mem(term, document))) \
			* (((1 + k2) * self.tf_mem(term, query)) / (k2 + self.tf_mem(term, query)))


	# scores the document and query pair by taking a summation of the BM25 scores of each query term
	def okapi_BM25(self, document, query):
		s = 0
		s = s + sum([self.BM25(query[i], document, query) for i in range(len(query))])
		return s


	# IMPLEMENTATION OF JELINEK-MERCER SMOOTHING
	def totalLength(self):
		sum = 0
		for i in range(1, TOTALDOCS):
			sum = sum + int(self.length(i))
		return sum

	def Jelinek_Mercer_smoothing(self, term, document):
		collectionLength = self.totalLength()
		lmbda = 0.6
		return lmbda*(self.tf_mem(term, document)/self.length(document)) + ((1-lmbda)*(self.getCumulativeFrequency(term)/collectionLength))


	def JM(self, document, query):
		Dterms, _ = self.getDocTerms(document)
		tfD = dict()
		tfQ = dict()
		Qterms = query
		for i in range(len(Dterms)):
			tfD[Dterms[i]] = self.Jelinek_Mercer_smoothing(Dterms[i], document)
		for j in range(len(Qterms)):
			tfQ[Qterms[j]] = self.Jelinek_Mercer_smoothing(Qterms[j], query)
		return self.sim(tfQ, tfD)


	# returns the cumulative frequency of a term
	def getCumulativeFrequency(self, term):
		with open("term_info.txt", "r", encoding="utf8") as term_info:
			for line in term_info:
				if term in line:
					data = line.split()
					if term == data[0]:
						return int(data[2])
		del term_info
		return 0


	# returns all the documents of a term using the deltaDecodeDocs function
	def getAllDocs(self, term):
		docs, _ = self.deltaDecodeDocs(self.invertedIndex[term])
		return docs


	# return all the documents of multiple terms in a single list. For convenient retrieval of all related docs of the query
	def getAllDocsOfaQuery(self, query):
		d = []
		for i in range(len(query)):
			d.append(self.getAllDocs(query[i]))
		d = list(set(list(itertools.chain.from_iterable(d))))
		return d

	def tokens(self, query):
		return self.textNormalize(query)


s1 = scoring()

tokens = s1.tokens(s1.queries[0][1])
query = [s1.getTermID(tokens[i]) for i in range(len(tokens))]
documents = s1.getAllDocsOfaQuery(query)
documents.sort()

scores = []
for i in range(1, TOTALDOCS):
	if i in documents:
		scores.append(s1.JM(i, query))
	else:
		scores.append(0.0)
	print(str(i) + " " + str(scores[i - 1]))


with open("scores2.txt", "w") as s:
	[s.write(str(202) + "\t" + str(0) + "\t" + s1.getDocTitle(i) + "\t" + str(scores[i]) + "\r\n") for i in range(1, len(scores))]

