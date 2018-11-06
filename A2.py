import os
import sys
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import itertools

TOTALDOCS = 3496


class scoring:
	invertedIndex = {}
	docLengths = {}
	docNormOktf = {}
	docNormtfidf = {}
	queries = {}
	tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*')
	stemmer = PorterStemmer()
	stopWords = open(os.getcwd() + r"\\stoplist.txt").read()

	def __init__(self):
		self.invertedIndex = self.loadInvertedIndex()
		self.docLengths = self.loadLengths()
		self.queries = self.loadQueries()
		self.docNormOktf, self.docNormtfidf = self.loadNorms()

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
		with open("term_index.txt") as file:
			while (True):
				l = file.readline()
				if l is "":
					break
				else:
					p = l.split("\t")
					p[len(p) - 1] = p[len(p) - 1].rstrip()
					self.invertedIndex[p[0]] = p[1:]
		return self.invertedIndex

	def loadNorms(self):
		for doc in range(1, TOTALDOCS):
			with open("doc_norms_Oktf.txt") as file1:
				while (True):
					l = file1.readline()
					if l is "":
						break
					else:
						p = l.split("\t")
						self.docNormOktf[p[0]] = p[1].rstrip()

			with open("doc_norms_tfidf.txt") as file1:
				while (True):
					l = file1.readline()
					if l is "":
						break
					else:
						p = l.split("\t")
						self.docNormtfidf[p[0]] = p[1]

			if self.docNormtfidf is not None and self.docNormOktf is not None:
				return self.docNormOktf, self.docNormtfidf

			Dterms, _ = self.getDocTerms(doc)
			oktf = dict()
			tfidf = dict()
			for i in range(len(Dterms)):
				oktf[Dterms[i]] = self.oktf(Dterms[i], doc)
				tfidf[Dterms[i]] = self.tf_idf(Dterms[i], doc)

			self.docNormOktf[doc] = self.vectorlength(list(oktf.values()))
			self.docNormtfidf[doc] = self.vectorlength(list(tfidf.values()))

			with open("doc_norms_Oktf.txt", "a") as file1:
				file1.write(str(doc) + "\t" + str(self.docNormOktf[doc]) + "\n")

			with open("doc_norms_tfidf.txt", "a") as file2:
				file2.write(str(doc) + "\t" + str(self.docNormtfidf[doc]) + "\n")

		return self.docNormOktf, self.docNormtfidf

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

	def isIndex(self, ind):
		if ind <= len(self.docNormtfidf):
			return True
		else:
			return False

	# returns the Length of a document. (This includes the count of all the terms' positions in the document)
	def Length(self, document):
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

	# returns the average document Length in the corpus
	def getAvgFieldLength(self):
		return self.totalLength() / TOTALDOCS

	# determines the cosine similarity between the query and doc
	# both the queryTerms and docTerms are dictionary having query/doc as keys and their tf/oktf/tf-idf as the value
	# queryTerms [query term number 1] = oktf/tf/tf-idf of the query term number 1
	def sim(self, query, document, queryTerms, docTerms):
		if self.isIndex(document):
			dlen = float(self.docNormtfidf[str(document)])
		else:
			Dterms = []
			tfD = {}
			Dterms = self.getDocTerms(document)
			for i in range(len(Dterms)):
				tfD[Dterms[i]] = self.tf_idf(Dterms[i], document)
			dlen = self.vectorlength(list(tfD.values()))
		qlen = self.vectorlength(list(queryTerms.values()))
		dxq = 0
		for key in queryTerms.keys():
			if key in docTerms.keys():
				dxq = dxq + queryTerms[key] * docTerms[key]

		if dlen == 0 or qlen == 0:
			return 0
		return dxq / (dlen * qlen)

	# this returns the norm or the vector Length. i.e By pathagoras theorem, square root of squared sum of all components
	def vectorlength(self, tfs):
		return math.sqrt(sum([tfs[i]*tfs[i] for i in range(len(tfs))]))

	# IMPLEMENTATION FUNCTIONS OF OKAPI-TF FROM HERE ONWARDS
	# returns the oktf score given a document and the term
	def oktf(self, term, document):
		avglen = self.getAvgFieldLength()
		freq = self.tf_mem(term, document)
		return float(self.tf_mem(term, document) / (self.tf_mem(term, document) + 0.5 + 1.5 * (float(self.Length(document)) / avglen)))

	# gets the document and query and computes their similarity (scores them) based on the okapi_tf method
	def okapi_tf(self, document, query):
		tfD = dict()
		tfQ = dict()
		Qterms = query
		for j in range(len(Qterms)):
			if self.isTermInDoc(Qterms[j], document):
				tfD[Qterms[j]] = self.oktf(Qterms[j], document)
			tfQ[Qterms[j]] = self.oktf(Qterms[j], query)

		return self.sim(query, document, tfQ, tfD)

	# IMPLEMENTATION FUNCTIONS FOR TF-IDF SCORING FROM HERE ONWARDS
	# returns the tf-idf, given a term and document
	def tf_idf(self, term, document):
		D = TOTALDOCS
		return float(self.oktf(term, document) * math.log(D / self.df(term)))

	# returns the similarity between document and query (scores them) based on tf-idf method
	def TF_IDF(self, document, query):
		tfD = dict()
		tfQ = dict()
		Qterms = query
		for j in range(len(Qterms)):
			if self.isTermInDoc(Qterms[j], document):
				tfD[Qterms[j]] = self.tf_idf(Qterms[j], document)
			tfQ[Qterms[j]] = self.tf_idf(Qterms[j], query)

		return self.sim(query, document, tfQ, tfD)

	# IMPLEMENTATION FUNCTIONS FOR OKAPI BM25 FROM HERE ONWARDS
	# returns K as required by the BM25 formula
	def K(self, document, k1, b):
		avglen = self.getAvgFieldLength()
		return float(k1 * ((1 - b) + (b * (self.Length(document) / avglen))))

	# returns the BM25 score for a single query term, given the term, document vector, query vector
	def BM25(self, term, document, query):
		D = TOTALDOCS
		k1 = 1.2
		k2 = 100
		b = 0.75
		dfreq = self.tf_mem(term, document)
		qfreq = self.tf_mem(term, query)
		return float(math.log((D + 0.5) / (self.df(term) + 0.5))) \
			* (((1 + k1) * dfreq) / (self.K(document, k1, b) + dfreq)) \
			* (((1 + k2) * qfreq) / (k2 + qfreq))

	# scores the document and query pair by taking a summation of the BM25 scores of each query term
	def okapi_BM25(self, document, query):
		return sum([self.BM25(query[i], document, query) for i in range(len(query))])

	# IMPLEMENTATION OF JELINEK-MERCER SMOOTHING
	def totalLength(self):
		return sum([int(self.Length(i)) for i in range(1, TOTALDOCS)])

	def Jelinek_Mercer_smoothing(self, term, document):
		collectionLength = self.totalLength()
		lmbda = 0.6
		return (lmbda*(self.tf_mem(term, document)/self.Length(document))) + ((1 - lmbda) * (self.getCumulativeFrequency(term) / collectionLength))

	def JM(self, document, query):
		s=1
		for i in range(len(query)):
			s = s*self.Jelinek_Mercer_smoothing(query[i], document)
		return s
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

	def score(self, method, document, query):
		if len(query) == 0:
			return 0
		m = method.lower()
		if m == "okapibm25":
			return self.okapi_BM25(document, query)
		elif m == "jm":
			return self.JM(document, query)
		elif m == "tfidf":
			return self.TF_IDF(document, query)
		elif m == "okapitf":
			return self.okapi_tf(document, query)
		else:
			print("Invalid scoring method chosen.")
			exit()


	def checkTermsIndoc(self, document, query):
		s = 0
		for i in range(len(query)):
			if self.isTermInDoc(query[i], document) is True:
				s = s + 1
		return s

	def isTermInDoc(self, term, doc):
		postings = self.invertedIndex[str(term)]
		docs, _ = self.deltaDecodeDocs(postings)
		if doc in docs:
			return True
		else:
			return False



s1 = scoring()
method = input("Enter your method:\t")
open(method + '.txt', 'w').close()
print(s1.queries)
for index in range(len(s1.queries)):

	tokens = s1.tokens(s1.queries[index][1])
	query = [s1.getTermID(tokens[i]) for i in range(len(tokens))]
	documents = s1.getAllDocsOfaQuery(query)
	documents.sort()
	scores = []

	for i in range(1, TOTALDOCS):
		if i in documents:
			scores.append(s1.score(method, i, query))
		else:
			scores.append(0.0)
		print(str(i) + "\t" + str(scores[i-1]))


	indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
	scores.sort(reverse=True)
	with open(method+".txt", "a") as s:
		[s.write(str(s1.queries[index][0]) + "\t" + str(0) + "\t" + s1.getDocTitle(indices[_]+1) + "\t" + str(_+1) + "\t" + str(scores[_]) + "\t" + "run1" + "\n") for _ in range(0, len(scores))]


# s1 = scoring()
# maxLen = 0
# minI = 0
# minLen = 0
# maxI = 0
# for i in range(len(s1.queries)):
# 	tokens = s1.tokens(s1.queries[i][1])
# 	query = [s1.getTermID(tokens[i]) for i in range(len(tokens))]
# 	documents = s1.getAllDocsOfaQuery(query)
# 	L = len(documents)
# 	print(i, "\t", L)
# 	if i == 0:
# 		minLen = L
# 		maxLen = L
# 		minI = s1.queries[i][0]
# 		maxI = s1.queries[i][0]
# 	if L < minLen:
# 		minLen = L
# 		minI = s1.queries[i][0]
# 	if L > maxLen:
# 		maxLen = L
# 		maxI = s1.queries[i][0]
#
# print("Max: ", maxI, "\t", maxLen)
# print("Min: ", minI, "\t", minLen)