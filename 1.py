import os
import sys
import nltk
import time
import codecs
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup, Comment


def index_exists(ls, i):
    return (0 <= i < len(ls)) or (-len(ls) <= i < 0)


def clean_me(html):
    soup = BeautifulSoup(html, 'html.parser')
    if soup.html is None:
        return ''
    else:
        soup = soup.html
    for s in soup(['script', 'style']):
        s.decompose()
    return ' '.join(soup.stripped_strings)


def parseThisShit(html):
    soup = BeautifulSoup(readfile, 'html.parser')
    if soup.html is None:
        return ''
    else:
        soup = soup.html

    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip()
              for line in lines for phrase in line.split(" "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text

def textNormalize(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [stemmer.stem(i) for i in tokens if i not in stopWords]
    return tokens


if len(sys.argv) > 1:
    indexingDir = os.getcwd() + r'\\' + sys.argv[1]
else:
    print("ERROR! Please enter the name of a directory containing the corpus documents.")
    exit()

if not os.path.exists(indexingDir):
    print("ERROR '" + indexingDir + "' does not exist or it is an invalid directory.")
    exit()


tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*')
# tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
stemmer = PorterStemmer()
stopWords = open(os.getcwd()+r"\\stoplist.txt").read()


#Clearing the output files
open('docids2.txt', 'w').close()
open('termids2.txt', 'w').close()
open('doc_index2.txt', 'w').close()

termList={}
docID = 0
termID = 0
l = len(os.listdir(indexingDir))
for filename in os.listdir(indexingDir):
    tempList = {}
    docToTerm = []
    docIndex = 0
    tmp = os.path.basename(filename)
    docTitle = tmp
    # print(docTitle)
    docID = docID + 1

    print("%.2f" % round(((docID/l)*100), 2) + r"%\t" + docTitle + "\n");

    with codecs.open('docids2.txt', 'a', encoding='utf8') as map_doc:
        map_doc.write(str(docID) + "\t" + docTitle + "\r\n")

    readfile = open(indexingDir + r"//" + tmp, encoding='utf-8', errors = 'ignore').read()

    text = parseThisShit(readfile)
    del readfile
    tokens = textNormalize(text)
    for j in range(0, len(tokens)):
        if tokens[j] not in termList:
            termID = termID + 1
            termList[tokens[j]] = termID
            with codecs.open('termids2.txt', 'a', encoding='utf8') as term_doc:
                term_doc.write(str(termList[tokens[j]]) + "\t" + tokens[j] + "\r\n")

        if tokens[j] not in tempList:
            tempList[tokens[j]] = docIndex
            docToTerm.append([termList[tokens[j]], j])
            docIndex = docIndex+1

        else:
            if index_exists(docToTerm, docID - 1):

                list = docToTerm[tempList[tokens[j]]]
                if list[0] == termList[tokens[j]]:
                    list.append(j)
                    docToTerm[tempList[tokens[j]]] = list
        j = j + 1

    with codecs.open('doc_index2.txt', 'a', encoding= 'utf8') as forwardIndex:
        for x in range(len(docToTerm)):
            forwardIndex.write(str(docID) + "\t" + '\t'.join([str(z) for z in docToTerm[x]]) + "\r\n")
    del forwardIndex
    del tempList
    del docToTerm


# with codecs.open('termids2.txt', 'w', encoding='utf8') as term_doc:
#     for key, value in termList.items():
#         term_doc.write(str(value) + "\t" + key)
#         term_doc.write("\r\n")

