import glob
import os
import nltk
from nltk.tokenize import RegexpTokenizer
import pickle
import json
import time
import lxml
from lxml.html.clean import Cleaner
from bs4 import BeautifulSoup, Comment


def clean_me(html):
    soup = BeautifulSoup(html, 'html.parser')
    soup = soup.body
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
              for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text


tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

directory_in_str = os.path.dirname(
    os.path.realpath(__file__)) + r"\\corpus\*.txt"

indexingDir = os.path.dirname(
    os.path.realpath(__file__)) + r"\\corpus\\"

directory = os.fsencode(directory_in_str)
tokenList = {}
docMapping = {}
tokenLocs = {}


limit = 1

i = 1
k = 1
open('mapping.txt', 'w').close()
open('terms.txt', 'w').close()
for filename in os.listdir(indexingDir):

    tmp = os.path.basename(filename)
    docTitle = tmp[:tmp.rfind(".")]
    print(docTitle)
    limit = limit + 1
    docMapping[docTitle] = i
    i = i + 1

    readfile = open(indexingDir + os.path.basename(filename), encoding='utf-8', errors = 'ignore').read()



    text = parseThisShit(readfile)

    tokens = nltk.word_tokenize(text)
    
    # tokens = tokenizer.tokenize(readfile)

    for j in range(0, len(tokens)):
        if tokens[j] not in tokenList:
            newToken = str(k) + r"|" + tokens[j]
            doc = os.path.basename(filename)[
                :os.path.basename(filename).rfind(".")]
            tokenList[tokens[j]] = doc
            # indices = {x for x, y in enumerate(tokens) if y == tokens[j]}
            # tokenLocs[tokens[j]] = [doc, i, indices]

            j = j + 1
            k = k + 1
        else:
            j = j + 1

# print(docMapping)
# print(limit)
# with open('terms.txt', 'w') as fp:
#     fp.write(json.dumps(tokenList))

z = 1
with open('mapping.txt', 'a') as map_doc:
    for key in docMapping.keys():
        map_doc.write(str(z) + "|" + key)
        map_doc.write("\n")
        z = z + 1


z = 1
with open('terms.txt', 'a') as f:
    for key in tokenList.keys():
        f.write(str(z) + r"|" + key)
        f.write("\n")
        z = z + 1
