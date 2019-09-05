#coding=utf-8

import re
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)


pattern = r"[dnl][']|\w+|\$[\d\.]+|\S+"
french_tokenizer = RegexpTokenizer(pattern)

def normalize(token):
    penn_tokens = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}'
    }

    if token in penn_tokens:
        return penn_tokens[token]

    token = token.lower()
    try:
        int(token)
        return "<NUM>"
    except:
        pass
    try:
        float(token.replace(',', ''))
        return "<FLOAT>"
    except:
        pass

    return token

"""
file1 = open("europarl-v7.fr-en.fr", 'r')
file2 = open("europarl-v7.fr-en.en", 'r')

file1_fucked = open("fucked.fr-en.fr", 'w')
file2_fucked = open("fucked.fr-en.en", 'w')
iii = 1
for sens1, sens2 in zip(file1.readlines(), file2.readlines()):
    if len(sens1)<= 1 or len(sens2)<= 1:
        continue
    if sens1.startswith('<') or sens2.startswith('<'):
        print(sens1)
        print(sens2)

    words1 = french_tokenizer.tokenize(unicode(sens1.strip()))

    words1 = [normalize(w) for w in words1]
    sent1 = ' '.join(words1)
    file1_fucked.write(sent1)
    file1_fucked.write('\n')

    words2 = word_tokenize(unicode(sens2.strip()))

    words2 = [normalize(w) for w in words2]
    sent2 = ' '.join(words2)
    file2_fucked.write(sent2)
    file2_fucked.write('\n')
"""
file = open("flat_train_fr", 'r')
file_cleaned = open("flat_train_fr_cleaned", 'w')
for sen in file.readlines():
    words = french_tokenizer.tokenize(unicode(sen.strip()))

    predicate_idx = -1
    new_words= []
    i = 0
    while i < len(words):
        if words[i].startswith("\"") and words[i].endswith("\""):
            words[i] = words[i][1:-1]
            predicate_idx = i
        elif words[i].startswith("«") and words[i].endswith("»"):
            words[i] = words[i][1:-1]
            predicate_idx = i
        i+=1

    words = [normalize(w) for w in words]
    sent = ' '.join(words)
    file_cleaned.write(str(predicate_idx))
    file_cleaned.write('\n')
    file_cleaned.write(sent)
    file_cleaned.write('\n')

