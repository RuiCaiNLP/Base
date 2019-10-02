#coding=utf-8

import re
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer
import sys
import chardet
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'cannot'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'l\'+(\w)'.encode('utf8'), 'l\' \g<1>'.encode('utf8')),
(r'l\’+(\w)', 'l\' \g<1>'),
(r'qu\'+(\w)', 'qu\' \g<1>'),
(r'd\'+(\w)', 'd\' \g<1>'),
(r's\'+(\w)', 's\' \g<1>'),
(r'c\'+(\w)', 'c\' \g<1>'),
(r'n\'+(\w)', 'n\' \g<1>'),
(r'(\w+)%', '\g<1> %'),
(r'du', 'de le')]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s



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



if __name__ == '__main__':
    file = open("flat_train_fr", 'r')
    file_cleaned = open("flat_train_fr_cleaned_2", 'w')
    replacer = RegexpReplacer()
    for sen in file.readlines():

        sen = unicode(sen.strip())
        words = sen.split()
        new_sen =[]
        for word in words:
            word.replace(r"\’", "\'")
            if word.startswith("l\'") or word.startswith("n\'") or word.startswith("s\'") or word.startswith("c\'")\
                    or word.startswith("d\'") or word.startswith("l\’"):
                word = word = word[0:2]+" " + word[2:]
            if word.startswith("qu\'"):
                word = word = word[0:3]+" " + word[3:]
            if word == "du":
                word = "de le"
            if word.startswith("\"") and not word.endswith("\""):
                word = word[1:]
            if word.startswith("\'") and not word.endswith("\'"):
                word = word[1:]
            if not word.startswith("\'") and word.endswith("\'"):
                word = word[:-1]
            if not word.startswith("\"") and word.endswith("\""):
                word = word[:-1]
            if word.endswith("%"):
                word = word[:-1] + " " + word[-1]
            if word.endswith(",") or word.endswith("."):
                if len(word)>3:
                    word = word[:-1]+ " " +word[-1]

            new_sen.append(word)

        new_sentence = " ".join(new_sen)
        words = new_sentence.split()

        predicate_idx = -1
        new_words= []
        i = 0
        while i < len(words):
            if words[i].startswith("\"") and words[i].endswith("\""):
                words[i] = words[i][1:-1]
                predicate_idx = i
                break
            elif words[i].startswith("«") and words[i].endswith("»"):
                words[i] = words[i][1:-1]
                predicate_idx = i
                break
            i+=1

        words = [normalize(w) for w in words]
        sent = ' '.join(words)
        file_cleaned.write(str(predicate_idx))
        file_cleaned.write('\n')
        file_cleaned.write(sent)
        file_cleaned.write('\n')





