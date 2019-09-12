#coding=utf-8
from fasttext import FastVector
fr_dictionary = FastVector(vector_file='wiki.en.vec')
fr_dictionary.export('fr.vec.txt')