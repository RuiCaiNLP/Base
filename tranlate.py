#coding=utf-8
from googletrans import Translator
file = open('./temp/flat_train_en', 'r')
out_file = open('flat_train_fr_2', 'w')
translator = Translator()
for line in file.readlines():
    line =translator.translate(line, dest='fr')
    out_file.write(line.text.encode('utf-8'))
    out_file.write('\n')

