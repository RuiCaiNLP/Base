# Implemented by Charles(charlee@sjtu.edu.cn) & Shexia He(heshexia@sjtu.edu.cn).
# This file is used for data process.

import os
from data_utils import *


def make_dataset():
    base_path = os.path.join(os.path.dirname(__file__), 'data/CoNLL-2009-Datasets')

    # because the train and dev file is with full format, wo just copy them
    #raw_train_file = os.path.join(base_path, 'CoNLL2009-ST-English-development.map')
    raw_train_file = os.path.join(base_path, 'CoNLL2009-ST-English-train.txt')
    unlabeled_train_file = os.path.join(base_path, 'fr-train')
    raw_dev_file = os.path.join(base_path,
                                'FR.Datasets')


    train_file = os.path.join(os.path.dirname(__file__), 'data/En_train.dataset')
    dev_file = os.path.join(os.path.dirname(__file__), 'data/Fr_dev.dataset')
    unlabeled_file = os.path.join(os.path.dirname(__file__), 'data/Unlabeled.dataset')
    #test_file = os.path.join(os.path.dirname(__file__), 'data/conll09-english/conll09_test.dataset')
    #test_ood_file = os.path.join(os.path.dirname(__file__), 'data/conll09-english/conll09_test_ood.dataset')

    # for train
    with open(raw_train_file, 'r') as fin:
        with open(train_file, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for dev
    with open(raw_dev_file, 'r') as fin:
        with open(dev_file, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for dev
    with open(unlabeled_train_file, 'r') as fin:
        with open(unlabeled_file, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    """
    # for test
    with open(raw_eval_file, 'r') as fin:
        with open(raw_eval_file_head, 'r') as fhead:
            with open(raw_eval_file_deprel, 'r') as fdeprel:
                with open(raw_eval_file_pred_apreds, 'r') as fpredapreds:
                    with open(test_file, 'w') as fout:
                        while True:
                            raw_line = fin.readline()
                            if len(raw_line) == 0:
                                break
                            head_line = fhead.readline()
                            deprel_line = fdeprel.readline()
                            pred_apreds_line = fpredapreds.readline()
                            if len(raw_line.strip()) > 0:
                                raw_line = raw_line.strip().split('\t')
                                head_line = head_line.strip().split('\t')
                                deprel_line = deprel_line.strip().split('\t')
                                pred_apreds_line = pred_apreds_line.strip().split('\t')
                                raw_line[8] = head_line[0]
                                raw_line[10] = deprel_line[0]
                                raw_line += pred_apreds_line
                                fout.write('\t'.join(raw_line))
                                fout.write('\n')
                            else:
                                fout.write(raw_line)

    # for test ood
    with open(raw_eval_ood_file, 'r') as fin:
        with open(raw_eval_ood_file_head, 'r') as fhead:
            with open(raw_eval_ood_file_deprel, 'r') as fdeprel:
                with open(raw_eval_ood_file_pred_apreds, 'r') as fpredapreds:
                    with open(test_ood_file, 'w') as fout:
                        while True:
                            raw_line = fin.readline()
                            if len(raw_line) == 0:
                                break
                            head_line = fhead.readline()
                            deprel_line = fdeprel.readline()
                            pred_apreds_line = fpredapreds.readline()
                            if len(raw_line.strip()) > 0:
                                raw_line = raw_line.strip().split('\t')
                                head_line = head_line.strip().split('\t')
                                deprel_line = deprel_line.strip().split('\t')
                                pred_apreds_line = pred_apreds_line.strip().split('\t')
                                raw_line[8] = head_line[0]
                                raw_line[10] = deprel_line[0]
                                raw_line += pred_apreds_line
                                fout.write('\t'.join(raw_line))
                                fout.write('\n')
                            else:
                                fout.write(raw_line)

    # because the train and dev file is with full format, wo just copy them
    raw_train_file = os.path.join(base_path, 'CoNLL2009-ST-Chinese-train/CoNLL2009-ST-Chinese-train.txt')
    raw_dev_file = os.path.join(base_path, 'CoNLL2009-ST-Chinese-dev/CoNLL2009-ST-Chinese-development.txt')

    # because the eval file is lack of 9, 11, 14, 15 so we need to merge them
    raw_eval_file = os.path.join(base_path, 'CoNLL2009-ST-eval-Ch-SRL/CoNLL2009-ST-evaluation-Chinese-SRLonly.txt')
    raw_eval_file_head = os.path.join(base_path,
                                      'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-Chinese.9.HEAD.txt')
    raw_eval_file_deprel = os.path.join(base_path,
                                        'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-Chinese.11.DEPREL.txt')
    raw_eval_file_pred_apreds = os.path.join(base_path,
                                             'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-Chinese.14-.PRED_APREDs.txt')
    """


def stat_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        data = f.readlines()

        # read data
        sentence_data = []
        sentence = []
        for line in data:
            if len(line.strip()) > 0:
                line = line.strip().split('\t')
                sentence.append(line)
            else:
                sentence_data.append(sentence)
                sentence = []

    predicate_number = 0
    non_predicate_number = 0
    argument_number = 0
    non_argument_number = 0
    predicate_dismatch = 0
    uas_correct = 0
    las_correct = 0
    syntactic_sum = 0
    for sentence in sentence_data:
        for item in sentence:
            syntactic_sum += 1
            if item[8] == item[9]:
                uas_correct += 1
            if item[8] == item[9] and item[10] == item[11]:
                las_correct += 1
            if item[12] == 'Y':
                predicate_number += 1
            else:
                non_predicate_number += 1
            if (item[12] == 'Y' and item[12] == '_') or (item[12] == '_' and item[12] != '_'):
                predicate_dismatch += 1
            for i in range(len(item) - 14):
                if item[14 + i] != '_':
                    argument_number += 1
                else:
                    non_argument_number += 1

    # sentence number
    # predicate number
    # argument number
    print(
    '\tsentence:{} \n\tpredicate:{} non predicate:{} predicate dismatch:{} \n\targument:{} non argument:{} \n\tUAS:{:.2f} LAS:{:.2f}'
    .format(len(sentence_data), predicate_number, non_predicate_number, predicate_dismatch, argument_number,
            non_argument_number, uas_correct / syntactic_sum * 100, las_correct / syntactic_sum * 100))


if __name__ == '__main__':
    # make train/dev/test dataset
    make_dataset()

    train_file = os.path.join(os.path.dirname(__file__), 'data/En_train.dataset')
    dev_file = os.path.join(os.path.dirname(__file__), 'data/Fr_dev.dataset')
    unlabeled_file = os.path.join(os.path.dirname(__file__), 'data/Unlabeled.dataset')

    # make_dataset_input

    make_dataset_input(train_file, os.path.join(os.path.dirname(__file__), 'temp/train.input'), unify_pred=False)
    make_dataset_input(dev_file, os.path.join(os.path.dirname(__file__), 'temp/dev.input'), unify_pred=False)
    make_dataset_input(unlabeled_file, os.path.join(os.path.dirname(__file__), 'temp/unlabeled.input'), unify_pred=False)


    # make word/pos/lemma/deprel/argument vocab
    print('\n-- making (word/lemma/pos/argument) vocab --')
    vocab_path = os.path.join(os.path.dirname(__file__), 'temp')
    print('word:')
    make_word_vocab(train_file, vocab_path, unify_pred=False)
    print('fr word:')
    fr_make_word_vocab(unlabeled_file, vocab_path, unify_pred=False)
    print('pos:')
    make_pos_vocab(train_file, vocab_path, unify_pred=False)
    print('lemma:')
    make_lemma_vocab(train_file, vocab_path, unify_pred=False)
    print('deprel:')
    make_deprel_vocab(train_file, vocab_path, unify_pred=False)
    print('argument:')
    make_argument_vocab(dev_file, dev_file, None, vocab_path, unify_pred=False)
    #print('predicate:')
    #make_pred_vocab(train_file, dev_file, None, vocab_path)

    pretrain_path = os.path.join(os.path.dirname(__file__), 'temp')
    deprel_vocab = load_deprel_vocab(os.path.join(pretrain_path, 'deprel.vocab'))
    # shrink pretrained embeding
    print('\n-- shrink pretrained embeding --')
    pretrain_file = os.path.join(os.path.dirname(__file__), 'data/en.vec.txt')  # words.vector
    pretrained_emb_size = 300

    shrink_pretrained_embedding(train_file, train_file, train_file, pretrain_file, pretrained_emb_size, pretrain_path)

    print('\n-- shrink french pretrained embeding --')
    pretrain_file_fr = os.path.join(os.path.dirname(__file__), 'data/fr.vec.txt')  # words.vector
    pretrained_emb_size_fr = 300
    pretrain_path_fr = os.path.join(os.path.dirname(__file__), 'temp')
    fr_shrink_pretrained_embedding(unlabeled_file, dev_file, dev_file, pretrain_file_fr, pretrained_emb_size_fr, pretrain_path_fr)

    make_dataset_input(train_file, os.path.join(pretrain_path, 'train.input'), unify_pred=False,
                       deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(pretrain_path, 'train.pickle.input'))
    make_dataset_input(dev_file, os.path.join(pretrain_path, 'dev.input'), unify_pred=False, deprel_vocab=deprel_vocab,
                       pickle_dump_path=os.path.join(pretrain_path, 'dev.pickle.input'))
    make_dataset_input(unlabeled_file, os.path.join(pretrain_path, 'unlabeled.input'), unify_pred=False, deprel_vocab=deprel_vocab,
                       pickle_dump_path=os.path.join(pretrain_path, 'unlabeled.pickle.input'))

    log(' data preprocessing finished!')
    #
    # make_pred_dataset_input(train_file, os.path.join(os.path.dirname(__file__),'temp/pred_train.input'))
    # make_pred_dataset_input(dev_file, os.path.join(os.path.dirname(__file__),'temp/pred_dev.input'))
    # make_pred_dataset_input(test_file, os.path.join(os.path.dirname(__file__),'temp/pred_test.input'))

    # make_pred_recog_dataset_input(train_file, os.path.join(os.path.dirname(__file__),'temp/pred_recog_train.input'))
    # make_pred_recog_dataset_input(dev_file, os.path.join(os.path.dirname(__file__),'temp/pred_recog_dev.input'))
    # make_pred_recog_dataset_input(test_file, os.path.join(os.path.dirname(__file__),'temp/pred_recog_test.input'))

