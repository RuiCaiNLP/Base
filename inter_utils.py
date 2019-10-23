from __future__ import print_function
from data_utils import _PAD_,_UNK_,_ROOT_,_NUM_
import math
import numpy as np
import random
import sys

def log(*args, **kwargs):
    print(*args,file=sys.stderr, **kwargs)

def pad_batch(batch_data, batch_size, pad_int):
    if len(batch_data) < batch_size:
        batch_data += [[pad_int]] * (batch_size - len(batch_data))
    max_length = max([len(item) for item in batch_data])
    return [item + [pad_int]*(max_length-len(item)) for item in batch_data]

char_file = open('char.voc.conll2009', 'r')
char_dict = {}
idx = 0
for char in char_file.readlines():
    char_dict[char.strip()] = idx
    idx += 1

print(char_dict['c'])


def get_batch(input_data, batch_size, word2idx, fr_word2idx, lemma2idx, pos2idx, pretrain2idx, fr_pretrain2idx,
              deprel2idx, argument2idx, idx2word, shuffle=False, withParrallel=False, lang="En"):


    role_number = len(argument2idx)

    if shuffle:
        random.shuffle(input_data)
    if withParrallel:
        fr_input_data = []
        fr_input_preidx = []
        fr_loss_mask = []
        fr_file = open("flat_train_fr_cleaned_2", 'r')
        idx = 0
        for line in fr_file.readlines():
            part = line.strip().split()
            if idx%2 == 0:
                fr_input_preidx.append(int(part[0]))
                if int(part[0]) == -1:
                    fr_loss_mask.append(0)
                else:
                    fr_loss_mask.append(1)
                #print(part[0])
            else:
                fr_input_data.append(part)
                #print(part)
            idx += 1
    else:
        fr_input_preidx = None

    for batch_i in range(int(math.ceil(len(input_data)/batch_size))):
        
        start_i = batch_i * batch_size
        end_i = start_i + batch_size
        if end_i > len(input_data):
            end_i = len(input_data)

        data_batch = input_data[start_i:end_i]
        if withParrallel:
            fr_data_batch = fr_input_data[start_i:end_i]
            fr_preidx_batch = fr_input_preidx[start_i:end_i]
        else:
            fr_preidx_batch = None


        role_index_batch = np.zeros((batch_size, role_number), dtype=int)
        role_mask_batch = np.zeros((batch_size, role_number), dtype=int)

        sentence_id_batch = [sentence[0][0] for sentence in data_batch]
        predicate_id_batch = [sentence[0][1] for sentence in data_batch]
        setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
        id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]
        index_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]

        seq_len_batch = [len(sentence) for sentence in data_batch]

        flag_batch = [[int(item[5]) for item in sentence] for sentence in data_batch]
        pad_flag_batch = np.array(pad_batch(flag_batch, batch_size, 0),dtype=int)


        sentence_flags_batch = [[int(item[16])+1 for item in sentence] for sentence in data_batch]
        pad_sentence_flags_batch = np.array(pad_batch(sentence_flags_batch, batch_size, 0),dtype=int)



        predicates_idx_batch = []
        for sentence in data_batch:
            for id, item in enumerate(sentence):
                if int(item[5]) == 1:
                    predicates_idx_batch.append(id)
                    break

        text_batch = [[item[6] for item in sentence] for sentence in data_batch]
        if len(text_batch) < batch_size:
            text_batch += [[_PAD_]] * (batch_size - len(text_batch))

        if lang=='En':
            word_batch = [[word2idx.get(item[6],word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
            pad_word_batch = np.array(pad_batch(word_batch, batch_size, word2idx[_PAD_]))
        else:
            word_batch = [[fr_word2idx.get(item[6], fr_word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
            pad_word_batch = np.array(pad_batch(word_batch, batch_size, fr_word2idx[_PAD_]))


        if withParrallel and False:
            fr_word_batch = [[fr_word2idx.get(item, fr_word2idx[_UNK_]) for item in sentence] for sentence in fr_data_batch]
            fr_pad_word_batch = np.array(pad_batch(fr_word_batch, batch_size, fr_word2idx[_PAD_]))
            fr_loss_mask_batch = np.array(fr_loss_mask[start_i:end_i])
            fr_pad_flag_batch = np.zeros_like(fr_pad_word_batch)
            #print(fr_data_batch)
            #print(batch_size)
            for i in range(batch_size):
                #log(fr_preidx_batch[i])
                fr_pad_flag_batch[i][fr_preidx_batch[i]] = 1
                #fr_pad_flag_batch[i][1] = 1
                #print(fr_pad_flag_batch[i])
        else:
            fr_pad_word_batch = None
            fr_pad_flag_batch = None
            fr_loss_mask_batch = None

        _, sen_max_len = pad_word_batch.shape
        flat_word_batch = pad_word_batch.ravel()
        #char_batch = [[char_dict[c] if char_dict.has_key(c) else 0 for c in idx2word[word]] for word in flat_word_batch]
        #pad_char_batch = np.array(pad_batch(char_batch, batch_size*sen_max_len, 0)).reshape(batch_size, sen_max_len, -1)

        lemma_batch = [[lemma2idx.get(item[7],lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_lemma_batch = np.array(pad_batch(lemma_batch, batch_size, lemma2idx[_PAD_]))

        pos_batch = [[pos2idx.get(item[8],pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pos_batch = np.array(pad_batch(pos_batch, batch_size, pos2idx[_PAD_]))

        gold_pos_batch = [[pos2idx.get(item[13], pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_gold_pos_batch = np.array(pad_batch(gold_pos_batch, batch_size, pos2idx[_PAD_]))

        if lang=='En' and False:
            head_batch = [[int(item[9]) for item in sentence] for sentence in data_batch]
            pad_head_batch = np.array(pad_batch(head_batch, batch_size, -1))

            gold_head_batch = [[int(item[14]) for item in sentence] for sentence in data_batch]
            pad_gold_head_batch = np.array(pad_batch(gold_head_batch, batch_size, -1))

            rhead_batch = [[int(item[10]) for item in sentence] for sentence in data_batch]
            pad_rhead_batch = np.array(pad_batch(rhead_batch, batch_size, -1))
        else:
            pad_head_batch = None

            pad_gold_head_batch = None

            pad_rhead_batch = None

        deprel_batch = [[deprel2idx.get(item[11],deprel2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_deprel_batch = np.array(pad_batch(deprel_batch, batch_size, deprel2idx[_PAD_]))

        gold_deprel_batch = [[deprel2idx.get(item[15], deprel2idx[_UNK_]) for item in sentence] for sentence in
                        data_batch]
        pad_gold_deprel_batch = np.array(pad_batch(gold_deprel_batch, batch_size, deprel2idx[_PAD_]))



        sep_pad_gold_deprel_batch = None #pad_gold_deprel_batch
        sep_pad_gold_link_batch = None #pad_gold_deprel_batch
        ### constructing specific gold deprel
        """
        for i, sentence in enumerate(data_batch):
            current_predicate_id = predicates_idx_batch[i]
            for j, item in enumerate(sentence):
                if pad_gold_head_batch[i][j]-1 == current_predicate_id:
                    sep_pad_gold_link_batch[i][j] = 3
                    continue
                if j == pad_gold_head_batch[i][current_predicate_id]-1:
                    sep_pad_gold_link_batch[i][j] = 2
                    continue
                sep_pad_gold_deprel_batch[i][j] = deprel2idx[_UNK_]
                sep_pad_gold_link_batch[i][j] = 1
        """

        argument_batch = [[argument2idx.get(item[12],argument2idx["_"]) for item in sentence] for sentence in data_batch]
        pad_argument_batch = np.array(pad_batch(argument_batch, batch_size, argument2idx[_PAD_]))
        flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

        for i in range(batch_size):
            for j in range(len(data_batch[i])):
                role = data_batch[i][j][12]
                role_idx = argument2idx.get(role, argument2idx["_"])
                if role_idx == 1:
                    continue
                #if withParrallel and fr_loss_mask_batch[i]== 0:
                #    continue

                role_index_batch[i][role_idx] = j
                role_mask_batch[i][role_idx] = 1



        if lang=='En':
            pretrain_word_batch = [[pretrain2idx.get(item[6],pretrain2idx[_UNK_]) for item in sentence] for sentence in data_batch]
            pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))
        else:
            pretrain_word_batch = [[fr_pretrain2idx.get(item[6], fr_pretrain2idx[_UNK_]) for item in sentence] for sentence in
                                   data_batch]
            pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))

        if withParrallel:
            fr_pretrain_word_batch = [[fr_pretrain2idx.get(item, fr_pretrain2idx[_UNK_]) for item in sentence] for sentence in
                                   fr_data_batch]
            fr_pad_pretrain_word_batch = np.array(pad_batch(fr_pretrain_word_batch, batch_size, fr_pretrain2idx[_PAD_]))
        else:
            fr_pad_pretrain_word_batch = None

        # flag indicies
        pad_flag_indices = [0 for _ in range(batch_size)]
        for idx in range(batch_size):
            for jdx in range(pad_flag_batch.shape[1]):
                if int(pad_flag_batch[idx, jdx]) == 1:
                    pad_flag_indices[idx] = jdx


        batch = {
            "sentence_id":sentence_id_batch,
            "predicate_id":predicate_id_batch,
            "fr_predicates_idx":fr_preidx_batch,
            "predicates_idx":predicates_idx_batch,
            "word_id":id_batch,
            "index":index_batch,
            "flag":pad_flag_batch,
            "fr_flag": fr_pad_flag_batch,
            "fr_loss_mask":fr_loss_mask_batch,
            "word":pad_word_batch,
            "fr_word": fr_pad_word_batch,
            "lemma":pad_lemma_batch,
            "pos":pad_pos_batch,
            "pretrain":pad_pretrain_word_batch,
            "fr_pretrain": fr_pad_pretrain_word_batch,
            "head":pad_head_batch,
            "rhead":pad_rhead_batch,
            "deprel":pad_deprel_batch,
            "argument":pad_argument_batch,
            "flat_argument":flat_argument_batch,
            "batch_size":pad_argument_batch.shape[0],
            "pad_seq_len":pad_argument_batch.shape[1],
            "text":text_batch,
            "sentence_len":setence_len_batch,
            "seq_len":seq_len_batch,
            "origin":data_batch,
            'flag_indices':pad_flag_indices,
            'gold_pos':pad_gold_pos_batch,
            'gold_head':pad_gold_head_batch,
            'gold_deprel':pad_gold_deprel_batch,
            'predicates_flag':pad_sentence_flags_batch,
            'sep_dep_rel': sep_pad_gold_deprel_batch,
            'sep_dep_link': sep_pad_gold_link_batch,
            'role_index': role_index_batch,
            'role_mask': role_mask_batch,
        }

        yield batch
            

