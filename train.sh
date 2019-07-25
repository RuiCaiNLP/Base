#!/usr/bin/env bash
python run.py --preprocess --train --train_data data/conll09_train.dataset --valid_data data/conll09_dev.dataset   --seed 100 --tmp_path temp --model_path model --result_path result --pretrain_embedding data/glove.6B.100d.txt --pretrain_emb_size 100  --epochs 20 --dropout 0.1 --lr 0.001 --batch_size 64 --word_emb_size 100 --pos_emb_size 32 --lemma_emb_size 100 --use_deprel --deprel_emb_size 64 --bilstm_hidden_size 512 --bilstm_num_layers 4 --valid_step 200 --use_highway --highway_num_layers 10 --use_flag_emb --flag_emb_size 16 --clip 5 > log