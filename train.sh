#!/usr/bin/env bash
python run.py  --train --train_data data/En_train.dataset --valid_data data/Fr_dev.dataset  \
 --seed 100 --tmp_path temp --model_path model --result_path result --pretrain_embedding data/en.vec.txt\
  --pretrain_emb_size 300  --epochs 30 --dropout 0.1 --lr 0.001 --batch_size 30 \
  --word_emb_size 300 --pos_emb_size 32 --lemma_emb_size 100 --use_deprel --deprel_emb_size 64 \
  --bilstm_hidden_size 300 --bilstm_num_layers 3 \
  --valid_step 500  --use_flag_emb --flag_emb_size 16 --clip 5 \
  --dropout_word 0.3 --dropout_mlp 0.3 --use_biaffine #--preprocess
#--use_highway --highway_num_layers 10  --preprocess