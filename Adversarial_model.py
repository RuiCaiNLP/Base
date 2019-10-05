from __future__ import print_function
import numpy as np
import torch
from torch import nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F

from utils import USE_CUDA
from utils import get_torch_variable_from_np, get_data
from utils import bilinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log(*args, **kwargs):
    print(*args,file=sys.stderr, **kwargs)


class EN_Labeler(nn.Module):
    def __init__(self, model_params):
        super(EN_Labeler, self).__init__()
        self.dropout = model_params['dropout']
        self.dropout_word = model_params['dropout_word']
        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']

        input_emb_size = 0
        if self.use_flag_embedding:
            input_emb_size += self.flag_emb_size
        else:
            input_emb_size += 1

        if self.use_deprel:
            input_emb_size += self.pretrain_emb_size  # + self.pos_emb_size# + self.word_emb_size
        else:
            input_emb_size += self.pretrain_emb_size  # + self.pos_emb_size #+ self.word_emb_size

        if USE_CUDA:
            self.bilstm_hidden_state = (
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state = (
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))

        self.bilstm_layer = nn.LSTM(input_size=input_emb_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=self.bilstm_num_layers,
                                    dropout=self.dropout, bidirectional=True,
                                    bias=True, batch_first=True)



        if self.use_biaffine:
            self.mlp_size = 300
            self.rel_W = nn.Parameter(
                torch.from_numpy(np.zeros((self.mlp_size + 1, self.target_vocab_size * (self.mlp_size + 1))).astype("float32")).to(
                    device))
            self.mlp_arg = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())
            self.mlp_pred = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())

    def forward(self, batch_input_emb, predicates_1D):
        input_emb = batch_input_emb
        seq_len = batch_input_emb.shape[1]
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size, seq_len, -1)

        arg_hidden = self.mlp_dropout(self.mlp_arg(hidden_input))
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.pred_dropout(self.mlp_pred(pred_recur))
        output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                          num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        en_output = output.view(self.batch_size * seq_len, -1)


        return en_output


class FR_Labeler(nn.Module):
    def __init__(self, model_params):
        super(FR_Labeler, self).__init__()
        self.dropout = model_params['dropout']
        self.dropout_word = model_params['dropout_word']
        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']

        input_emb_size = 0
        if self.use_flag_embedding:
            input_emb_size += self.flag_emb_size
        else:
            input_emb_size += 1

        if self.use_deprel:
            input_emb_size += self.pretrain_emb_size  # + self.pos_emb_size# + self.word_emb_size
        else:
            input_emb_size += self.pretrain_emb_size  # + self.pos_emb_size #+ self.word_emb_size

        if USE_CUDA:
            self.bilstm_hidden_state = (
                Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                         requires_grad=True).cuda(),
                Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                         requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state = (
                Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                         requires_grad=True),
                Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                         requires_grad=True))

        self.bilstm_layer = nn.LSTM(input_size=input_emb_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=self.bilstm_num_layers,
                                    dropout=self.dropout, bidirectional=True,
                                    bias=True, batch_first=True)

        if self.use_biaffine:
            self.mlp_size = 300
            self.rel_W = nn.Parameter(
                torch.from_numpy(
                    np.zeros((self.mlp_size + 1, self.target_vocab_size * (self.mlp_size + 1))).astype("float32")).to(
                    device))
            self.mlp_arg = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())
            self.mlp_pred = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())

    def forward(self, batch_input_emb, predicates_1D):
        input_emb = batch_input_emb
        seq_len = batch_input_emb.shape[1]
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size, seq_len, -1)

        arg_hidden = self.mlp_dropout(self.mlp_arg(hidden_input))
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.pred_dropout(self.mlp_pred(pred_recur))
        output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                          num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        fr_output = output.view(self.batch_size * seq_len, -1)

        return fr_output

class Adversarial_Model(nn.Module):
    def __init__(self, model_params):
        super(Adversarial_Model, self).__init__()

