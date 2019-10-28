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

## word-level
def Input4Gan_0(hidden_input, predicates_1D):
    return hidden_input


## sentence level
def Input4Gan_1(output, predicates_1D):
    output = F.softmax(output, dim=2)
    seq_len = output.shape[1]
    shuffled_timestep = np.arange(0, seq_len)
    np.random.shuffle(shuffled_timestep)
    #Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
    shuffled_output = output.index_select(dim=1, index=get_torch_variable_from_np(shuffled_timestep))
    return shuffled_output


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
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.use_biaffine = model_params['use_biaffine']

        self.word_vocab_size = model_params['word_vocab_size']
        self.fr_word_vocab_size = model_params['fr_word_vocab_size']
        self.pretrain_vocab_size = model_params['pretrain_vocab_size']
        self.fr_pretrain_vocab_size = model_params['fr_pretrain_vocab_size']
        self.word_emb_size = model_params['word_emb_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']
        self.pretrain_emb_weight = model_params['pretrain_emb_weight']
        self.fr_pretrain_emb_weight = model_params['fr_pretrain_emb_weight']

        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']

        self.pretrained_embedding = nn.Embedding(self.pretrain_vocab_size, self.pretrain_emb_size)
        self.pretrained_embedding.weight.data.copy_(torch.from_numpy(self.pretrain_emb_weight))
        self.fr_pretrained_embedding = nn.Embedding(self.fr_pretrain_vocab_size, self.pretrain_emb_size)
        self.fr_pretrained_embedding.weight.data.copy_(torch.from_numpy(self.fr_pretrain_emb_weight))

        if self.use_flag_embedding:
            self.flag_embedding = nn.Embedding(2, self.flag_emb_size)
            self.flag_embedding.weight.data.uniform_(-1.0, 1.0)

        self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_emb_size)
        self.word_embedding.weight.data.uniform_(-1.0, 1.0)

        self.fr_word_embedding = nn.Embedding(self.fr_word_vocab_size, self.word_emb_size)
        self.fr_word_embedding.weight.data.uniform_(-1.0, 1.0)


        self.word_dropout = nn.Dropout(p=0.5)
        self.out_dropout = nn.Dropout(p=0.3)

        input_emb_size = 0
        if self.use_flag_embedding:
            input_emb_size += self.flag_emb_size
        else:
            input_emb_size += 1


        input_emb_size += self.pretrain_emb_size  # + self.pos_emb_size# + self.word_emb_size


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
                                    bidirectional=True,
                                    bias=True, batch_first=True)



        if self.use_biaffine:
            self.mlp_size = 300
            self.rel_W = nn.Parameter(
                torch.from_numpy(np.zeros((self.mlp_size + 1, self.target_vocab_size * (self.mlp_size + 1))).astype("float32")).to(
                    device))
            self.mlp_arg = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())
            self.mlp_pred = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())

    def forward(self, batch_input, unlabeled_batch_input, lang, unlabeled):

        pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        flag_emb = self.flag_embedding(flag_batch)

        if unlabeled:
            fr_pretrain_batch = get_torch_variable_from_np(unlabeled_batch_input['pretrain'])
            fr_flag_batch = get_torch_variable_from_np(unlabeled_batch_input['flag'])
            fr_pretrain_emb = self.fr_pretrained_embedding(fr_pretrain_batch).detach()
            fr_flag_emb = self.flag_embedding(fr_flag_batch).detach()
            predicates_1D = unlabeled_batch_input['predicates_idx']
        else:
            predicates_1D = batch_input['predicates_idx']

        if lang == "En":
            pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()
        else:
            pretrain_emb = self.fr_pretrained_embedding(pretrain_batch).detach()

        if not unlabeled:
            input_emb = torch.cat([flag_emb, pretrain_emb], 2)
        else:
            input_emb = torch.cat([fr_flag_emb, fr_pretrain_emb], 2)

        input_emb = self.word_dropout(input_emb)
        seq_len = input_emb.shape[1]
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size,seq_len, -1)
        hidden_input = self.out_dropout(hidden_input)
        #predicate_hidden = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        #predicates_hidden = predicate_hidden.unsqueeze(1).expand(self.batch_size, seq_len, 2*self.bilstm_hidden_size)
        arg_hidden = self.mlp_arg(hidden_input)
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.mlp_pred(pred_recur)
        output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                          num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        output = output.view(self.batch_size * seq_len, -1)

        """
        cat_output = en_output.view(self.batch_size, seq_len, -1)
        pred_recur = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len, 2*self.bilstm_hidden_size)
        all_cat = torch.cat((hidden_input, pred_recur, cat_output), 2)
        shuffled_timestep = np.arange(0, seq_len)
        np.random.shuffle(shuffled_timestep)
        all_cat = all_cat.index_select(dim=1, index=get_torch_variable_from_np(shuffled_timestep))
        all_cat = self.out_dropout(all_cat)
        """
        #enc = torch.mean(hidden_input, dim=1)
        #enc = Input4Gan_0(hidden_input, predicates_1D)
        enc = Input4Gan_1(output.view(self.batch_size, seq_len, -1), predicates_1D)
        return output, enc.view(self.batch_size, seq_len, -1)



class Discriminator(nn.Module):
    def __init__(self, model_params):
        super(Discriminator, self).__init__()
        self.batch_size = model_params['batch_size']
        self.bilstm_num_layers = 1
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.target_vocab_size = model_params['target_vocab_size']

        if USE_CUDA:
            self.bilstm_hidden_state = (
                Variable(torch.randn(2 * 1, self.batch_size, 10),
                         requires_grad=True).cuda(),
                Variable(torch.randn(2 * 1, self.batch_size, 10),
                         requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state = (
                Variable(torch.randn(2 * 1, self.batch_size, 10),
                         requires_grad=True),
                Variable(torch.randn(2 * 1, self.batch_size, 10),
                         requires_grad=True))

        self.bilstm_layer = nn.LSTM(input_size=self.target_vocab_size,
                                    hidden_size=10, num_layers=1,
                                    bidirectional=True,
                                    bias=True, batch_first=True)

        #self.MLP = nn.Sequential(
        #    nn.Linear(4*self.bilstm_hidden_size+self.target_vocab_size, 2*128),
        #    nn.ReLU()
        #)
        self.scorer = nn.Sequential(
            nn.Linear(self.bilstm_hidden_size, 1),
            nn.Sigmoid(),
        )

        self.emb_dim = 2*10
        self.dis_hid_dim = 30
        self.dis_layers = 1
        self.dis_input_dropout = 0.2
        self.dis_dropout = 0.2
        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        self.real = np.random.uniform(0.7, 1.0)  # 1
        self.fake = np.random.uniform(0.0, 0.3)  # 0


    def forward(self, x, sentence_level=False):
        #bilstm_output, (H_n, C_n) = self.bilstm_layer(hidden_states, self.bilstm_hidden_state)
        #score = self.scorer(H_n.view(-1))
        seq_len = x.shape[1]
        if sentence_level:
            bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(x, self.bilstm_hidden_state)
            bilstm_output = bilstm_output.contiguous()
            x = bilstm_output.view(self.batch_size*seq_len, -1)
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

class Adversarial_TModel(nn.Module):
    def __init__(self, model_params):
        super(Adversarial_TModel, self).__init__()
        self.batch_size = model_params['batch_size']
        self.EN_Labeler = EN_Labeler(model_params)
        self.Discriminator = Discriminator(model_params)
        self.dis_smooth = 0.1

        self.real = np.random.uniform(0.7, 1.0)  # 1
        self.fake = np.random.uniform(0.0, 0.3)  # 0


    def forward(self, batch_input, elmo, unlabeled_batch_input=None, unlabeled=False, withParallel=False, lang='En', isPretrain=False, TrainGenerator=False):

        output_en, enc_real = self.EN_Labeler(batch_input, unlabeled_batch_input, lang, unlabeled=False)
        if not unlabeled:
            return output_en
        _, enc_fake = self.EN_Labeler(batch_input, unlabeled_batch_input, lang, unlabeled=True)

        if not TrainGenerator:
            x_D_real = enc_real.detach()
            x_D_fake = enc_fake.detach()
            preds = self.Discriminator(Variable(x_D_real.data), sentence_level=True)
            real_labels = torch.empty(*preds.size()).fill_(self.real).type_as(preds)
            D_loss_real = F.binary_cross_entropy(preds, real_labels)
            preds = self.Discriminator(Variable(x_D_fake.data), sentence_level=True)
            fake_labels = torch.empty(*preds.size()).fill_(self.fake).type_as(preds)
            D_loss_fake = F.binary_cross_entropy(preds, fake_labels)
            D_loss = 0.5*(D_loss_real + D_loss_fake)
            return D_loss
        else:
            x_G = enc_fake
            preds = self.Discriminator(x_G, sentence_level=True)
            fake_labels = torch.empty(*preds.size()).fill_(self.real).type_as(preds)
            G_loss = F.binary_cross_entropy(preds, fake_labels)
            return output_en, G_loss



