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
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.use_biaffine = model_params['use_biaffine']

        self.out_dropout = nn.Dropout(p=0.5)

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

    def forward(self, batch_input_emb, predicates_1D):
        input_emb = batch_input_emb
        seq_len = batch_input_emb.shape[1]
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size, seq_len, -1)

        arg_hidden = self.mlp_arg(hidden_input)
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.mlp_pred(pred_recur)
        output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                          num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        en_output = output.view(self.batch_size * seq_len, -1)

        """
        cat_output = en_output.view(self.batch_size, seq_len, -1)
        pred_recur = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len, 2*self.bilstm_hidden_size)
        all_cat = torch.cat((hidden_input, pred_recur, cat_output), 2)
        shuffled_timestep = np.arange(0, seq_len)
        np.random.shuffle(shuffled_timestep)
        all_cat = all_cat.index_select(dim=1, index=get_torch_variable_from_np(shuffled_timestep))
        all_cat = self.out_dropout(all_cat)
        """
        enc_real = torch.mean(hidden_input, dim=1)
        return en_output, enc_real


class FR_Labeler(nn.Module):
    def __init__(self, model_params):
        super(FR_Labeler, self).__init__()
        self.batch_size = model_params['batch_size']
        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.use_biaffine = model_params['use_biaffine']

        self.out_dropout = nn.Dropout(p=0.5)

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

        arg_hidden = self.mlp_arg(hidden_input)
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden =self.mlp_pred(pred_recur)
        output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                          num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        fr_output = output.view(self.batch_size * seq_len, -1)

        """
        cat_output = fr_output.view(self.batch_size, seq_len, -1)
        pred_recur = pred_recur.unsqueeze(dim=1).expand(self.batch_size, seq_len, 2 * self.bilstm_hidden_size)
        all_cat = torch.cat((hidden_input, pred_recur, cat_output), 2)
        shuffled_timestep = np.arange(0, seq_len)
        np.random.shuffle(shuffled_timestep)
        all_cat = all_cat.index_select( dim=1, index=get_torch_variable_from_np(shuffled_timestep))
        all_cat = self.out_dropout(all_cat)
        return fr_output, all_cat
        """
        enc_fake = torch.mean(hidden_input, dim=1)
        return fr_output, enc_fake

class Discriminator(nn.Module):
    def __init__(self, model_params):
        super(Discriminator, self).__init__()
        self.batch_size = model_params['batch_size']
        self.bilstm_num_layers = 1
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.target_vocab_size = model_params['target_vocab_size']

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

        self.bilstm_layer = nn.LSTM(input_size=4*self.bilstm_hidden_size+self.target_vocab_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=self.bilstm_num_layers,
                                    bidirectional=True,
                                    bias=True, batch_first=True)

        #self.MLP = nn.Sequential(
        #    nn.Linear(4*self.bilstm_hidden_size+self.target_vocab_size, 2*128),
        #    nn.ReLU()
        #)
        self.scorer = nn.Sequential(
            nn.Linear(2*self.bilstm_hidden_size, 1),
            nn.Sigmoid(),
        )

        self.emb_dim = 300
        self.dis_hid_dim = 200
        self.dis_layers = 2
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

    def forward(self, x):
        #bilstm_output, (H_n, C_n) = self.bilstm_layer(hidden_states, self.bilstm_hidden_state)
        #score = self.scorer(H_n.view(-1))
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

class Adversarial_TModel(nn.Module):
    def __init__(self, model_params):
        super(Adversarial_TModel, self).__init__()
        self.batch_size = model_params['batch_size']
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

        self.FR_Labeler = FR_Labeler(model_params)
        self.EN_Labeler = EN_Labeler(model_params)
        self.Discriminator = Discriminator(model_params)
        self.dis_smooth = 0.1


    def forward(self, batch_input, elmo, withParallel=True, lang='En', isPretrain=False, TrainGenerator=False):
        if lang=='En':
            pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        else:
            pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        flag_batch = get_torch_variable_from_np(batch_input['flag'])

        if withParallel:
            fr_pretrain_batch = get_torch_variable_from_np(batch_input['fr_pretrain'])
            fr_flag_batch = get_torch_variable_from_np(batch_input['fr_flag'])

        if self.use_flag_embedding:
            flag_emb = self.flag_embedding(flag_batch)
        else:
            flag_emb = flag_batch.view(flag_batch.shape[0], flag_batch.shape[1], 1).float()

        if lang == "En":
            pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()
        else:
            pretrain_emb = self.fr_pretrained_embedding(pretrain_batch).detach()

        if withParallel:
            fr_pretrain_emb = self.fr_pretrained_embedding(fr_pretrain_batch).detach()
            fr_flag_emb = self.flag_embedding(fr_flag_batch).detach()



        input_emb = torch.cat([flag_emb, pretrain_emb], 2)
        predicates_1D = batch_input['predicates_idx']
        if withParallel:
            fr_input_emb = torch.cat([fr_flag_emb, fr_pretrain_emb], 2)

        output_en, enc_real = self.EN_Labeler(input_emb, predicates_1D)
        output_fr, _ = self.FR_Labeler(input_emb, predicates_1D)

        if not withParallel:
            if isPretrain:
                return output_en
            else:
                return output_fr

        predicates_1D = batch_input['fr_predicates_idx']
        _, enc_fake = self.FR_Labeler(fr_input_emb, predicates_1D)

        x_D = torch.cat([real_states.detach(), fake_states.detach()], 0)
        x_G = torch.cat([real_states.detach(), fake_states], 0)
        y = torch.FloatTensor(2 * self.batch_size).zero_().to(device)
        y[:self.batch_size] = 1 - self.dis_smooth
        y[self.batch_size:] = self.dis_smooth


        if not TrainGenerator:
            #prob_real_decision = self.Discriminator(real_states.detach())
            #prob_fake_decision = self.Discriminator(fake_states.detach())
            #D_loss= - torch.mean(torch.log(prob_real_decision) + torch.log(1. - prob_fake_decision))

            preds = self.Discriminator(Variable(x_D.data))
            D_loss = F.binary_cross_entropy(preds, 1-y)
            return D_loss*get_torch_variable_from_np(batch_input['fr_loss_mask']).float()
        else:
            #prob_fake_decision_G = self.Discriminator(real_states.detach())
            #G_loss = -torch.mean(torch.log(prob_fake_decision_G))
            #log("G loss:", G_loss)
            preds = self.Discriminator(x_G)
            G_loss = F.binary_cross_entropy(preds, y)
            return G_loss*get_torch_variable_from_np(batch_input['fr_loss_mask']).float()



