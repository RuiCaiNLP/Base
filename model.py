from __future__ import print_function
import numpy as np
import torch
from torch import nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
from highway import HighwayMLP
from attention import Attention
from attention import BiAFAttention
from syntactic_gcn import SyntacticGCN

from utils import USE_CUDA
from utils import get_torch_variable_from_np, get_data
from utils import bilinear
from layer import CharCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log(*args, **kwargs):
    print(*args,file=sys.stderr, **kwargs)

def _roll(arr, direction, sparse=False):
  if sparse:
    return torch.cat((arr[:, direction:], arr[:, :direction]), dim=1)
  return torch.cat((arr[:, direction:, :], arr[:, :direction, :]),  dim=1)


def cat(l, dimension=-1):
    valid_l = l
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)

def mask_loss(self, Semi_loss, lengths):
    for i in range(Semi_loss.size()[0]):
        for j in range(Semi_loss.size()[1]):
            if j >= lengths[i]:
                Semi_loss[i][j] = 0 * Semi_loss[i][j]
    return Semi_loss



class End2EndModel(nn.Module):
    def __init__(self, model_params):
        super(End2EndModel, self).__init__()
        self.charCNN = CharCNN(num_of_conv=3, in_channels=1, out_channels=50, kernel_size=[2, 3, 4],
                                     in_features=50, out_features=100)
        self.dropout = model_params['dropout']
        self.dropout_word = model_params['dropout_word']
        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.word_vocab_size = model_params['word_vocab_size']
        self.fr_word_vocab_size = model_params['fr_word_vocab_size']
        self.lemma_vocab_size = model_params['lemma_vocab_size']
        self.pos_vocab_size = model_params['pos_vocab_size']
        self.deprel_vocab_size = model_params['deprel_vocab_size']
        self.pretrain_vocab_size = model_params['pretrain_vocab_size']
        self.fr_pretrain_vocab_size = model_params['fr_pretrain_vocab_size']

        self.word_emb_size = model_params['word_emb_size']
        self.lemma_emb_size = model_params['lemma_emb_size']
        self.pos_emb_size = model_params['pos_emb_size']

        self.use_deprel = model_params['use_deprel']
        self.deprel_emb_size = model_params['deprel_emb_size']

        self.pretrain_emb_size = model_params['pretrain_emb_size']
        self.pretrain_emb_weight = model_params['pretrain_emb_weight']
        self.fr_pretrain_emb_weight = model_params['fr_pretrain_emb_weight']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']

        self.target_vocab_size = model_params['target_vocab_size']

        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']

        self.use_gcn = model_params['use_gcn']
        self.use_sa_lstm = model_params['use_sa_lstm']
        self.use_rcnn = model_params['use_rcnn']
        self.use_tree_lstm = model_params['use_tree_lstm']
        self.use_biaffine = model_params['use_biaffine']
        self.deprel2idx = model_params['deprel2idx']



        if self.use_flag_embedding:
            self.flag_embedding = nn.Embedding(2, self.flag_emb_size)
            self.flag_embedding.weight.data.uniform_(-1.0, 1.0)

        self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_emb_size)
        self.word_embedding.weight.data.uniform_(-1.0, 1.0)

        self.fr_word_embedding = nn.Embedding(self.fr_word_vocab_size, self.word_emb_size)
        self.fr_word_embedding.weight.data.uniform_(-1.0, 1.0)

        self.lemma_embedding = nn.Embedding(self.lemma_vocab_size, self.lemma_emb_size)
        self.lemma_embedding.weight.data.uniform_(-1.0, 1.0)

        self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_emb_size)
        self.pos_embedding.weight.data.uniform_(-1.0, 1.0)

        self.char_embeddings = nn.Embedding(106, 50)

        if self.use_deprel:
            self.deprel_embedding = nn.Embedding(self.deprel_vocab_size, self.deprel_emb_size)
            self.deprel_embedding.weight.data.uniform_(-1.0, 1.0)

        self.pretrained_embedding = nn.Embedding(self.pretrain_vocab_size, self.pretrain_emb_size)
        self.pretrained_embedding.weight.data.copy_(torch.from_numpy(self.pretrain_emb_weight))
        self.fr_pretrained_embedding = nn.Embedding(self.fr_pretrain_vocab_size, self.pretrain_emb_size)
        self.fr_pretrained_embedding.weight.data.copy_(torch.from_numpy(self.fr_pretrain_emb_weight))

        input_emb_size = 0
        if self.use_flag_embedding:
            input_emb_size += self.flag_emb_size
        else:
            input_emb_size += 1

        if self.use_deprel:
            input_emb_size += self.pretrain_emb_size #+ self.pos_emb_size# + self.word_emb_size
        else:
            input_emb_size += self.pretrain_emb_size #+ self.pos_emb_size #+ self.word_emb_size

        self.use_elmo = model_params['use_elmo']
        self.elmo_emb_size = model_params['elmo_embedding_size']
        if self.use_elmo:
            input_emb_size += self.elmo_emb_size
            self.elmo_mlp = nn.Sequential(nn.Linear(1024, self.elmo_emb_size), nn.ReLU())
            self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
            self.elmo_gamma = nn.Parameter(torch.ones(1))



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


        if USE_CUDA:
            self.fr_bilstm_hidden_state = (
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.fr_bilstm_hidden_state = (
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))




        self.bilstm_layer = nn.LSTM(input_size=input_emb_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=self.bilstm_num_layers,
                                    dropout=self.dropout, bidirectional=True,
                                    bias=True, batch_first=True)


        # self.bilstm_mlp = nn.Sequential(nn.Linear(self.bilstm_hidden_size*2, self.bilstm_hidden_size), nn.ReLU())
        self.use_self_attn = model_params['use_self_attn']
        if self.use_self_attn:
            self.self_attn_head = model_params['self_attn_head']
            self.attn_linear_first = nn.Linear(self.bilstm_hidden_size * 2, self.bilstm_hidden_size)
            self.attn_linear_first.bias.data.fill_(0)

            self.attn_linear_second = nn.Linear(self.bilstm_hidden_size, self.self_attn_head)
            self.attn_linear_second.bias.data.fill_(0)

            self.attn_linear_final = nn.Sequential(
                nn.Linear(self.bilstm_hidden_size * 2 * 2, self.bilstm_hidden_size * 2), nn.Tanh())

            # self.biaf_attn = BiAFAttention(self.bilstm_hidden_size*2, self.bilstm_hidden_size*2, self.self_attn_head)

            # self.attn_linear_final = nn.Sequential(nn.Linear(self.bilstm_hidden_size*4,self.bilstm_hidden_size*2), nn.ReLU())

        if self.use_gcn:
            # self.W_in = nn.Parameter(torch.randn(2*self.bilstm_hidden_size, 2*self.bilstm_hidden_size))
            # self.W_out = nn.Parameter(torch.randn(2*self.bilstm_hidden_size, 2*self.bilstm_hidden_size))
            # self.W_self = nn.Parameter(torch.randn(2*self.bilstm_hidden_size, 2*self.bilstm_hidden_size))
            # self.gcn_bias = nn.Parameter(torch.randn(2*self.bilstm_hidden_size))
            self.syntactic_gcn = SyntacticGCN(self.bilstm_hidden_size * 2, self.bilstm_hidden_size,
                                              self.deprel_vocab_size, batch_first=True)

            self.gcn_mlp = nn.Sequential(nn.Linear(self.bilstm_hidden_size * 3, self.bilstm_hidden_size * 2), nn.ReLU())


        self.use_highway = model_params['use_highway']
        self.highway_layers = model_params['highway_layers']
        if self.use_highway:
            self.highway_layers = nn.ModuleList([HighwayMLP(self.bilstm_hidden_size * 2, activation_function=F.relu)
                                                 for _ in range(self.highway_layers)])

            self.output_layer = nn.Linear(self.bilstm_hidden_size * 2, self.target_vocab_size)
        else:
            self.output_layer = nn.Linear(self.bilstm_hidden_size * 2, self.target_vocab_size)

        if self.use_biaffine:
            self.mlp_size = 300
            self.rel_W = nn.Parameter(
                torch.from_numpy(np.zeros((self.mlp_size + 1, self.target_vocab_size * (self.mlp_size + 1))).astype("float32")).to(
                    device))
            self.mlp_arg = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())
            self.mlp_pred = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())

        self.mlp_dropout = nn.Dropout(p=self.dropout_mlp)
        self.pred_dropout = nn.Dropout(p=self.dropout_mlp)
        self.word_dropout = nn.Dropout(p=self.dropout_word)

        self.pos_classifier = nn.Sequential(nn.Linear(2 * 300, 300), nn.ReLU(), nn.Linear(300, self.pos_vocab_size))
        self.PI_classifier = nn.Sequential(nn.Linear(2 * 300, 300), nn.ReLU(),  nn.Linear(300, 3))
        self.mlp_arg_deprel = nn.Sequential(nn.Linear(2 * 300, 300), nn.ReLU())
        self.mlp_pred_deprel = nn.Sequential(nn.Linear(2 * 300, 300), nn.ReLU())

        self.mlp_arg_link = nn.Sequential(nn.Linear(2 * 300, self.mlp_size), nn.ReLU())
        self.mlp_pred_link = nn.Sequential(nn.Linear(2 * 300, self.mlp_size), nn.ReLU())

        self.deprel_W = nn.Parameter(
            torch.from_numpy(
                np.zeros((self.mlp_size + 1, self.deprel_vocab_size * (self.mlp_size + 1))).astype("float32")).to(
                device))

        self.link_W = nn.Parameter(
            torch.from_numpy(
                np.zeros((self.mlp_size + 1, 4 * (self.mlp_size + 1))).astype("float32")).to(
                device))

        self.elmo_mlp = nn.Sequential(nn.Linear(2 * 300, 200), nn.ReLU())
        self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma = nn.Parameter(torch.ones(1))

        self.POS2hidden = nn.Linear(self.pos_vocab_size, self.pos_emb_size)
        self.deprel2hidden = nn.Linear(self.deprel_vocab_size, self.deprel_emb_size)


    def Semi_SRL_Loss(self, hidden_forward, hidden_backward, TagProbs_use, sentence, lengths, target_idx_in):
        TagProbs_use_softmax = F.softmax(TagProbs_use, dim=2).detach()
        sample_nums = lengths.sum()
        unlabeled_loss_function = nn.KLDivLoss(reduce=False)

        hidden_future = _roll(hidden_forward, -1)
        tag_space = self.SRL_MLP_Future(self.hidden_future_unlabeled(hidden_future))
        tag_space = tag_space.view(self.batch_size, len(sentence[0]), -1)
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_Future_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        hidden_past = _roll(hidden_backward, 1)
        tag_space = self.SRL_MLP_Past(self.hidden_past_unlabeled(hidden_past))
        tag_space = tag_space.view(self.batch_size, len(sentence[0]), -1)
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_Past_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        DEP_Future_loss = torch.sum(DEP_Future_loss, dim=2)
        DEP_Past_loss = torch.sum(DEP_Past_loss, dim=2)

        wordBeforePre_mask = np.ones((self.batch_size, len(sentence[0])), dtype='float32')
        for i in range(self.batch_size):
            for j in range(len(sentence[0])):
                if j >= target_idx_in[i]:
                    wordBeforePre_mask[i][j] = 0.0
        wordBeforePre_mask = torch.from_numpy(wordBeforePre_mask).to(device)

        wordAfterPre_mask = np.ones((self.batch_size, len(sentence[0])), dtype='float32')
        for i in range(self.batch_size):
            for j in range(len(sentence[0])):
                if j <= target_idx_in[i]:
                    wordAfterPre_mask[i][j] = 0.0
        wordAfterPre_mask = torch.from_numpy(wordAfterPre_mask).to(device)

        DEP_Semi_loss = wordBeforePre_mask * DEP_Past_loss + wordAfterPre_mask * DEP_Future_loss

        loss_mask = np.ones(DEP_Semi_loss.size(), dtype='float32')
        for i in range(self.batch_size):
            for j in range(len(sentence[0])):
                if j >= lengths[i]:
                    loss_mask[i][j] = 0.0
        loss_mask = torch.from_numpy(loss_mask).to(device)

        DEP_Semi_loss = DEP_Semi_loss * loss_mask

        DEP_Semi_loss = torch.sum(DEP_Semi_loss)
        return DEP_Semi_loss / sample_nums

    def find_predicate_embeds(self, hidden_states, target_idx_in):
        Label_composer = hidden_states
        predicate_embeds = Label_composer[np.arange(0, Label_composer.size()[0]), target_idx_in]
        # T * B * H
        added_embeds = torch.zeros(Label_composer.size()[1], Label_composer.size()[0],
                                   Label_composer.size()[2]).to(device)
        concat_embeds = (added_embeds + predicate_embeds).transpose(0, 1)
        return concat_embeds

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n

        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors

        """

        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, batch_input, elmo, withParallel=True, lang='En'):

        if lang=='En':
            word_batch = get_torch_variable_from_np(batch_input['word'])
            pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        else:
            word_batch = get_torch_variable_from_np(batch_input['word'])
            pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])

        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        pos_batch = get_torch_variable_from_np(batch_input['pos'])
        pos_emb = self.pos_embedding(pos_batch)
        #log(pretrain_batch)
        #log(flag_batch)

        role_index = get_torch_variable_from_np(batch_input['role_index'])
        role_mask = get_torch_variable_from_np(batch_input['role_mask'])
        role_mask_expand = role_mask.unsqueeze(2).expand(self.batch_size, self.target_vocab_size, self.word_emb_size)

        role2word_batch = pretrain_batch.gather(dim=1, index=role_index)
        #log(role2word_batch)
        role2word_emb = self.pretrained_embedding(role2word_batch).detach()
        role2word_emb = role2word_emb * role_mask_expand.float()
        #log(role2word_emb[0][1])
        #log(role2word_emb[0][2])
        #log(role_mask)
        #log(role_mask_expand)
        #log("#################")
        #log(role_index)
        #log(role_mask)
        #log(role2word_batch)

        if withParallel:
            fr_word_batch = get_torch_variable_from_np(batch_input['fr_word'])
            fr_pretrain_batch = get_torch_variable_from_np(batch_input['fr_pretrain'])
            fr_flag_batch = get_torch_variable_from_np(batch_input['fr_flag'])

        #log(fr_pretrain_batch[0])
        #log(fr_flag_batch)
        if self.use_flag_embedding:
            flag_emb = self.flag_embedding(flag_batch)
        else:
            flag_emb = flag_batch.view(flag_batch.shape[0], flag_batch.shape[1], 1).float()

        seq_len = flag_batch.shape[1]
        if lang == "En":
            word_emb = self.word_embedding(word_batch)
            pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()

        else:
            word_emb = self.fr_word_embedding(word_batch)
            pretrain_emb = self.fr_pretrained_embedding(pretrain_batch).detach()

        if withParallel:
            #fr_word_emb = self.fr_word_embedding(fr_word_batch)
            fr_pretrain_emb = self.fr_pretrained_embedding(fr_pretrain_batch).detach()
            fr_flag_emb = self.flag_embedding(fr_flag_batch)
            fr_seq_len = fr_flag_batch.shape[1]
            role_mask_expand_timestep = role_mask.unsqueeze(2).expand(self.batch_size, self.target_vocab_size, fr_seq_len)
            role_mask_expand_forFR = \
                role_mask_expand.unsqueeze(1).expand(self.batch_size, fr_seq_len, self.target_vocab_size,
                                                     self.pretrain_emb_size)




        # predicate_emb = self.word_embedding(predicate_batch)
        # predicate_pretrain_emb = self.pretrained_embedding(predicate_pretrain_batch)

        #######semantic role labelerxxxxxxxxxx

        if self.use_deprel:
            input_emb = torch.cat([flag_emb, pretrain_emb], 2)  #
        else:
            input_emb = torch.cat([flag_emb,  pretrain_emb], 2)  #

        if withParallel:
            fr_input_emb = torch.cat([fr_flag_emb, fr_pretrain_emb], 2)



        input_emb = self.word_dropout(input_emb)
        #input_emb = torch.cat((input_emb, get_torch_variable_from_np(np.zeros((self.batch_size, seq_len, self.target_vocab_size))).float()), 2)
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size, seq_len, -1)
        #output = self.output_layer(hidden_input)

        arg_hidden = self.mlp_dropout(self.mlp_arg(hidden_input))
        predicates_1D = batch_input['predicates_idx']
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.pred_dropout(self.mlp_pred(pred_recur))
        output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                              num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        en_output = output.view(self.batch_size*seq_len, -1)


        if withParallel:
            role2word_emb_1 = role2word_emb.view(self.batch_size, self.target_vocab_size, -1)
            role2word_emb_2 = role2word_emb_1.unsqueeze(dim=1)
            role2word_emb_expand = role2word_emb_2.expand(self.batch_size, fr_seq_len, self.target_vocab_size,
                                                        self.pretrain_emb_size)

            #log(role2word_emb_expand[0, 1, 4])
            #log(role2word_emb_expand[0, 2, 4])

            fr_pretrain_emb = fr_pretrain_emb.view(self.batch_size, fr_seq_len, -1)
            fr_pretrain_emb = fr_pretrain_emb.unsqueeze(dim=2)
            fr_pretrain_emb_expand = fr_pretrain_emb.expand(self.batch_size, fr_seq_len, self.target_vocab_size,
                                                            self.pretrain_emb_size)
            fr_pretrain_emb_expand = fr_pretrain_emb_expand  # * role_mask_expand_forFR.float()
            # B T R W
            emb_distance = fr_pretrain_emb_expand - role2word_emb_expand
            emb_distance = emb_distance * emb_distance
            # B T R
            emb_distance = emb_distance.sum(dim=3)
            emb_distance = torch.sqrt(emb_distance)
            emb_distance = emb_distance.transpose(1, 2)
            # emb_distance_min = emb_distance_min.expand(self.batch_size, fr_seq_len, self.target_vocab_size)
            # emb_distance = F.softmax(emb_distance, dim=1)*role_mask_expand_timestep.float()
            # B R
            emb_distance_min, emb_distance_argmin = torch.min(emb_distance, dim=2, keepdim=True)
            emb_distance_max, emb_distance_argmax = torch.max(emb_distance, dim=2, keepdim=True)
            emb_distance_max_expand = emb_distance_max.expand(self.batch_size, self.target_vocab_size, fr_seq_len)
            emb_distance_min_expand = emb_distance_min.expand(self.batch_size, self.target_vocab_size, fr_seq_len)
            emb_distance_nomalized = (emb_distance - emb_distance_min_expand) / (
            emb_distance_max_expand - emb_distance_min_expand)
                                     #* role_mask_expand_timestep.float()
            # emb_distance_argmin_expand = emb_distance_argmin.expand(self.batch_size, fr_seq_len, self.target_vocab_size)
            # log("######################")

            emb_distance_nomalized = emb_distance_nomalized.detach()


            top2 = emb_distance_nomalized.topk(2, dim=2, largest=False, sorted=True)[0]

            top2_gap = top2[:,:, 1] - top2[:,:,0]
            #log(top2_gap)
            #log(role_mask)
            #log(emb_distance_nomalized[0,:,2])
            #log(emb_distance_argmax)

            #emb_distance_argmin = emb_distance_argmin*role_mask
            #log(emb_distance_argmin)
            #log(emb_distance_nomalized[0][2])
            """
            weight_4_loss = emb_distance_nomalized
            #log(emb_distance_argmin[0,2])
            for i in range(self.batch_size):
                for j in range(self.target_vocab_size):
                    weight_4_loss[i,j].index_fill_(0, emb_distance_argmin[i][j], 1)
            """
            #log(weight_4_loss[0][2])


            fr_input_emb = self.word_dropout(fr_input_emb).detach()
            #fr_input_emb = torch.cat((fr_input_emb, emb_distance_nomalized), 2)
            fr_bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(fr_input_emb, self.fr_bilstm_hidden_state)
            fr_bilstm_output = fr_bilstm_output.contiguous()
            hidden_input = fr_bilstm_output.view(fr_bilstm_output.shape[0] * fr_bilstm_output.shape[1], -1)
            hidden_input = hidden_input.view(self.batch_size, fr_seq_len, -1)
            arg_hidden = self.mlp_dropout(self.mlp_arg(hidden_input))
            predicates_1D = batch_input['fr_predicates_idx']
            #log(predicates_1D)
            #log(predicates_1D)
            pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
            pred_hidden = self.pred_dropout(self.mlp_pred(pred_recur))
            output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, fr_seq_len, 1, self.batch_size,
                              num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
            output = output.view(self.batch_size,  fr_seq_len, -1)
            output = output.transpose(1, 2)
            #B R T
            #output_p = F.softmax(output, dim=2)
            #output = output * role_mask_expand_timestep.float()
            #log(role_mask_expand_timestep)
            """
            output_exp = torch.exp(output)
            output_exp_weighted = output_exp #* weight_4_loss
            output_expsum = output_exp_weighted.sum(dim=2, keepdim=True).expand(self.batch_size, self.target_vocab_size,
                                                                                fr_seq_len)
            output = output_exp/output_expsum
            """
            #log(output_expsum)
            #log(output_p[0, 2])
            # B R 1
            #output_pmax, output_pmax_arg = torch.max(output_p, dim=2, keepdim=True)

            #log(output[0])
            #log(emb_distance[0,:, 2])
            #log(emb_distance_nomalized[0,:, 2])
            #log(role_mask[0])
            #log(role_mask[0])
            #log(output[0][0])
            #log("++++++++++++++++++++")
            #log(emb_distance)
            #log(emb_distance.gather(1, emb_distance_argmin))
            #output_pargminD = output_p.gather(2, emb_distance_argmin)
            #log(output_argminD)
            #weighted_distance = (output/output_argminD) * emb_distance_nomalized
            #bias = torch.FloatTensor(output_pmax.size()).fill_(1).cuda()
            #rank_loss = (output_pmax/output_pargminD)# * emb_distance_nomalized.gather(2, output_pmax_arg)
            #rank_loss = rank_loss.view(self.batch_size, self.target_vocab_size)
            #rank_loss = rank_loss * role_mask.float()
            #log("++++++++++++++++++++++")
            #log(output_max-output_argminD)
            #log(emb_distance_nomalized.gather(1, output_max_arg))
            # B R
            #weighted_distance = weighted_distance.squeeze() * role_mask.float()

            """
            output = F.softmax(output, dim=1)
            output = output.transpose(1, 2)
            fr_role2word_emb = torch.bmm(output, fr_pretrain_emb)
            criterion = nn.MSELoss(reduce=False)
            l2_loss = criterion(fr_role2word_emb.view(self.batch_size*self.target_vocab_size, -1),
                                  role2word_emb.view(self.batch_size*self.target_vocab_size, -1))
            """
            criterion = nn.CrossEntropyLoss(ignore_index=0, reduce=False)


            output = output.view(self.batch_size*self.target_vocab_size, -1)
            emb_distance_argmin = emb_distance_argmin*role_mask.unsqueeze(2)
            emb_distance_argmin = emb_distance_argmin.view(-1)
            #log(emb_distance_argmin[0][2])
            l2_loss = criterion(output, emb_distance_argmin)
            l2_loss = l2_loss.view(self.batch_size, self.target_vocab_size)
            top2_gap = top2_gap.view(self.batch_size, self.target_vocab_size)
            l2_loss = l2_loss * top2_gap
            l2_loss = l2_loss.sum()
            #l2_loss = F.nll_loss(torch.log(output), emb_distance_argmin, ignore_index=0)

            #log(emb_distance_argmin)
            #log(l2_loss)
            #log("+++++++++++++++++++++")
            #log(l2_loss)
            #log(batch_input['fr_loss_mask'])
            l2_loss = l2_loss*get_torch_variable_from_np(batch_input['fr_loss_mask']).float()
            l2_loss = l2_loss.sum()#/float_role_mask.sum()
            #log("+")
            #log(l2_loss)
            return en_output, l2_loss
        return en_output

