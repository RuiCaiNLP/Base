import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from highway import HighwayMLP
from attention import Attention
from attention import BiAFAttention
from syntactic_gcn import SyntacticGCN

from utils import USE_CUDA
from utils import get_torch_variable_from_np, get_data
from utils import bilinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# <!-- pytorch and allennlp upgrade
# from allennlp.modules.elmo import Elmo
# from allennlp.data.dataset import Dataset
# from allennlp.data import Token, Vocabulary, Instance
# from allennlp.data.fields import TextField
# from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer




# indexer = ELMoTokenCharactersIndexer()
# def batch_to_ids(batch):
#     """
#     Given a batch (as list of tokenized sentences), return a batch
#     of padded character ids.
#     """
#     instances = []
#     for sentence in batch:
#         tokens = [Token(token) for token in sentence]
#         field = TextField(tokens, {'character_ids': indexer})
#         instance = Instance({"elmo": field})
#         instances.append(instance)

#     dataset = Dataset(instances)
#     vocab = Vocabulary()
#     # dataset.index_instances(vocab)
#     for instance in dataset.instances:
#         instance.index_fields(vocab)
#     return dataset.as_tensor_dict()['elmo']['character_ids']


# -->

class End2EndModel(nn.Module):
    def __init__(self, model_params):
        super(End2EndModel, self).__init__()
        self.dropout = model_params['dropout']
        self.dropout_word = model_params['dropout_word']
        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.word_vocab_size = model_params['word_vocab_size']
        self.lemma_vocab_size = model_params['lemma_vocab_size']
        self.pos_vocab_size = model_params['pos_vocab_size']
        self.deprel_vocab_size = model_params['deprel_vocab_size']
        self.pretrain_vocab_size = model_params['pretrain_vocab_size']

        self.word_emb_size = model_params['word_emb_size']
        self.lemma_emb_size = model_params['lemma_emb_size']
        self.pos_emb_size = model_params['pos_emb_size']

        self.use_deprel = model_params['use_deprel']
        self.deprel_emb_size = model_params['deprel_emb_size']

        self.pretrain_emb_size = model_params['pretrain_emb_size']
        self.pretrain_emb_weight = model_params['pretrain_emb_weight']

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

        self.lemma_embedding = nn.Embedding(self.lemma_vocab_size, self.lemma_emb_size)
        self.lemma_embedding.weight.data.uniform_(-1.0, 1.0)

        self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_emb_size)
        self.pos_embedding.weight.data.uniform_(-1.0, 1.0)

        if self.use_deprel:
            self.deprel_embedding = nn.Embedding(self.deprel_vocab_size, self.deprel_emb_size)
            self.deprel_embedding.weight.data.uniform_(-1.0, 1.0)

        self.pretrained_embedding = nn.Embedding(self.pretrain_vocab_size, self.pretrain_emb_size)
        self.pretrained_embedding.weight.data.copy_(torch.from_numpy(self.pretrain_emb_weight))

        input_emb_size = 0
        if self.use_flag_embedding:
            input_emb_size += self.flag_emb_size
        else:
            input_emb_size += 1

        if self.use_deprel:
            input_emb_size += self.pretrain_emb_size + self.word_emb_size + self.lemma_emb_size + self.pos_emb_size + self.deprel_emb_size + 4  #
        else:
            input_emb_size += self.pretrain_emb_size + self.word_emb_size + self.lemma_emb_size + self.pos_emb_size

        self.use_elmo = model_params['use_elmo']
        self.elmo_emb_size = model_params['elmo_embedding_size']
        if self.use_elmo:
            input_emb_size += self.elmo_emb_size
            self.elmo_mlp = nn.Sequential(nn.Linear(1024, self.elmo_emb_size), nn.ReLU())
            self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
            self.elmo_gamma = nn.Parameter(torch.ones(1))



        if USE_CUDA:
            self.bilstm_hidden_state0 = (
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state0 = (
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))

        if USE_CUDA:
            self.bilstm_hidden_state0_high = (
            Variable(torch.randn(2 * (self.bilstm_num_layers-1), self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * (self.bilstm_num_layers-1), self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state0_high = (
            Variable(torch.randn(2 * (self.bilstm_num_layers-1), self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * (self.bilstm_num_layers-1), self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))


        if USE_CUDA:
            self.SL_hidden_state0 = (
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.SL_hidden_state0 = (
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))

        if USE_CUDA:
            self.SL_hidden_state0_high = (
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.SL_hidden_state0_high = (
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * 1, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))


        self.bilstm_layer = nn.LSTM(input_size=input_emb_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=1,
                                    dropout=self.dropout, bidirectional=True,
                                    bias=True, batch_first=True)

        self.bilstm_layer_high = nn.LSTM(input_size=3*self.bilstm_hidden_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=self.bilstm_num_layers-1,
                                    dropout=self.dropout, bidirectional=True,
                                    bias=True, batch_first=True)

        self.sentence_learner = nn.LSTM(input_size=self.pretrain_emb_size + self.word_emb_size,
                                        hidden_size=self.bilstm_hidden_size, num_layers=1,
                                        dropout=self.dropout, bidirectional=True,
                                        bias=True, batch_first=True)

        self.sentence_learner_high = nn.LSTM(input_size=2*self.bilstm_hidden_size,
                                        hidden_size=self.bilstm_hidden_size, num_layers=1,
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

        self.pos_classifier = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, 300), nn.ReLU(), nn.Linear(300, self.pos_vocab_size))
        self.PI_classifier = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, 300), nn.ReLU(),  nn.Linear(300, 3))
        self.mlp_arg_deprel = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())
        self.mlp_pred_deprel = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())
        self.deprel_W = nn.Parameter(
            torch.from_numpy(
                np.zeros((self.mlp_size + 1, self.deprel_vocab_size * (self.mlp_size + 1))).astype("float32")).to(
                device))
        self.Link_classifier = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, 300), nn.ReLU(), nn.Linear(300, 4))

        self.elmo_mlp = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.bilstm_hidden_size), nn.ReLU())
        self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma = nn.Parameter(torch.ones(1))

        self.POS2hidden = nn.Linear(self.pos_vocab_size, self.pos_emb_size)
        self.deprel2hidden = nn.Linear(self.deprel_vocab_size, self.deprel_emb_size)

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

    def forward(self, batch_input, elmo):

        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        word_batch = get_torch_variable_from_np(batch_input['word'])
        lemma_batch = get_torch_variable_from_np(batch_input['lemma'])
        pos_batch = get_torch_variable_from_np(batch_input['pos'])
        deprel_batch = get_torch_variable_from_np(batch_input['deprel'])
        pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        origin_batch = batch_input['origin']
        origin_deprel_batch = batch_input['deprel']

        if self.use_flag_embedding:
            flag_emb = self.flag_embedding(flag_batch)
        else:
            flag_emb = flag_batch.view(flag_batch.shape[0], flag_batch.shape[1], 1).float()
        seq_len = flag_batch.shape[1]
        word_emb = self.word_embedding(word_batch)
        lemma_emb = self.lemma_embedding(lemma_batch)
        pos_emb = self.pos_embedding(pos_batch)
        pretrain_emb = self.pretrained_embedding(pretrain_batch)

        if self.use_deprel:
            deprel_emb = self.deprel_embedding(deprel_batch)

        # predicate_emb = self.word_embedding(predicate_batch)
        # predicate_pretrain_emb = self.pretrained_embedding(predicate_pretrain_batch)



        ##sentence learner#####################################
        SL_input_emb = self.word_dropout(torch.cat([word_emb, pretrain_emb], 2))
        h0, (_, SL_final_state) = self.sentence_learner(SL_input_emb, self.SL_hidden_state0)
        h1, (_, SL_final_state) = self.sentence_learner_high(h0, self.SL_hidden_state0_high)
        SL_output = h1
        POS_output = self.pos_classifier(SL_output).view(self.batch_size * seq_len, -1)
        PI_output = self.PI_classifier(SL_output).view(self.batch_size * seq_len, -1)
        Link_output = self.Link_classifier(SL_output).view(self.batch_size * seq_len, 4)
        ## deprel
        hidden_input = SL_output
        arg_hidden = self.mlp_dropout(self.mlp_arg_deprel(SL_output))
        predicates_1D = batch_input['predicates_idx']
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.pred_dropout(self.mlp_pred_deprel(pred_recur))
        deprel_output = bilinear(arg_hidden, self.deprel_W, pred_hidden, self.mlp_size, seq_len, 1,
                                 self.batch_size,
                                 num_outputs=self.deprel_vocab_size, bias_x=True, bias_y=True)
        deprel_output = deprel_output.view(self.batch_size * seq_len, -1)

        POS_probs = F.softmax(POS_output, dim=1).view(self.batch_size, seq_len, -1)
        deprel_probs = F.softmax(deprel_output, dim=1).view(self.batch_size, seq_len, -1)
        POS_compose = F.tanh(self.POS2hidden(POS_probs))
        deprel_compose = F.tanh(self.deprel2hidden(deprel_probs))
        link_compose = F.softmax(Link_output, dim=1).view(self.batch_size, seq_len, -1)

        #######semantic role labelerxxxxxxxxxx

        if self.use_deprel:
            input_emb = torch.cat([flag_emb, word_emb, pretrain_emb, lemma_emb, POS_compose, deprel_compose, link_compose], 2)  #
        else:
            input_emb = torch.cat([flag_emb, word_emb, pretrain_emb, lemma_emb, POS_compose], 2)  #

        input_emb = self.word_dropout(input_emb)

        w = F.softmax(self.elmo_w, dim=0)
        SRL_composer = self.elmo_gamma * (w[0] * h0 + w[1] * h1)
        SRL_composer = self.elmo_mlp(SRL_composer)
        bilstm_output_0, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state0)
        high_input = torch.cat((bilstm_output_0, SRL_composer), 2)
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer_high(high_input, self.bilstm_hidden_state0_high)



        # bilstm_final_state = bilstm_final_state.view(self.bilstm_num_layers, 2, self.batch_size, self.bilstm_hidden_size)

        # bilstm_final_state = bilstm_final_state[-1]

        # sentence latent representation
        # bilstm_final_state = torch.cat([bilstm_final_state[0], bilstm_final_state[1]], 1)

        # bilstm_output = self.bilstm_mlp(bilstm_output)

        if self.use_self_attn:
            x = F.tanh(self.attn_linear_first(bilstm_output))
            x = self.attn_linear_second(x)
            x = self.softmax(x, 1)
            attention = x.transpose(1, 2)
            sentence_embeddings = torch.matmul(attention, bilstm_output)
            sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.self_attn_head

            context = sentence_embeddings.repeat(bilstm_output.size(1), 1, 1).transpose(0, 1)

            bilstm_output = torch.cat([bilstm_output, context], 2)

            bilstm_output = self.attn_linear_final(bilstm_output)

            # energy = self.biaf_attn(bilstm_output, bilstm_output)

            # # energy = energy.transpose(1, 2)

            # flag_indices = batch_input['flag_indices']

            # attention = []
            # for idx in range(len(flag_indices)):
            #     attention.append(energy[idx,:,:,flag_indices[idx]].view(1, self.bilstm_hidden_size, -1))

            # attention = torch.cat(attention,dim=0)

            # # attention = attention.transpose(1, 2)

            # attention = self.softmax(attention, 2)

            # # attention = attention.transpose(1,2)

            # sentence_embeddings = attention@bilstm_output

            # sentence_embeddings = torch.sum(sentence_embeddings,1)/self.self_attn_head

            # context = sentence_embeddings.repeat(bilstm_output.size(1), 1, 1).transpose(0, 1)

            # bilstm_output = torch.cat([bilstm_output, context], 2)

            # bilstm_output = self.attn_linear_final(bilstm_output)
        else:
            bilstm_output = bilstm_output.contiguous()

        if self.use_gcn:
            # in_rep = torch.matmul(bilstm_output, self.W_in)
            # out_rep = torch.matmul(bilstm_output, self.W_out)
            # self_rep = torch.matmul(bilstm_output, self.W_self)

            # child_indicies = batch_input['children_indicies']

            # head_batch = batch_input['head']

            # context = []
            # for idx in range(head_batch.shape[0]):
            #     states = []
            #     for jdx in range(head_batch.shape[1]):
            #         head_ind = head_batch[idx, jdx]-1
            #         childs = child_indicies[idx][jdx]
            #         state = self_rep[idx, jdx]
            #         if head_ind != -1:
            #               state = state + in_rep[idx, head_ind]
            #         for child in childs:
            #             state = state + out_rep[idx, child]
            #         state = F.relu(state + self.gcn_bias)
            #         states.append(state.unsqueeze(0))
            #     context.append(torch.cat(states, dim=0))

            # context = torch.cat(context, dim=0)

            # bilstm_output = context

            seq_len = bilstm_output.shape[1]

            adj_arc_in = np.zeros((self.batch_size * seq_len, 2), dtype='int32')
            adj_lab_in = np.zeros((self.batch_size * seq_len), dtype='int32')

            adj_arc_out = np.zeros((self.batch_size * seq_len, 2), dtype='int32')
            adj_lab_out = np.zeros((self.batch_size * seq_len), dtype='int32')

            mask_in = np.zeros((self.batch_size * seq_len), dtype='float32')
            mask_out = np.zeros((self.batch_size * seq_len), dtype='float32')

            mask_loop = np.ones((self.batch_size * seq_len, 1), dtype='float32')

            for idx in range(len(origin_batch)):
                for jdx in range(len(origin_batch[idx])):

                    offset = jdx + idx * seq_len

                    head_ind = int(origin_batch[idx][jdx][10]) - 1

                    if head_ind == -1:
                        continue

                    dependent_ind = int(origin_batch[idx][jdx][4]) - 1

                    adj_arc_in[offset] = np.array([idx, dependent_ind])

                    adj_lab_in[offset] = np.array([origin_deprel_batch[idx, jdx]])

                    mask_in[offset] = 1

                    adj_arc_out[offset] = np.array([idx, head_ind])

                    adj_lab_out[offset] = np.array([origin_deprel_batch[idx, jdx]])

                    mask_out[offset] = 1

            if USE_CUDA:
                adj_arc_in = torch.LongTensor(np.transpose(adj_arc_in)).cuda()
                adj_arc_out = torch.LongTensor(np.transpose(adj_arc_out)).cuda()

                adj_lab_in = Variable(torch.LongTensor(adj_lab_in).cuda())
                adj_lab_out = Variable(torch.LongTensor(adj_lab_out).cuda())

                mask_in = Variable(torch.FloatTensor(mask_in.reshape((self.batch_size * seq_len, 1))).cuda())
                mask_out = Variable(torch.FloatTensor(mask_out.reshape((self.batch_size * seq_len, 1))).cuda())
                mask_loop = Variable(torch.FloatTensor(mask_loop).cuda())
            else:
                adj_arc_in = torch.LongTensor(np.transpose(adj_arc_in))
                adj_arc_out = torch.LongTensor(np.transpose(adj_arc_out))

                adj_lab_in = Variable(torch.LongTensor(adj_lab_in))
                adj_lab_out = Variable(torch.LongTensor(adj_lab_out))

                mask_in = Variable(torch.FloatTensor(mask_in.reshape((self.batch_size * seq_len, 1))))
                mask_out = Variable(torch.FloatTensor(mask_out.reshape((self.batch_size * seq_len, 1))))
                mask_loop = Variable(torch.FloatTensor(mask_loop))

            gcn_context = self.syntactic_gcn(bilstm_output,
                                             adj_arc_in, adj_arc_out,
                                             adj_lab_in, adj_lab_out,
                                             mask_in, mask_out,
                                             mask_loop)

            #gcn_context = self.softmax(gcn_context, axis=2)
            gcn_context = F.softmax(gcn_context, dim=2)

            bilstm_output = torch.cat([bilstm_output, gcn_context], dim=2)

            bilstm_output = self.gcn_mlp(bilstm_output)

        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)

        if self.use_highway:
            for current_layer in self.highway_layers:
                hidden_input = current_layer(hidden_input)

            output = self.output_layer(hidden_input)
        else:
            hidden_input = hidden_input.view(self.batch_size, seq_len, -1)
            #output = self.output_layer(hidden_input)

        if self.use_biaffine:
            arg_hidden = self.mlp_dropout(self.mlp_arg(hidden_input))
            predicates_1D = batch_input['predicates_idx']
            pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
            pred_hidden = self.pred_dropout(self.mlp_pred(pred_recur))
            output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                                  num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
            output = output.view(self.batch_size*seq_len, -1)


        return output, POS_output, PI_output, deprel_output, Link_output

