import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from highway import HighwayMLP
from attention import Attention
from attention import BiAFAttention

from utils import USE_CUDA
from utils import get_torch_variable_from_np, get_data




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
            input_emb_size += self.pretrain_emb_size + self.word_emb_size + self.lemma_emb_size + self.pos_emb_size + self.deprel_emb_size  #
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
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state0 = (
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




        self.use_highway = model_params['use_highway']
        self.highway_layers = model_params['highway_layers']
        if self.use_highway:
            self.highway_layers = nn.ModuleList([HighwayMLP(self.bilstm_hidden_size * 2, activation_function=F.relu)
                                                 for _ in range(self.highway_layers)])

            self.output_layer = nn.Linear(self.bilstm_hidden_size * 2, self.target_vocab_size)
        else:
            self.output_layer = nn.Linear(self.bilstm_hidden_size * 2, self.target_vocab_size)

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

        word_emb = self.word_embedding(word_batch)
        lemma_emb = self.lemma_embedding(lemma_batch)
        pos_emb = self.pos_embedding(pos_batch)
        pretrain_emb = self.pretrained_embedding(pretrain_batch)

        if self.use_deprel:
            deprel_emb = self.deprel_embedding(deprel_batch)

        # predicate_emb = self.word_embedding(predicate_batch)
        # predicate_pretrain_emb = self.pretrained_embedding(predicate_pretrain_batch)

        if self.use_deprel:
            input_emb = torch.cat([flag_emb, word_emb, pretrain_emb, lemma_emb, pos_emb, deprel_emb], 2)  #
        else:
            input_emb = torch.cat([flag_emb, word_emb, pretrain_emb, lemma_emb, pos_emb], 2)  #



        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state0)

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

        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)

        if self.use_highway:
            for current_layer in self.highway_layers:
                hidden_input = current_layer(hidden_input)

            output = self.output_layer(hidden_input)
        else:
            output = self.output_layer(hidden_input)
        return output
