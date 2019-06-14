import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logit, target, length):
    logit_flat = logit.view(-1, logit.size(-1))
    target_flat = target.view(-1)
    losses_flat = F.cross_entropy(logit_flat, target_flat, reduction='none')
    losses = losses_flat.view(*target.size())
    mask = _sequence_mask(length, target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, dropout, use_bn):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Invalid type for input_dims!'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            l_i = i + 1
            layers['fc{}'.format(l_i)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(l_i)] = nn.ReLU()
            layers['drop{}'.format(l_i)] = nn.Dropout(dropout)
            if use_bn:
                layers['bn{}'.format(l_i)] = nn.BatchNorm1d(n_hidden)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model.forward(input)

class MLP_Discriminator(nn.Module):
    def __init__(self, embed_dim, hparams, use_cuda):
        super(MLP_Discriminator, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_state = hparams["hidden_state"]
        self.hidden_layers = hparams["hidden_layers"]
        self.hidden_dropout = hparams["hidden_dropout"]
        self.input_dropout = hparams["input_dropout"]
        self.use_bn = hparams["use_bn"]
        self.bidirectional = hparams["bidirectional"]
        self.use_cuda = use_cuda

        self.mlp = MLP(embed_dim * 5, [self.hidden_state] * self.hidden_layers,
                       1, self.hidden_dropout, self.use_bn)
        self.dropout = nn.Dropout(self.input_dropout)
        if self.bidirectional:
            self.backward_mlp = MLP(embed_dim * 5, [self.hidden_state] * self.hidden_layers,
                                    1, self.hidden_dropout, self.use_bn)
            self.backward_dropout = nn.Dropout(self.input_dropout)

    def forward(self, s1, s2):
        inputs = torch.cat([s1, s2, s1 - s2, s1 * s2, torch.abs(s1 - s2)], -1)
        scores = self.mlp(self.dropout(inputs))
        if self.bidirectional:
            backward_inputs = torch.cat(
                [s2, s1, s2 - s1, s1 * s2, torch.abs(s1 - s2)], -1)
            backward_scores = self.backward_mlp(
                self.backward_dropout(backward_inputs))
            scores = (scores + backward_scores) / 2
        return scores

class RNN_LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hparams, use_cuda):
        super(RNN_LM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hparams["hidden_size"]
        self.num_layers = hparams["num_layers"]
        self.cell_type = hparams["cell_type"]
        self.tie_embed = hparams["tie_embed"]
        self.rnn_dropout = hparams["rnn_dropout"]
        self.hidden_dropout = hparams["hidden_dropout"]
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(vocab_size, embed_size)
        rnn_class = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM,
        }[self.cell_type]
        self.rnn = rnn_class(self.embed_size, self.hidden_size, self.num_layers,
                             dropout=self.rnn_dropout)
        self.dropout = nn.Dropout(self.hidden_dropout)

        if self.tie_embed:
            self.linear_out = nn.Linear(embed_size, vocab_size)
            if embed_size != self.hidden_size:
                in_size = self.hidden_size
                self.linear_proj = nn.Linear(
                    in_size, embed_size, bias=None)
            self.linear_out.weight = self.embedding.weight
        else:
            self.linear_out = nn.Linear(self.hidden_size, vocab_size)
            self.linear_proj = lambda x: x

    def set_embed(self, emb):
        with torch.no_grad():
            self.embedding.weight.fill_(0.)
            self.embedding.weight += emb

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.num_layers,
                                  batch_size, self.hidden_size))
        if self.cell_type == 'lstm':
            c0 = Variable(torch.zeros(self.num_layers,
                                      batch_size, self.hidden_size))
            return (h0.cuda(), c0.cuda()) if self.use_cuda else (h0, c0)
        else:
            return h0.cuda() if self.use_cuda else c0

    def encode(self, input, hidden):
        embedded = self.dropout(self.embedding(input))
        output, _ = self.rnn(embedded, hidden)
        return output

    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        max_len, batch_size, _ = output.size()
        output = output.view(max_len * batch_size, -1)
        output = self.dropout(output)

        output = self.linear_proj(output)
        output = self.dropout(output)
        output = self.linear_out(output)

        output = output.view(max_len, batch_size, -1)
        return output, hidden
