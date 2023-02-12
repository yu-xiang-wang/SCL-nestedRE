# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : tiny_models.py
# @Software: PyCharm


import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
import numpy as np
import copy
import math


class TreeLSTM(nn.Module):
    def __init__(self, in_features, out_features, relation_types):
        super(TreeLSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relation_types = relation_types
        self.relation_embedding = nn.Embedding(relation_types, in_features)
        self.W_i = nn.Linear(in_features, out_features)
        self.U_i = nn.Linear(in_features*2, out_features)
        self.W_f = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(2)])
        self.U_f = nn.ModuleList([nn.Linear(in_features*2, out_features) for _ in range(2)])
        self.W_o = nn.Linear(in_features, out_features)
        self.U_o = nn.Linear(in_features*2, out_features)
        self.W_c = nn.Linear(in_features, out_features)
        self.U_c = nn.Linear(in_features*2, out_features)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, relation_ids, layer_hidden, layer_context, source):
        """
        :param relation_ids: b x s
        :param layer_hidden: b x N x H
        :param layer_context: b x N x H
        :param source: b x s x k ; k is max num of child nodes
        :return: candidate hidden
        """
        b, s, k = source.shape
        source = source.reshape(b, -1)
        relation_emb = self.relation_embedding(relation_ids)
        source_hidden = t.cat([layer_hidden[i][source[i]].unsqueeze(0) for i in range(b)]).to(relation_ids.device)  # b x s*k x H
        source_hidden = source_hidden.reshape(b, s, k, -1)  # b x s x k x H
        source_context = t.cat([layer_context[i][source[i]] for i in range(b)]).to(relation_ids.device)  # b x s*k x H
        source_context = source_context.reshape(b, s, k, -1).float()  # b x s x k x H
        last_hidden = source_hidden.reshape(b, s, -1)  # b x s x k*H
        i_out = self.sigmoid(self.W_i(relation_emb) + self.U_i(last_hidden))  # b x s x H
        f_out = t.cat(
            [self.sigmoid(W(relation_emb) + U(last_hidden)).unsqueeze(-2) for W, U in zip(self.W_f, self.U_f)],
            dim=-2)  # b x s x k x H
        o_out = self.sigmoid(self.W_o(relation_emb) + self.U_o(last_hidden))  # b x s x H
        c_out = self.tanh(self.W_c(relation_emb) + self.U_c(last_hidden))  # b x s x H
        forget = f_out * source_context  # b x s x k x H
        c = i_out * c_out + forget.sum(dim=-2)  # b x s x H
        h = self.tanh(c) * o_out  # b x s x H
        return h, c


class Classifier(nn.Module):
    def __init__(self, in_features, dropout=0.1):
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, 2*in_features)
        self.fc2 = nn.Linear(2*in_features, 1)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # b x s x h
        # x1 = self.fc1(self.activation(x))  # b x s x 2h
        x1 = self.fc1(x)
        x1 = self.dropout(x1)  # b x s x 2h
        x2 = self.fc2(x1)  # b x s x 2
        # x2 = F.softmax(x2, dim=-1)[..., 0]  # b x s
        return F.sigmoid(x2.squeeze(-1))


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size=6000, embedding_size=256, hidden_size=256):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x, x_lens):  # b x l
        embed = self.embedding(x)  # b x l x e
        packed_embedded = pack_padded_sequence(embed, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, cell) = self.lstm(packed_embedded)  # b x l x 2h
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, mask):
        """
        attention for entity，
        :param x: b x s x h, entity word hidden
        :param mask: b x s, 1 for word, 0 for pad
        :return: b x h
        """
        attention = self.w(x)  # b x s x 1
        attention = attention.squeeze(-1)  # b x s
        attention.masked_fill_(mask, -1e9)  # fill -1e9 into pad place
        attention = t.softmax(attention, dim=-1).unsqueeze(-2)  # b x 1 x s
        context = t.matmul(attention, x).squeeze(-2)  # b x h
        return context


def do_pad(x, max_l, pad_elem):
    """
    padding x into target shape，
    :param x: l x h
    :param max_l: target l
    :param pad_elem: padding element
    :return: padded x with shape ml x h
    """
    l, h = x.shape
    if l < max_l:
        pad_tensor = t.full([max_l - l, h], pad_elem, device=x.device)
        x = t.cat([x, pad_tensor])
    return x


class MyPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512, is_trainable=True):
        super(MyPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not is_trainable:
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def clones(module, N):
    """"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Attention
    :param query: b x n_head x seq_len x d_k
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.shape[-1]
    scores = t.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # b x h x s x s
    # print('attention shape = ', scores.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return t.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, hidden_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        """Multi-Head Attention"""
        assert hidden_dim % n_head == 0
        self.d_k = hidden_dim // n_head
        self.n_head = n_head
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        multi head attention
        :param query: b x s x h
        :param key:
        :param value:
        :param mask: b x s
        :return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        b = query.size(0)
        q, k, v = [l(x).view(b, -1, self.n_head, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, [query, key, value])]

        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.n_head*self.d_k)
        return self.linears[-1](x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        """"""
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x+self.dropout(sublayer(x)))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        """FNN module"""
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, in_dim, dropout, n_head, d_ff):
        super(MyTransformerEncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, in_dim, dropout)
        self.feed_forward = PositionWiseFeedForward(in_dim, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(in_dim, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda e: self.attn(e, e, e, mask))
        return self.sublayer[1](x, self.feed_forward)


class MyTransformerEncoder(nn.Module):
    def __init__(self, in_dim, dropout, n_head, d_ff, N):
        super(MyTransformerEncoder, self).__init__()
        layer = MyTransformerEncoderLayer(in_dim, dropout, n_head, d_ff)
        self.layers = clones(layer, N)
        # self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


