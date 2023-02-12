# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : models.py
# @Software: PyCharm


import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, AutoModel, RobertaModel
import numpy as np
import copy
from tiny_models import TreeLSTM, Attention, LSTMEncoder, do_pad, Classifier
from tiny_models import MyPositionalEncoding, MyTransformerEncoder
from data_utils import has_candidate, generate_re_type2id, generate_type2index, generate_candidate1
from data_utils import generate_candidate2, generate_candidate3
from utils import compute_similarity_examples


class IModel(nn.Module):
    def __init__(self, vocab_size=6000, embedding_size=256, lstm_size=256, dag_size=512,
                 relation_types=11):
        super(IModel, self).__init__()
        self.encoder = LSTMEncoder(vocab_size, embedding_size, lstm_size)
        self.DAG_block = TreeLSTM(in_features=lstm_size*2, out_features=dag_size,
                                  relation_types=relation_types)
        self.attention = Attention(lstm_size*2)
        self.attention2 = Attention(lstm_size*2)
        self.cf = Classifier(in_features=dag_size)

    def forward(self, x, x_lens, entity_index, entity_type, elements, relations, all_type, sim_label=None, c_w=0):
        """
        compute the label for all candidate
        :param x: b x l, token ids
        :param x_lens: b, token lens
        :param entity_index: b x ?s x 2, start and end index of all entity in sentences of a batch
        :param entity_type: b x ?s, entity type of entities in each sentence
        :param elements: b x ?n, element of all sentences in a batch, [(type: str, (op_idx, ...))]
        :param relations: r, contain type and ops types (type: str, (op_type))
        :param all_type: type of entity and relation
        :return:
        """
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1]-entity_index[i][j][0]+1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # hidden state
        # encode input words
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x h*2
        # contrastive loss
        loss_c = t.tensor(0)
        if c_w != 0:
            cls_hidden = self.attention2(
                encoder_output,
                t.tensor([[0]*x_lens[i] + [1]*(max(x_lens)-x_lens[i]) for i in range(b)], dtype=t.bool, device=device))
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1 + t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                            max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, encoder_output.shape[-1]], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h*2
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        loss, num = None, 0

        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        layer = 1
        while has_candidate(candidates):
            candidate_label = [[c in elements[i] for c in candidates[i]] for i in range(b)]  # b x ?sc, label
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k

            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1]*len(l)+[0]*(max_c_len-len(l)) for l in candidate_label]  # b x sc
            candidate_label = [l+[False]*(max_c_len-len(l)) for l in candidate_label]
            candidate_ids = [cs+[0]*(max_c_len-len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1]*2 for _ in range(max_c_len-len(cs))] for cs in candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)

            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index
            )  # b x sc x H
            predict = self.cf(candidate_hidden)  # b x s
            #       mask.sum().item())
            layer += 1

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False,)
                                            # weight=candidate_label.masked_fill(candidate_label.le(0), 0.1))
            t_loss = t.sum(t_loss * mask)
            loss = t_loss if not loss else (t_loss+loss)
            num += t.sum(mask)

            # update elem hidden and context by true candidates
            for i in range(b):
                for tp in all_type:
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]  # hidden
                        all_elem_context[i][idx] = candidate_context[i][j]  # context
                        type2index[i][candidates[i][j][0]][-1].append(idx)  # elem type2index
                pre_candidates[i].extend(candidates[i])
            # generate new candidates
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        return (loss/num if loss else t.tensor(0)) * (1-c_w) + loss_c * c_w

    def forward_cont(self, x, x_lens, entity_index, entity_type, elements, relations, all_type, sim_label):
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # hidden state
        # encode input words
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x h*2
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, encoder_output.shape[-1]], device=device,
                                                      dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h*2
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        loss, num = None, 0

        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]  # b x ?s
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        layer = 1
        while has_candidate(candidates):
            candidate_label = [[c in elements[i] for c in candidates[i]] for i in range(b)]  # b x ?sc, label
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k

            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_label]  # b x sc
            candidate_label = [l + [False] * (max_c_len - len(l)) for l in candidate_label]
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)

            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index
            )  # b x sc x H
            predict = self.cf(candidate_hidden)  # b x s
            #       mask.sum().item())
            layer += 1

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            # weight=candidate_label.masked_fill(candidate_label.le(0), 0.1))
            t_loss = t.sum(t_loss * mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += t.sum(mask)

            # update elem hidden and context by true candidates
            for i in range(b):
                for tp in all_type:
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]  # hidden
                        all_elem_context[i][idx] = candidate_context[i][j]  # context
                        type2index[i][candidates[i][j][0]][-1].append(idx)  # elem type2index
                pre_candidates[i].extend(candidates[i])
            # generate new candidates
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        return loss / num if loss else t.tensor(0)

    def predict_scratch(self, x, x_lens, entity_index, entity_type, relations, all_type, threshold=0.5):
        """
        predict result from provided entities layer by layer
        :param x:
        :param x_lens:
        :param entity_index:
        :param entity_type:
        :param relations:
        :param all_type:
        :param threshold: filter predict > threshold
        :return:
        """
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        # hidden state
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x 2h
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get element hidden by Attention, padding to elem_len+1
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0]+1, pad_elem=0) if h.shape[0]
                           else t.zeros([h.shape[0]+1, encoder_output.shape[-1]], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify & predict
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        num, layer = 0, 1
        result = [[] for _ in range(b)]

        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k
            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_ids]  # b x sc
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index)
            predict = self.cf(candidate_hidden)  # b x s
            assert mask.shape == predict.shape
            predict = predict*mask
            # print('layer %d | predict %d' % (layer, t.sum(predict.ge(threshold).long()).item()))

            for i in range(b):
                index2type_prob = {}
                for tp in all_type:
                    type2index[i][tp].append([])
                # 
                m_r = 1
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        if candidates[i][j] in index2type_prob:
                            index2type_prob[candidates[i][j][1]].append((candidates[i][j][0], j, predict[i][j].item()))
                        else:
                            index2type_prob[candidates[i][j][1]] = [(candidates[i][j][0], j, predict[i][j].item())]
                choose_candidates = sorted(list(index2type_prob.items()), key=lambda c: c[0])
                for cs in choose_candidates:
                    cs = list(sorted(cs[1], key=lambda c: c[2], reverse=True))[:m_r]
                    for c in cs:
                        idx = all_elem_hidden[i].shape[0] - 1
                        j = c[1]
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]
                        all_elem_context[i][idx] = candidate_context[i][j]
                        all_elem_hidden[i] = do_pad(all_elem_hidden[i], max_l=idx+2, pad_elem=0)
                        all_elem_context[i] = do_pad(all_elem_context[i], max_l=idx + 2, pad_elem=0)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                        result[i].append(candidates[i][j])
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            layer += 1
        return result

    def predict_guided(self, x, x_lens, entity_index, entity_type, elements, relations, all_type, threshold=0.5):
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # hidden state
        # encode input words
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x h*2
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, encoder_output.shape[-1]], device=device,
                                                      dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h*2
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify & predict
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        num, layer = 0, 1
        result = [[] for _ in range(b)]
        gold = [[] for _ in range(b)]
        all_candidate = [[] for _ in range(b)]
        all_prob = [[] for _ in range(b)]

        while layer <= 6:
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            for i_ in range(b):
                all_candidate[i_].append(candidates[i_])
            layer += 1
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k
            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_ids]  # b x sc
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index)
            predict = self.cf(candidate_hidden)  # b x s
            assert mask.shape == predict.shape
            predict = predict * mask
            # print('layer %d | predict %d' % (layer, t.sum(predict.ge(threshold).long()).item()))

            for i in range(b):
                # index2type_prob = {}
                result[i].append([])
                gold[i].append([])
                all_prob[i].append(predict[i].tolist())
                for tp in all_type:
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        result[i][-1].append(candidates[i][j])
                    if candidates[i][j] in elements[i]:
                        gold[i][-1].append(candidates[i][j])
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]  # hidden
                        all_elem_context[i][idx] = candidate_context[i][j]  # context
                        type2index[i][candidates[i][j][0]][-1].append(idx)  # elem type2index
        return result, gold, all_candidate, all_prob

    def teach_forward(self, trained_model, x, x_lens, entity_index, entity_type, relations, all_type, threshold=0.5):
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        # hidden state
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x 2h
        t_encoder_output, _ = trained_model.encoder(x, x_lens)

        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        t_entity_hidden = [t_encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]

        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get element hidden by Attention, padding to elem_len+1
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0] + 1, pad_elem=0) if h.shape[0]
                           else t.zeros([h.shape[0] + 1, encoder_output.shape[-1]], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        t_all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0] + 1, pad_elem=0) if h.shape[0]
                             else t.zeros([h.shape[0] + 1, t_encoder_output.shape[-1]], device=device, dtype=t.float)
                             for h, m in zip(t_entity_hidden, entity_mask)]
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]
        t_all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in t_all_elem_hidden]

        # generate relation candidate and classify & predict
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        num, layer = 0, 1
        loss = None

        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k
            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_ids]  # b x sc
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            # print(candidate_ids.shape)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index)
            t_candidate_hidden, t_candidate_context = trained_model.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index)

            predict = self.cf(candidate_hidden)  # b x s
            t_predict = trained_model.cf(t_candidate_hidden)
            assert mask.shape == predict.shape
            t_label = t.tensor(t_predict.clone().detach() > threshold, dtype=t.float, device=device)
            # t_label = t_predict.clone().detach()
            t_loss = F.binary_cross_entropy(predict, t.tensor(t_label, dtype=t.float, device=device), reduce=False)
            t_loss = t.sum(t_loss * mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += t.sum(mask)

            for i in range(b):
                for tp in all_type:
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if t_predict[i][j] > threshold:
                        idx = all_elem_hidden[i].shape[0] - 1
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]  # hidden
                        all_elem_context[i][idx] = candidate_context[i][j]  # context
                        all_elem_hidden[i] = do_pad(all_elem_hidden[i], max_l=idx + 2, pad_elem=0)
                        all_elem_context[i] = do_pad(all_elem_context[i], max_l=idx + 2, pad_elem=0)
                        t_all_elem_hidden[i][idx] = t_candidate_hidden[i][j]
                        t_all_elem_context[i][idx] = t_candidate_context[i][j]
                        t_all_elem_hidden[i] = do_pad(t_all_elem_hidden[i], max_l=idx + 2, pad_elem=0)
                        t_all_elem_context[i] = do_pad(t_all_elem_context[i], max_l=idx + 2, pad_elem=0)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
            layer += 1
            # print(layer)
        return loss


class ITransformer(nn.Module):
    def __init__(self, bert_path, max_seq_len, hidden_size, head, trans_layers, all_types, trans_type=0):
        super(ITransformer, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        self.DAG_trans = MyTransformerEncoder(hidden_size, 0.1, head, 4*hidden_size, trans_layers)
        # self.pos_encoder = PositionalEncoding(hidden_size, max_seq_len)
        self.pos_encoder = MyPositionalEncoding(hidden_size)
        self.type2id = {tp: i for i, tp in enumerate(all_types)}
        self.type_embedding = nn.Embedding(len(all_types)+2, hidden_size)  # all types, cls, sep
        self.type_trans = nn.Linear(hidden_size, hidden_size)
        self.hid_trans = nn.Linear(hidden_size, hidden_size)
        self.cls_id, self.sep_id = len(all_types), len(all_types)+1
        self.attention = Attention(hidden_size)
        self.cls = Classifier(in_features=hidden_size)
        self.trans_type = trans_type

    def forward(self, x, x_mask, entity_index, entity_type, elements, relations, sim_label=None, c_w=0):
        """
        train model iteratively layer by layer
        :param x: input words ids b x s
        :param x_mask: input mask b x s
        :param entity_index: b x ?s x 2, start and end index of all entity in sentences of a batch
        :param entity_type: b x ?s, entity types in sentences of a batch
        :param elements: b x ?n, element of all sentences in a batch, [(type: str, (op_idx, ...))]
        :param relations: r, contain type and ops types (type: str, (op_type))
        :return:
        """
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]

        # contrastive loss
        loss_c = t.tensor(0)
        if c_w != 0:
            cls_hidden = encoder_output[:, 0]  # b x h
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat)/(1+t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)

        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1+entity_index[i][j][1]))+[0]*(max_ew_len-entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len+1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h

        # generate candidates and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_label = [[c in elements[i] for c in candidates[i]] + [False]*(max_c_len-c_lens[i])
                               for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            # if candidate_label.sum() == 0:
            #     break
            candidate_mask = t.tensor([[1]*c_lens[i]+[0]*(max_c_len-c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]]+[0]*(max_c_len-c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [[self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0]*(max_c_len-c_lens[i])
                            for i in range(b)]
            op2_type_ids = [[self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0]*(max_c_len-c_lens[i])
                            for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0]*(max_c_len-c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0]*(max_c_len-c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id]*max_c_len for _ in range(b)]
            sep_index = [[self.sep_id]*max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                # op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                # op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])

            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc

            # print('train:', layer, '|', t.sum(predict.ge(0.5).long()).item(), '|', candidate_label.sum().item(), '|',
            #       candidate_mask.sum().item())

            # compute loss and update hidden state
            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False,)
            t_loss = t.sum(t_loss * candidate_mask)
            loss = t_loss if not loss else (t_loss+loss)
            num += candidate_mask.sum().item()

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        # type2index[i][candidates[i][j][0]].append(idx)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                # pre_candidates[i].extend(candidates[i])
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return (loss / num if loss else t.tensor(0)) * (1 - c_w) + loss_c * c_w

    def forward_cont(self, x, x_mask, entity_index, entity_type, elements, relations, sim_label, c_w=0.1):
        """train model with main loss and contrastive loss"""
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]

        # compute contrastive loss
        cls_hidden = encoder_output[:, 0]  # b x h
        sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
        sim_mat = t.triu(sim_mat, diagonal=1)
        # print(sim_mat)
        sim_mat = t.exp(sim_mat)/(1+t.exp(sim_mat))
        sim_label = t.tensor(sim_label, dtype=t.float, device=device)
        # print(sim_label, sim_mat)
        loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)

        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                            max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h

        # generate candidates and classify
        # pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # 
            candidate_label = [[c in elements[i] for c in candidates[i]] + [False] * (max_c_len - c_lens[i])
                               for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            # if candidate_label.sum() == 0:
            #     break
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc

            #       candidate_mask.sum().item())

            # compute loss and update hidden state
            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            t_loss = t.sum(t_loss * candidate_mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += candidate_mask.sum().item()

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        # type2index[i][candidates[i][j][0]].append(idx)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                # pre_candidates[i].extend(candidates[i])
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return (loss / num if loss else t.tensor(0)) * (1 - c_w) + loss_c * c_w

    def predict_scratch(self, x, x_mask, entity_index, entity_type, relations, threshold=0.5):
        """
        predict result from provided entities layer by layer
        :param x:
        :param x_mask:
        :param entity_index:
        :param entity_type:
        :param relations:
        :param threshold:
        :return:
        """
        b, device = x.shape[0], x.device
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        # hidden state
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get element hidden by Attention, padding to elem_len+1
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0] + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([h.shape[0]+1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        # generate relation candidate and classify & predict
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = copy.deepcopy(entity_type)
        num, layer = 0, 1
        result = [[] for _ in range(b)]
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]
        #
        # while has_candidate(candidates):
        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                # op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                # op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc
            assert candidate_mask.shape == predict.shape
            predict = predict * candidate_mask
            # print('layer %d | predict %d' % (layer, t.sum(predict.ge(threshold).long()).item()))

            for i in range(b):
                index2type_prob = {}
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                
                m_r = 1
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        if candidates[i][j][1] in index2type_prob:
                            index2type_prob[candidates[i][j][1]].append((candidates[i][j][0], j, predict[i][j].item()))
                        else:
                            index2type_prob[candidates[i][j][1]] = [(candidates[i][j][0], j, predict[i][j].item())]
                choose_candidates = sorted(list(index2type_prob.items()), key=lambda c: c[0])
                for cs in choose_candidates:
                    cs = list(sorted(cs[1], key=lambda c: c[2], reverse=True))[:m_r]
                    for c in cs:
                        idx = all_elem_hidden[i].shape[0] - 1
                        j = c[1]
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        all_elem_hidden[i] = do_pad(all_elem_hidden[i], max_l=idx+2, pad_elem=0)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                        index2type[i].append(candidates[i][j][0])
                        result[i].append(candidates[i][j])
                
            layer += 1
        return result

    def predict_guided(self, x, x_mask, entity_index, entity_type, elements, relations, threshold=0.5):
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                            max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h

        # generate candidates and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        num, layer = 0, 1
        result = [[] for _ in range(b)]
        gold = [[] for _ in range(b)]
        all_candidate = [[] for _ in range(b)]
        all_prob = [[] for _ in range(b)]

        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            for i_ in range(b):
                all_candidate[i_].append(candidates[i_])
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc
            assert candidate_mask.shape == predict.shape
            predict = predict * candidate_mask
            # print('layer %d | predict %d' % (layer, t.sum(predict.ge(threshold).long()).item()))

            for i in range(b):
                # index2type_prob = {}
                result[i].append([])
                gold[i].append([])
                all_prob[i].append(predict[i].tolist())
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        result[i][-1].append(candidates[i][j])
                    if candidates[i][j] in elements[i]:
                        gold[i][-1].append(candidates[i][j])
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]  # hidden
                        type2index[i][candidates[i][j][0]][-1].append(idx)
            layer += 1
        return result, gold, all_candidate, all_prob

    def forward_weight(self, x, x_mask, entity_index, entity_type, elements, relations, relation_weight):
        """
        :param x: input words ids b x s
        :param x_mask: input mask b x s
        :param entity_index: b x ?s x 2, start and end index of all entity in sentences of a batch
        :param entity_type: b x ?s, entity types in sentences of a batch
        :param elements: b x ?n, element of all sentences in a batch, [(type: str, (op_idx, ...))]
        :param relations: r, contain type and ops types (type: str, (op_type))
        :param relation_weight: weight of relations, dict(relation, weight)
        :return:
        """
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        # pad entity word hidden state to max word num for each sentence
        # entity_hidden = [t.cat([do_pad(
        #     encoder_output[i][t.range(entity_index[i][j][0], entity_index[i][j][1], dtype=t.long, device=device)],
        #     max_l=max_ew_len, pad_elem=0).unsqueeze(0) for j in range(len(entity_index[i]))
        #                         ]) for i in range(b)]  # b x ?s x mw x h*2
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                            max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h

        # generate candidates and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_label = [[c in elements[i] for c in candidates[i]] + [False] * (max_c_len - c_lens[i])
                               for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            # if candidate_label.sum() == 0:
            #     break
            candidate_weight = [[relation_weight[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                for i in range(b)]
            candidate_weight = t.tensor(candidate_weight, dtype=t.float, device=device)
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                # op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                # op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            t_loss = t.sum(t_loss * candidate_weight)
            loss = t_loss if not loss else (t_loss + loss)
            num += candidate_mask.sum().item()

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        type2index[i][candidates[i][j][0]][-1].append(idx)
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return loss / num if loss else t.tensor(0)


class ILTransformer(nn.Module):
    def __init__(self, vocab_size=6000, embedding_size=256, lstm_size=256, head=2, trans_layers=1,
                 max_seq_len=128, all_types=None, trans_type=0):
        """LSTM + Transformer"""
        super(ILTransformer, self).__init__()
        self.encoder = LSTMEncoder(vocab_size, embedding_size, lstm_size)
        hidden_size = 2*lstm_size
        # transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=head)
        # self.DAG_trans = nn.TransformerEncoder(transformer_layer, num_layers=trans_layers)
        self.DAG_trans = MyTransformerEncoder(hidden_size, dropout=0.1, n_head=head, d_ff=4*hidden_size, N=trans_layers)
        self.pos_encoder = MyPositionalEncoding(hidden_size, max_len=max_seq_len)
        self.type2id = {tp: i for i, tp in enumerate(all_types)}
        self.type_embedding = nn.Embedding(len(all_types) + 2, hidden_size)  # all types, cls, sep
        self.type_trans = nn.Linear(hidden_size, hidden_size)
        self.hid_trans = nn.Linear(hidden_size, hidden_size)
        self.cls_id, self.sep_id = len(all_types), len(all_types) + 1
        self.attention = Attention(hidden_size)
        self.cls = Classifier(in_features=hidden_size)
        self.trans_type = trans_type

    def forward(self, x, x_lens, entity_index, entity_type, elements, relations, sim_label=None, c_w=0):
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # hidden state
        # encode input words
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x h*2
        hs = encoder_output.shape[-1]
        # contrastive loss
        loss_c = t.tensor(0)
        if c_w != 0:
            # contrastive loss
            cls_hidden = encoder_output[:, 0]
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1 + t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device,
                                                      dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h*2
        # generate candidates and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_label = [[c in elements[i] for c in candidates[i]] + [False] * (max_c_len - c_lens[i])
                               for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            else:  # w/o op type
                # op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                # op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            trans_out = self.DAG_trans(candidate_hid.float())  # b*mc x 8 x h
            # print('trans out shape = ', trans_out.shape)
            trans_out = trans_out[:, 0].reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc

            # compute loss and update hidden state
            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            t_loss = t.sum(t_loss * candidate_mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += candidate_mask.sum().item()
            layer += 1
            # print('train:', layer, '|', t.sum(predict.ge(0.5).long()).item(), '|', candidate_label.sum().item(), '|',
            #       candidate_mask.sum().item())

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        # type2index[i][candidates[i][j][0]].append(idx)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                pre_candidates[i].extend(candidates[i])
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return (loss/num if loss else t.tensor(0)) * (1-c_w) + loss_c * c_w

    def predict_scratch(self, x, x_lens, entity_index, entity_type, relations, threshold=0.5):
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])

        # hidden state
        # encode input words
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x h*2
        hs = encoder_output.shape[-1]
        # pad entity word hidden state to max word num for each sentence
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0] + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([h.shape[0] + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h*2

        # generate relation candidate and classify & predict
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = copy.deepcopy(entity_type)
        num, layer = 0, 1
        result = [[] for _ in range(b)]
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]

        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            else:  # w/o op type
                # op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                # op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            trans_out = self.DAG_trans(candidate_hid.float())  # b*mc x 8 x h
            # print('trans out shape = ', trans_out.shape)
            trans_out = trans_out[:, 0].reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc
            assert candidate_mask.shape == predict.shape
            predict = predict * candidate_mask

            for i in range(b):
                index2type_prob = {}
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                
                m_r = 1
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        if candidates[i][j][1] in index2type_prob:
                            index2type_prob[candidates[i][j][1]].append((candidates[i][j][0], j, predict[i][j].item()))
                        else:
                            index2type_prob[candidates[i][j][1]] = [(candidates[i][j][0], j, predict[i][j].item())]
                choose_candidates = sorted(list(index2type_prob.items()), key=lambda c: c[0])
                for cs in choose_candidates:
                    cs = list(sorted(cs[1], key=lambda c: c[2], reverse=True))[:m_r]
                    for c in cs:
                        idx = all_elem_hidden[i].shape[0] - 1
                        j = c[1]
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        all_elem_hidden[i] = do_pad(all_elem_hidden[i], max_l=idx + 2, pad_elem=0)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                        index2type[i].append(candidates[i][j][0])
                        result[i].append(candidates[i][j])
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            layer += 1
        return result

    def predict_guided(self, x, x_lens, entity_index, entity_type, elements, relations, threshold=0.5):
        # param
        b, device = x.shape[0], x.device  # batch size, device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # hidden state
        # encode input words
        encoder_output, _ = self.encoder(x, x_lens)  # b x l x h*2
        hs = encoder_output.shape[-1]
        
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device,
                                                      dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h*2
        # generate candidates and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        result = [[] for _ in range(b)]
        gold = [[] for _ in range(b)]
        all_candidate = [[] for _ in range(b)]
        all_prob = [[] for _ in range(b)]

        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
            #               for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            for i_ in range(b):
                all_candidate[i_].append(candidates[i_])
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # 
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            else:  # w/o op type
                # op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                # op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc
            assert candidate_mask.shape == predict.shape
            predict = predict * candidate_mask

            for i in range(b):
                result[i].append([])
                gold[i].append([])
                all_prob[i].append(predict[i].tolist())
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        result[i][-1].append(candidates[i][j])
                    if candidates[i][j] in elements[i]:
                        gold[i][-1].append(candidates[i][j])
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]  # hidden
                        type2index[i][candidates[i][j][0]][-1].append(idx)
            layer += 1
        return result, gold, all_candidate, all_prob


class IBLModel(nn.Module):
    def __init__(self, bert_path, hidden_size=768, relation_types=11):
        super(IBLModel, self).__init__()
        """Bert + DAG-LSTM"""
        self.encoder = BertModel.from_pretrained(bert_path)
        self.DAG_block = TreeLSTM(in_features=hidden_size, out_features=hidden_size,
                                  relation_types=relation_types)
        self.attention = Attention(hidden_size)
        self.cf = Classifier(in_features=hidden_size)

    def forward(self, x, x_mask, entity_index, entity_type, elements, relations, all_type, sim_label=None, c_w=0):
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        # contrastive loss
        loss_c = t.tensor(0)
        if c_w != 0:
            # contrastive loss
            cls_hidden = encoder_output[:, 0]
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1 + t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                            max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        loss, num = None, 0

        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]  # b x ?s
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        layer = 1
        while has_candidate(candidates):
            candidate_label = [[c in elements[i] for c in candidates[i]] for i in range(b)]  # b x ?sc, label
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k

            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_label]  # b x sc
            candidate_label = [l + [False] * (max_c_len - len(l)) for l in candidate_label]
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)

            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index
            )  # b x sc x H
            predict = self.cf(candidate_hidden)  # b x s
            # print('train:', layer, '|', t.sum(predict.ge(0.5).long()).item(), '|', candidate_label.sum().item(), '|',
            #       mask.sum().item())
            layer += 1

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            # weight=candidate_label.masked_fill(candidate_label.le(0), 0.1))
            t_loss = t.sum(t_loss * mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += t.sum(mask)

            # update elem hidden and context by true candidates
            for i in range(b):
                for tp in list(all_type):
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]  # hidden
                        all_elem_context[i][idx] = candidate_context[i][j]  # context
                        # type2index[i][candidates[i][j][0]].append(idx)  # elem type2index
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                # pre_candidates[i].extend(candidates[i])
            # generate new candidates
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        return (loss/num if loss else t.tensor(0)) * (1-c_w) + loss_c * c_w

    def forward_cont(self, x, x_mask, entity_index, entity_type, elements, relations, all_type, sim_label, c_w=0.1):
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]

        # compute contrastive loss
        cls_hidden = encoder_output[:, 0]  # b x h
        sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
        sim_mat = t.triu(sim_mat, diagonal=1)
        # print(sim_mat)
        sim_mat = t.exp(sim_mat) / (1 + t.exp(sim_mat))
        sim_label = t.tensor(sim_label, dtype=t.float, device=device)
        # print(sim_label, sim_mat)
        loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)

        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        loss, num = None, 0

        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]  # b x ?s
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        layer = 1
        while has_candidate(candidates):
            candidate_label = [[c in elements[i] for c in candidates[i]] for i in range(b)]  # b x ?sc, label
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k

            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_label]  # b x sc
            candidate_label = [l + [False] * (max_c_len - len(l)) for l in candidate_label]
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)

            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index
            )  # b x sc x H
            predict = self.cf(candidate_hidden)  # b x s
            # print('train:', layer, '|', t.sum(predict.ge(0.5).long()).item(), '|', candidate_label.sum().item(), '|',
            #       mask.sum().item())
            layer += 1

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            # weight=candidate_label.masked_fill(candidate_label.le(0), 0.1))
            t_loss = t.sum(t_loss * mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += t.sum(mask)

            # update elem hidden and context by true candidates
            for i in range(b):
                for tp in list(all_type):
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]  # hidden
                        all_elem_context[i][idx] = candidate_context[i][j]  # context
                        # type2index[i][candidates[i][j][0]].append(idx)  # elem type2index
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                # pre_candidates[i].extend(candidates[i])
            # generate new candidates
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return (loss / num if loss else t.tensor(0)) * (1-c_w) + loss_c * c_w

    def predict_scratch(self, x, x_mask, entity_index, entity_type, relations, all_type, threshold=0.5):
        b, device = x.shape[0], x.device
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        # hidden state
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get element hidden by Attention, padding to elem_len+1
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0] + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([h.shape[0] + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify & predict
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        num, layer = 0, 1
        result = [[] for _ in range(b)]

        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k
            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_ids]  # b x sc
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index)
            predict = self.cf(candidate_hidden)  # b x s
            assert mask.shape == predict.shape
            predict = predict * mask
            # print('layer %d | predict %d' % (layer, t.sum(predict.ge(threshold).long()).item()))

            for i in range(b):
                index2type_prob = {}
                for tp in all_type:
                    type2index[i][tp].append([])
                
                m_r = 1
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        if candidates[i][j][1] in index2type_prob:
                            index2type_prob[candidates[i][j][1]].append((candidates[i][j][0], j, predict[i][j].item()))
                        else:
                            index2type_prob[candidates[i][j][1]] = [(candidates[i][j][0], j, predict[i][j].item())]
                choose_candidates = sorted(list(index2type_prob.items()), key=lambda c: c[0])
                for cs in choose_candidates:
                    cs = list(sorted(cs[1], key=lambda c: c[2], reverse=True))[:m_r]
                    for c in cs:
                        idx = all_elem_hidden[i].shape[0] - 1
                        j = c[1]
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]
                        all_elem_context[i][idx] = candidate_context[i][j]
                        all_elem_hidden[i] = do_pad(all_elem_hidden[i], max_l=idx + 2, pad_elem=0)
                        all_elem_context[i] = do_pad(all_elem_context[i], max_l=idx + 2, pad_elem=0)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
                        result[i].append(candidates[i][j])

            layer += 1

        
        return result

    def predict_guided(self, x, x_mask, entity_index, entity_type, elements, relations, all_type, threshold=0.5):
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]

        # generate relation candidate and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(all_type, entity_type)
        r_type2idx = generate_re_type2id(relations)
        num, layer = 0, 0
        result = [[] for _ in range(b)]
        gold = [[] for _ in range(b)]
        all_candidate = [[] for _ in range(b)]
        all_prob = [[] for _ in range(b)]

        while layer <= 6:
            # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i]) for i in range(b)]
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            for i_ in range(b):
                all_candidate[i_].append(candidates[i_])
            candidate_ids = [[r_type2idx[c[0]] for c in cs] for cs in candidates]  # b x ?sc
            candidate_index = [[list(c[1]) for c in cs] for cs in candidates]  # b x ?sc x k
            # padding to same shape
            max_c_len = max([len(c) for c in candidate_index])
            mask = [[1] * len(l) + [0] * (max_c_len - len(l)) for l in candidate_ids]  # b x sc
            candidate_ids = [cs + [0] * (max_c_len - len(cs)) for cs in candidate_ids]  # relation type
            candidate_index = [cs + [[-1] * 2 for _ in range(max_c_len - len(cs))] for cs in
                               candidate_index]  # b x sc x 2

            # data to tensor
            mask = t.tensor(mask, dtype=t.float, device=device)
            candidate_ids = t.tensor(candidate_ids, dtype=t.long, device=device)
            candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
            candidate_hidden, candidate_context = self.DAG_block(
                relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
                layer_context=all_elem_context, source=candidate_index)
            predict = self.cf(candidate_hidden)  # b x s
            assert mask.shape == predict.shape
            predict = predict * mask

            for i in range(b):
                result[i].append([])
                gold[i].append([])
                all_prob[i].append(predict[i].tolist())
                for tp in all_type:
                    type2index[i][tp].append([])
                for j in range(len(candidates[i])):
                    if predict[i][j] > threshold:
                        result[i][-1].append(candidates[i][j])
                    if candidates[i][j] in elements[i]:
                        gold[i][-1].append(candidates[i][j])
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = candidate_hidden[i][j]  # hidden
                        all_elem_context[i][idx] = candidate_context[i][j]  # context
                        type2index[i][candidates[i][j][0]][-1].append(idx)  # elem type2index
            layer += 1
        return result, gold, all_candidate, all_prob


class FlatBTModel(nn.Module):
    def __init__(self, bert_path, hidden_size=768, relation_types=10, n_head=4, trans_layer=3):
        super(FlatBTModel, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(bert_path)
        self.DAG_trans = MyTransformerEncoder(
            hidden_size, dropout=0.1, n_head=n_head, d_ff=4*hidden_size, N=trans_layer)
        self.pos_encoder = MyPositionalEncoding(hidden_size, max_len=64)
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 2*hidden_size)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2*hidden_size, relation_types, bias=False)
        self.bias = nn.Parameter(t.zeros(relation_types))
        self.fc2.bias = self.bias
        self.type_embedding = nn.Embedding(2, hidden_size)

    def forward(self, x, x_mask, entity_index, label=None, sim_label=None, c_w=0):
        b, device = x.shape[0], x.device
        entity_words_len = [[entity_index[i][j][1]-entity_index[i][j][0]
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1]))+[0]*(max_ew_len-entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_entity_hidden = t.cat(
            [self.attention(h, m) for h, m in zip(entity_hidden, entity_mask)], dim=0).reshape([b, -1, hs])
        cls_emb = self.type_embedding(t.tensor([0]*b, dtype=t.long, device=device))
        sep_emb = self.type_embedding(t.tensor([1]*b, dtype=t.long, device=device))
        candidate_hid = t.cat(
            [cls_emb, all_entity_hidden[:, 0], sep_emb, all_entity_hidden[:, 1]], dim=-1).reshape([b, -1, hs])
        candidate_hid = self.pos_encoder(candidate_hid)
        trans_out = self.DAG_trans(candidate_hid)[:, 0]
        predict = self.fc2(self.drop1(self.fc1(trans_out)))
        loss = F.cross_entropy(predict, label, reduce=False).sum() / x.shape[0] if label is not None else None
        if c_w != 0:
            cls_hidden = encoder_output[:, 0]
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1+t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
            loss = loss * (1-c_w) + loss_c * c_w
        return predict, loss


class FlatBLModel(nn.Module):
    def __init__(self, bert_path, hidden_size=768, relation_types=10):
        super(FlatBLModel, self).__init__()
        # self.encoder = BertModel.from_pretrained(bert_path)
        self.encoder = AutoModel.from_pretrained(bert_path)
        # self.encoder = RobertaModel.from_pretrained(bert_path)
        self.DAG_block = TreeLSTM(in_features=hidden_size, out_features=hidden_size, relation_types=1)
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 2*hidden_size)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2 * hidden_size, relation_types, bias=False)
        self.bias = nn.Parameter(t.zeros(relation_types))
        self.fc2.bias = self.bias

    def forward(self, x, x_mask, entity_index, label=None, sim_label=None, c_w=0):
        b, device = x.shape[0], x.device
        entity_words_len = [[entity_index[i][j][1]-entity_index[i][j][0]
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1])) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_entity_hidden = t.cat(
            [self.attention(h, m) for h, m in zip(entity_hidden, entity_mask)], dim=0).reshape([b, -1, hs])
        all_entity_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_entity_hidden]
        candidate_hid, _ = self.DAG_block(
            relation_ids=t.tensor([[0] for _ in range(b)], dtype=t.long, device=device),
            layer_hidden=all_entity_hidden, layer_context=all_entity_context,
            source=t.tensor([[[0, 1]] for _ in range(b)], dtype=t.long, device=device)
        )
        predict = self.fc2(self.drop1(self.fc1(candidate_hid.squeeze(1))))
        loss = F.cross_entropy(predict, label, reduce=False).sum() / b if label is not None else None
        if c_w != 0:
            cls_hidden = encoder_output[:, 0]
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1+t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
            loss = loss * (1-c_w) + loss_c * c_w
        return predict, loss


class FlatLLModel(nn.Module):
    def __init__(self, vocab_size=6000, embedding_size=256, hidden_size=256, relation_types=10):
        super(FlatLLModel, self).__init__()
        self.encoder = LSTMEncoder(vocab_size, embedding_size, hidden_size)
        self.DAG_block = TreeLSTM(in_features=2*hidden_size, out_features=2*hidden_size, relation_types=1)
        self.attention = Attention(hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*2, 2*hidden_size)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2*hidden_size, relation_types, bias=False)
        self.bias = nn.Parameter(t.zeros(relation_types))
        self.fc2.bias = self.bias

    def forward(self, x, x_lens, entity_index, label=None, sim_label=None, c_w=0):
        b, device = x.shape[0], x.device
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0]
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, x_lens)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1])) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_entity_hidden = t.cat(
            [self.attention(h, m) for h, m in zip(entity_hidden, entity_mask)], dim=0).reshape([b, -1, hs])
        all_entity_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_entity_hidden]
        candidate_hid, _ = self.DAG_block(
            relation_ids=t.tensor([[0] for _ in range(b)], dtype=t.long, device=device),
            layer_hidden=all_entity_hidden, layer_context=all_entity_context,
            source=t.tensor([[[0, 1]] for _ in range(b)], dtype=t.long, device=device)
        )
        predict = self.fc2(self.drop1(self.fc1(candidate_hid.squeeze(1))))
        loss = F.cross_entropy(predict, label, reduce=False).sum() / b if label is not None else None
        if c_w != 0:
            cls_hidden = encoder_output[:, 0]
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1+t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
            loss = loss * (1-c_w) + loss_c * c_w
        return predict, loss


class BTsModel(nn.Module):
    def __init__(self, bert_path, hidden_size, head, trans_layer, all_types, rel_types, trans_type=0):
        super(BTsModel, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        self.DAG_trans = MyTransformerEncoder(hidden_size, 0.1, head, 4*hidden_size, trans_layer)
        self.pos_encoder = MyPositionalEncoding(hidden_size)
        self.type2id = {tp: i for i, tp in enumerate(all_types)}
        self.type_embedding = nn.Embedding(len(all_types)+2, hidden_size)
        self.type_trans = nn.Linear(hidden_size, hidden_size)
        self.hid_trans = nn.Linear(hidden_size, hidden_size)
        self.cls_id, self.sep_id = len(all_types), len(all_types) + 1
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 2*hidden_size)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2*hidden_size, len(rel_types)+1, bias=False)
        self.bias = nn.Parameter(t.zeros(len(rel_types)+1))
        self.fc2.bias = self.bias
        self.trans_type = trans_type
        self.rel = rel_types
        self.rel2id = {r: i for i, r in enumerate(rel_types+['no'])}

    def forward(self, x, x_mask, entity_index, entity_type, elements, relations, sim_label=None, c_w=0):
        b, device = x.shape[0], x.device
        elem_lens = [len(elem) for elem in elements]
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        # contrastive loss
        loss_c = t.tensor(0)
        if c_w != 0:
            cls_hidden = encoder_output[:, 0]
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1 + t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                            max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        c2label = [{(elem[1][0], elem[1][1]): elem[0] for elem in elements[i][len(entity_index[i]):]} for i in range(b)]
        candidates = [generate_candidate3(relations, type2index[i]) for i in range(b)]
        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)
            candidate_label = [
                [self.rel2id[c2label[i].get(c, 'no')] for c in candidates[i]]
                + [len(self.rel2id)-1]*(max_c_len-c_lens[i]) for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.long, device=device)
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            op1_type_ids = [[self.type2id[index2type[i][c[0]]] for c in candidates[i]] + [0]*(max_c_len-c_lens[i])
                            for i in range(b)]
            op2_type_ids = [[self.type2id[index2type[i][c[1]]] for c in candidates[i]] + [0]*(max_c_len-c_lens[i])
                            for i in range(b)]
            op1_index = [[c[0] for c in candidates[i]] + [0]*(max_c_len-c_lens[i]) for i in range(b)]
            op2_index = [[c[1] for c in candidates[i]] + [0]*(max_c_len-c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]
            if self.trans_type == 0:
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat([cls_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                                      dim=-1).reshape([b*max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat([cls_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                                      dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat([cls_emb, op1_hid, sep_emb, op2_hid], dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat([cls_emb, op1_hid, sep_emb, op2_hid], dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]
            trans_out = trans_out.reshape([b, max_c_len, -1])
            prob = self.fc2(self.drop1(self.fc1(trans_out)))
            t_loss = F.cross_entropy(prob.reshape([b*max_c_len, -1]), candidate_label.reshape([-1]), reduce=False)
            t_loss = t.sum(t_loss * candidate_mask.reshape([-1]))
            loss = t_loss if loss is None else (t_loss+loss)
            num += candidate_mask.sum().item()

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j] != len(self.rel2id)-1:
                        idx = elements[i].index((self.rel[candidate_label[i][j]], candidates[i][j]))
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        type2index[i][self.rel[candidate_label[i][j]]][-1].append(idx)
            candidates = [generate_candidate3(relations, type2index[i]) for i in range(b)]
        return (loss/num if loss else t.tensor(0)) * (1-c_w) + loss_c * c_w

    def predict_scratch(self, x, x_mask, entity_index, entity_type, relations):
        b, device = x.shape[0], x.device
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0]+1, pad_elem=0)
                           if h.shape[0] else t.zeros([h.shape[0]+1, hs], dtype=t.float, device=device)
                           for h, m in zip(entity_hidden, entity_mask)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = copy.deepcopy(entity_type)
        num, layer = 0, 1
        result = [[] for _ in range(b)]
        while layer <= 6:
            candidates = [generate_candidate3(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # 
            
            op1_type_ids = [[self.type2id[index2type[i][c[0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                            for i in range(b)]
            op2_type_ids = [[self.type2id[index2type[i][c[1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                            for i in range(b)]
            op1_index = [[c[0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]
            if self.trans_type == 0:
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat([cls_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                                      dim=-1).reshape([b*max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat([cls_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                                      dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat([cls_emb, op1_hid, sep_emb, op2_hid], dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                candidate_hid = t.cat([cls_emb, op1_hid, sep_emb, op2_hid], dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]
            trans_out = trans_out.reshape([b, max_c_len, -1])
            prob = self.fc2(self.drop1(self.fc1(trans_out)))
            predict = prob.softmax(-1).argmax(-1)

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if predict[i][j] != len(self.rel2id)-1:
                        idx = all_elem_hidden[i].shape[0]-1
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        all_elem_hidden[i] = do_pad(all_elem_hidden[i], max_l=idx+2, pad_elem=0)
                        type2index[i][self.rel[predict[i][j]]][-1].append(idx)
                        index2type[i].append(self.rel[predict[i][j]])
                        result[i].append((self.rel[predict[i][j]], candidates[i][j]))
            layer += 1
        return result


class OverlapBTModel(nn.Module):
    def __init__(self, bert_path, hidden_size=768, relation_types=10, n_head=4, trans_layer=3):
        super(OverlapBTModel, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        self.DAG_trans = MyTransformerEncoder(
            hidden_size, dropout=0.1, n_head=n_head, d_ff=4 * hidden_size, N=trans_layer)
        self.pos_encoder = MyPositionalEncoding(hidden_size, max_len=64)
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2 * hidden_size, relation_types, bias=False)
        self.bias = nn.Parameter(t.zeros(relation_types))
        self.fc2.bias = self.bias
        self.type_embedding = nn.Embedding(2, hidden_size)
        self.rel_num = relation_types

    def forward(self, x, x_mask, entity_index, candidates, label):
        b, device = x.shape[0], x.device
        en_lens = [len(entity_index[i]) for i in range(b)]
        entity_words_len = [[entity_index[i][j][1]-entity_index[i][j][0]+1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_en_len, max_ew_len = max(en_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1]+1)) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_en_len+1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_en_len+1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]

        c_lens = [len(c) for c in candidates]
        max_c_len = max(c_lens)
        candidate_label = [label[i] + [0] * (max_c_len-c_lens[i]) for i in range(b)]
        candidate_label = t.tensor(candidate_label, dtype=t.long, device=device)
        candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len-c_lens[i]) for i in range(b)],
                                  dtype=t.float, device=device)
        op1_index = [[c[0] for c in candidates[i]] + [0] * (max_c_len-c_lens[i]) for i in range(b)]
        op2_index = [[c[1] for c in candidates[i]] + [0] * (max_c_len-c_lens[i]) for i in range(b)]
        cls_index = [[0] * max_c_len for _ in range(b)]
        sep_index = [[1] * max_c_len for _ in range(b)]
        op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                         for i in range(b)])
        op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                         for i in range(b)])
        cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
        sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
        candidate_hid = t.cat([cls_emb, op1_hid, sep_emb, op2_hid], dim=-1).reshape([b*max_c_len, -1, hs])
        candidate_hid = self.pos_encoder(candidate_hid)
        trans_out = self.DAG_trans(candidate_hid.float())[:, 0]
        trans_out = trans_out.reshape([b, max_c_len, -1])
        predict = self.fc2(self.drop1(self.fc1(trans_out)))
        t_loss = F.cross_entropy(predict.reshape([b*max_c_len, -1]), candidate_label.reshape([-1]), reduce=False)
        t_loss = t.sum(t_loss * candidate_mask.reshape([-1]))
        return t_loss

    def predict(self, x, x_mask, entity_index, candidates):
        b, device = x.shape[0], x.device
        en_lens = [len(entity_index[i]) for i in range(b)]
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0]+1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_en_len, max_ew_len = max(en_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1]+1)) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_en_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_en_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]

        c_lens = [len(c) for c in candidates]
        max_c_len = max(c_lens)
        op1_index = [[c[0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
        op2_index = [[c[1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
        cls_index = [[0] * max_c_len for _ in range(b)]
        sep_index = [[1] * max_c_len for _ in range(b)]
        op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                         for i in range(b)])
        op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                         for i in range(b)])
        cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
        sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
        candidate_hid = t.cat([cls_emb, op1_hid, sep_emb, op2_hid], dim=-1).reshape([b * max_c_len, -1, hs])
        candidate_hid = self.pos_encoder(candidate_hid)
        trans_out = self.DAG_trans(candidate_hid.float())[:, 0]
        trans_out = trans_out.reshape([b, max_c_len, -1])
        prob = self.fc2(self.drop1(self.fc1(trans_out))).softmax(-1)
        predict = prob.argmax(-1)
        result = [[] for _ in range(b)]
        for i in range(b):
            for j in range(c_lens[i]):
                if predict[i][j] != self.rel_num-1:
                    result[i].append((predict[i][j], candidates[i][j]))
        return result


class OverlapBLModel(nn.Module):
    def __init__(self, bert_path, hidden_size, relation_types=10):
        super(OverlapBLModel, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        self.DAG_block = TreeLSTM(in_features=hidden_size, out_features=hidden_size, relation_types=1)
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2 * hidden_size, relation_types, bias=False)
        self.bias = nn.Parameter(t.zeros(relation_types))
        self.fc2.bias = self.bias
        self.rel_num = relation_types

    def forward(self, x, x_mask, entity_index, candidates, label):
        b, device = x.shape[0], x.device
        en_lens = [len(entity_index[i]) for i in range(b)]
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_en_len, max_ew_len = max(en_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1] + 1)) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_en_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_en_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]
        c_lens = [len(c) for c in candidates]
        max_c_len = max(c_lens)
        candidate_label = [label[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
        candidate_label = t.tensor(candidate_label, dtype=t.long, device=device)
        candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                  dtype=t.float, device=device)
        candidate_ids = t.zeros([b, max_c_len], dtype=t.long, device=device)
        candidate_index = [[list(c) for c in cs] + [[-1, -1] for _ in range(max_c_len-len(cs))] for cs in candidates]
        candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
        candidate_hid, _ = self.DAG_block(
            relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
            layer_context=all_elem_context, source=candidate_index
        )
        predict = self.fc2(self.drop1(self.fc1(candidate_hid)))
        loss = F.cross_entropy(predict.reshape([b*max_c_len, -1]), candidate_label.reshape([-1]), reduce=False)
        loss = t.sum(loss * candidate_mask.reshape([-1]))
        return loss

    def predict(self, x, x_mask, entity_index, candidates):
        b, device = x.shape[0], x.device
        en_lens = [len(entity_index[i]) for i in range(b)]
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_en_len, max_ew_len = max(en_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1] + 1)) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_en_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_en_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]
        c_lens = [len(c) for c in candidates]
        max_c_len = max(c_lens)
        candidate_ids = t.zeros([b, max_c_len], dtype=t.long, device=device)
        candidate_index = [[list(c) for c in cs] + [[-1, -1] for _ in range(max_c_len - len(cs))] for cs in candidates]
        candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
        candidate_hid, _ = self.DAG_block(
            relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
            layer_context=all_elem_context, source=candidate_index
        )
        prob = self.fc2(self.drop1(self.fc1(candidate_hid))).softmax(-1)
        predict = prob.argmax(-1)
        result = [[] for _ in range(b)]
        for i in range(b):
            for j in range(c_lens[i]):
                if predict[i][j] != self.rel_num - 1:
                    result[i].append((predict[i][j], candidates[i][j]))
        return result


class OverlapLLModel(nn.Module):
    def __init__(self, vocab_size=6000, embedding_size=256, hidden_size=256, relation_types=10):
        super(OverlapLLModel, self).__init__()
        self.encoder = LSTMEncoder(vocab_size, embedding_size, hidden_size)
        self.DAG_block = TreeLSTM(in_features=2*hidden_size, out_features=2*hidden_size, relation_types=1)
        self.attention = Attention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 2 * hidden_size)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2 * hidden_size, relation_types, bias=False)
        self.bias = nn.Parameter(t.zeros(relation_types))
        self.fc2.bias = self.bias
        self.rel_num = relation_types

    def forward(self, x, x_lens, entity_index, candidates, label):
        b, device = x.shape[0], x.device
        en_lens = [len(entity_index[i]) for i in range(b)]
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0]+1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_en_len, max_ew_len = max(en_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, x_lens)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1]+1)) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_en_len+1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_en_len+1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]
        c_lens = [len(c) for c in candidates]
        max_c_len = max(c_lens)
        candidate_label = [label[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
        candidate_label = t.tensor(candidate_label, dtype=t.long, device=device)
        candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                  dtype=t.float, device=device)
        candidate_ids = t.zeros([b, max_c_len], dtype=t.long, device=device)
        candidate_index = [[list(c) for c in cs] + [[-1, -1] for _ in range(max_c_len - len(cs))] for cs in candidates]
        candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
        candidate_hid, _ = self.DAG_block(
            relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
            layer_context=all_elem_context, source=candidate_index
        )
        predict = self.fc2(self.drop1(self.fc1(candidate_hid)))
        loss = F.cross_entropy(predict.reshape([b * max_c_len, -1]), candidate_label.reshape([-1]), reduce=False)
        loss = t.sum(loss * candidate_mask.reshape([-1]))
        return loss

    def predict(self, x, x_lens, entity_index, candidates):
        b, device = x.shape[0], x.device
        en_lens = [len(entity_index[i]) for i in range(b)]
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_en_len, max_ew_len = max(en_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, x_lens)[0]
        hs = encoder_output.shape[-1]
        entity_hidden = [encoder_output[i][t.tensor(
            [list(range(entity_index[i][j][0], entity_index[i][j][1] + 1)) + [0] * (max_ew_len - entity_words_len[i][j])
             for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_en_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_en_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        all_elem_context = [t.zeros(h.shape, dtype=t.float, device=device) for h in all_elem_hidden]
        c_lens = [len(c) for c in candidates]
        max_c_len = max(c_lens)
        candidate_ids = t.zeros([b, max_c_len], dtype=t.long, device=device)
        candidate_index = [[list(c) for c in cs] + [[-1, -1] for _ in range(max_c_len - len(cs))] for cs in candidates]
        candidate_index = t.tensor(candidate_index, dtype=t.long, device=device)
        candidate_hid, _ = self.DAG_block(
            relation_ids=candidate_ids, layer_hidden=all_elem_hidden,
            layer_context=all_elem_context, source=candidate_index
        )
        prob = self.fc2(self.drop1(self.fc1(candidate_hid))).softmax(-1)
        predict = prob.argmax(-1)
        result = [[] for _ in range(b)]
        for i in range(b):
            for j in range(c_lens[i]):
                if predict[i][j] != self.rel_num - 1:
                    result[i].append((predict[i][j], candidates[i][j]))
        return result

