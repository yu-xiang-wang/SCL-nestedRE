# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : framework.py
# @Software: PyCharm


import os
from models import IModel, ITransformer, ILTransformer, IBLModel, FlatBTModel, FlatBLModel, FlatLLModel
from models import BTsModel, OverlapBTModel
from data_utils import get_data_loader1, get_data_loader2, load_vocab, get_type_info
from data_utils import save_predict, analysis_relation_distribution, generate_ne_entity_replace
from utils import set_random, compute_metric, calc_running_avg_loss, compute_metric_by_dist, print_dist
from utils import save_config, compute_metric_by_dist_layers, compute_similarity_examples
from utils import relation_filtration_study, relation_filtration_study_guide
import datetime
import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import argparse
import copy
import json
import time


class FrameWorkIM(object):
    def __init__(self, config):
        print('model name is LSTM + LSTM')
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        # print(self.all_type)
        self.word2id, self.id2word = load_vocab(config.vocab_file)
        self.model = IModel(vocab_size=config.vocab_size, embedding_size=config.embedding_size,
                            lstm_size=config.hidden_size, dag_size=config.hidden_size * 2,
                            relation_types=len(self.relations)).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        
        self.model_dir = os.path.join(config.model_dir, 'LL-%d' % (100 * config.cont_weight))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

    def _setup_train(self):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.lr}
        ])

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def load_model(self, model_file):
        self._load_model(model_file)

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def train(self, train_file=None):
        train_iter = get_data_loader1(self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
                                      word2id=self.word2id, data_file=train_file)
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                time.sleep(0.003)
                self.model.train()
                self.optimizer.zero_grad()

                # data prepare
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len = batch['sent_len']
                entity, element = batch['entity'], batch['element']
                entity_type = [[e[0] for e in se] for se in entity]
                entity_index = [[e[1] for e in se] for se in entity]

                loss = self.model(x=sentence, x_lens=sentence_len, entity_index=entity_index, entity_type=entity_type,
                                  elements=element, relations=self.relations, all_type=self.all_type,
                                  sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss.item():
                    loss.backward()
                else:
                    print(batch['sid'])
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                # print info when training
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1, batch_len,
                        round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = get_data_loader1(op, self.data_dir, self.config.batch_size, word2id=self.word2id)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_candidate, guide_c_prob = [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len = batch['sent_len']
                entity, element = batch['entity'], batch['element']
                entity_type = [[e[0] for e in se] for se in entity]
                entity_index = [[e[1] for e in se] for se in entity]
                predict = self.model.predict_scratch(
                    x=sentence, x_lens=sentence_len, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations, all_type=self.all_type)
                if self.config.guide:
                    g_p, g_e, g_c, g_prob = self.model.predict_guided(
                        x=sentence, x_lens=sentence_len, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, all_type=self.all_type
                    )
                    guide_predict.extend(g_p)
                    guide_elem.extend(g_e)
                    guide_candidate.extend(g_c)
                    guide_c_prob.extend(g_prob)
                all_predict.extend(predict)
                all_elem.extend(element)
                all_entity.extend(entity)
        if op == self.config.test_prefix:
            all_examples = []
            for batch in valid_iter:
                for sid, entity, element in zip(batch['sid'], batch['entity'], batch['element']):
                    all_examples.append({'sid': sid, 'entity': entity, 'element': element})
            print('*' * 10 + f"predict on test dataset and save to {self.model_dir}/predict.json" + '*' * 10)
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
            analysis_relation_distribution(all_elem, all_entity, [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'labeled_analysis.csv'))
            analysis_relation_distribution([e + p for e, p in zip(all_entity, all_predict)], all_entity,
                                           [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'predict_analysis.csv'))
            relation_filtration_study(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            if op == self.config.test_prefix:
                relation_filtration_study_guide(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            print('#' * 5, 'guide result', '#' * 5)
            print_dist(gp, gr, gf1, [r_[0] for r_ in self.relations],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'))
            print('[INFO] {} | guided auc result: {}'.format(
                datetime.datetime.now(), compute_auc(guide_candidate, guide_c_prob, guide_elem)))
        return p, r, f1

    def eval_pipeline(self, data_file):
        valid_iter = get_pipeline_loader1(self.config.batch_size, self.word2id, self.config.max_sent_len, data_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | pipeline batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_raw_entity, all_new_entity = [], [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len = batch['sent_len']
                entity_type = [[e[0] for e in se] for se in batch['new_entity']]
                entity_index = [[e[1] for e in se] for se in batch['new_entity']]
                predict = self.model.predict_scratch(
                    x=sentence, x_lens=sentence_len, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations, all_type=self.all_type)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_raw_entity.extend(batch['raw_entity'])
                all_new_entity.extend(batch['new_entity'])
                # print('[INFO] {} | batch: {}'.format(datetime.datetime.now(), idx))
        p, r, f1 = compute_pipeline_metric(all_elem, all_predict, all_raw_entity, all_new_entity)
        print('[INFO] {} | pipeline test result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), p, r, f1))

    def predict(self):
        pass


class FrameWorkTrans(object):
    def __init__(self, config):
        print('model name is BERT + Trans')
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.model = ITransformer(config.bert_dir, config.max_sent_len, config.bert_hidden,
                                  config.bert_head, config.transformer_layer,
                                  self.all_type, config.trans_type).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model_dir = os.path.join(config.model_dir, 'BT-%d-%d' % (config.trans_type, 100 * config.cont_weight))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def load_model(self, model_file):
        self._load_model(model_file)

    def _setup_train(self):
       
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])
        self.scheduler = t.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def train(self, train_file=None):
        train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=train_file)
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                time.sleep(0.003)
                self.model.train()
                self.optimizer.zero_grad()

                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                  elements=batch['element'], relations=self.relations,
                                  sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1, batch_len,
                        round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = get_data_loader2(op, self.data_dir, batch_size=self.config.batch_size // 2,
                                      tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_candidate, guide_c_prob = [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations)
                if self.config.guide:
                    g_p, g_e, g_c, g_prob = self.model.predict_guided(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations
                    )
                    guide_predict.extend(g_p)
                    guide_elem.extend(g_e)
                    guide_candidate.extend(g_c)
                    guide_c_prob.extend(g_prob)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        if op == self.config.test_prefix:
            all_examples = []
            for batch in valid_iter:
                for sid, entity, element in zip(batch['sid'], batch['entity'], batch['element']):
                    all_examples.append({'sid': sid, 'entity': entity, 'element': element})
            print(
                f"------------------------------predict on test dataset and save to {self.model_dir}/predict.json----------------------------------------")
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
            analysis_relation_distribution(all_elem, all_entity, [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'labeled_analysis.csv'))
            analysis_relation_distribution([e + p for e, p in zip(all_entity, all_predict)], all_entity,
                                           [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'predict_analysis.csv'))
            relation_filtration_study(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            if op == self.config.test_prefix:
                relation_filtration_study_guide(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            print('#' * 5, 'guide result', '#' * 5)
            print_dist(gp, gr, gf1, [r_[0] for r_ in self.relations],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'))
            print('[INFO] {} | guided auc result: {}'.format(
                datetime.datetime.now(), compute_auc(guide_candidate, guide_c_prob, guide_elem)))
        return p, r, f1

    def eval_pipeline(self, data_file):
        valid_iter = get_pipeline_loader2(self.config.batch_size, self.tokenizer,
                                          max_seq_len=self.config.max_sent_len, data_file=data_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | pipeline batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_raw_entity, all_new_entity = [], [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                new_entity_type = [[e[0] for e in se] for se in batch['new_entity']]
                new_entity_index = [[e[1] for e in se] for se in batch['new_entity']]
                predict = self.model.predict_scratch(x=sentence, x_mask=mask, entity_index=new_entity_index,
                                                     entity_type=new_entity_type, relations=self.relations)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_raw_entity.extend(batch['raw_entity'])
                all_new_entity.extend(batch['new_entity'])
        p, r, f1 = compute_pipeline_metric(all_elem, all_predict, all_raw_entity, all_new_entity)
        print('[INFO] {} | pipeline test result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), p, r, f1))

    def predict(self):
        pass


class FrameWorkLTrans(object):
    def __init__(self, config):
        print('model name is LSTM + Trans')
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model = ILTransformer(
            vocab_size=self.tokenizer.vocab_size, embedding_size=config.embedding_size, lstm_size=config.hidden_size,
            head=config.bert_head, trans_layers=config.transformer_layer,
            max_seq_len=config.max_sent_len, all_types=self.all_type, trans_type=config.trans_type
        ).to(config.device)
        self.model_dir = os.path.join(config.model_dir, 'LT-%d-%d' % (config.trans_type, config.cont_weight * 100))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def _setup_train(self):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.lr}
        ])

    def train(self, train_file=None):
        
        train_iter = get_data_loader2(self.config.train_prefix, self.data_dir, self.config.batch_size,
                                      self.tokenizer, self.config.max_sent_len, train_file)
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                time.sleep(0.003)
                self.model.train()
                self.optimizer.zero_grad()

                
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                sentence_len = sentence.ne(self.tokenizer.pad_token_id).sum(-1).cpu()
                entity, element = batch['entity'], batch['element']
                entity_type = [[e[0] for e in se] for se in entity]
                entity_index = [[e[1] for e in se] for se in entity]

                loss = self.model(x=sentence, x_lens=sentence_len, entity_index=entity_index, entity_type=entity_type,
                                  elements=element, relations=self.relations,
                                  sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1, batch_len,
                        round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        # valid_iter = get_data_loader1(op, self.data_dir, self.config.batch_size, word2id=self.word2id)
        valid_iter = get_data_loader2(op, self.data_dir, self.config.batch_size // 2,
                                      self.tokenizer, self.config.max_sent_len)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_candidate, guide_c_prob = [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
             
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                sentence_len = sentence.ne(self.tokenizer.pad_token_id).sum(-1).cpu()
                entity, element = batch['entity'], batch['element']
                entity_type = [[e[0] for e in se] for se in entity]
                entity_index = [[e[1] for e in se] for se in entity]
                predict = self.model.predict_scratch(x=sentence, x_lens=sentence_len, entity_index=entity_index,
                                                     entity_type=entity_type, relations=self.relations)
                if self.config.guide:
                    g_p, g_e, g_c, g_prob = self.model.predict_guided(
                        x=sentence, x_lens=sentence_len, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations
                    )
                    guide_predict.extend(g_p)
                    guide_elem.extend(g_e)
                    guide_candidate.extend(g_c)
                    guide_c_prob.extend(g_prob)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        if op == self.config.test_prefix:
            all_examples = []
            for batch in valid_iter:
                for sid, entity, element in zip(batch['sid'], batch['entity'], batch['element']):
                    all_examples.append({'sid': sid, 'entity': entity, 'element': element})
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
            analysis_relation_distribution(all_elem, all_entity, [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'labeled_analysis.csv'))
            analysis_relation_distribution([e + p for e, p in zip(all_entity, all_predict)], all_entity,
                                           [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'predict_analysis.csv'))
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            print('#' * 5, 'guide result', '#' * 5)
            print_dist(gp, gr, gf1, [r_[0] for r_ in self.relations],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'))
            print('[INFO] {} | guided auc result: {}'.format(
                datetime.datetime.now(), compute_auc(guide_candidate, guide_c_prob, guide_elem)))
        return p, r, f1


class FrameWorkBLM(object):
    def __init__(self, config):
        print('model name is BERT + LSTM')
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        # print(self.all_type)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model = IBLModel(bert_path=config.bert_dir, hidden_size=config.bert_hidden,
                              relation_types=len(self.relations)).to(config.device)
        self.model_dir = os.path.join(config.model_dir, 'BL-%d' % (100 * config.cont_weight))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def load_model(self, model_file):
        self._load_model(model_file)

    def _setup_train(self):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])

    def train(self, train_file=None):
        train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=train_file)
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                time.sleep(0.003)
                self.model.train()
                self.optimizer.zero_grad()

                # data prepare
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                  elements=batch['element'], relations=self.relations, all_type=self.all_type,
                                  sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
          
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1, batch_len,
                        round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = get_data_loader2(op, self.data_dir, batch_size=self.config.batch_size // 2,
                                      tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_candidate, guide_c_prob = [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(iter(valid_iter)):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations, all_type=self.all_type)
                if self.config.guide:
                    g_p, g_e, g_c, g_prob = self.model.predict_guided(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, all_type=self.all_type
                    )
                    guide_predict.extend(g_p)
                    guide_elem.extend(g_e)
                    guide_candidate.extend(g_c)
                    guide_c_prob.extend(g_prob)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        if op == self.config.test_prefix:
            all_examples = []
            for batch in valid_iter:
                for sid, entity, element in zip(batch['sid'], batch['entity'], batch['element']):
                    all_examples.append({'sid': sid, 'entity': entity, 'element': element})
            print(
                f"------------------------------predict on test dataset and save to {self.model_dir}/predict.json----------------------------------------")
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
            analysis_relation_distribution(all_elem, all_entity, [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'labeled_analysis.csv'))
            analysis_relation_distribution([e + p for e, p in zip(all_entity, all_predict)], all_entity,
                                           [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'predict_analysis.csv'))
            relation_filtration_study(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            if op == self.config.test_prefix:
                relation_filtration_study_guide(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            print('#' * 5, 'guide result', '#' * 5)
            print_dist(gp, gr, gf1, [r_[0] for r_ in self.relations],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'))
            print('[INFO] {} | guided auc result: {}'.format(
                datetime.datetime.now(), compute_auc(guide_candidate, guide_c_prob, guide_elem)))
        return p, r, f1

    def eval_pipeline(self, data_file):
        valid_iter = get_pipeline_loader2(self.config.batch_size, self.tokenizer,
                                          self.config.max_sent_len, data_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | pipeline batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_raw_entity, all_new_entity = [], [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                new_entity_type = [[e[0] for e in se] for se in batch['new_entity']]
                new_entity_index = [[e[1] for e in se] for se in batch['new_entity']]
                predict = self.model.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=new_entity_index,
                    entity_type=new_entity_type, relations=self.relations, all_type=self.all_type)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_raw_entity.extend(batch['raw_entity'])
                all_new_entity.extend(batch['new_entity'])
        p, r, f1 = compute_pipeline_metric(all_elem, all_predict, all_raw_entity, all_new_entity)
        print('[INFO] {} | pipeline test result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), p, r, f1))

    def predict(self):
        pass


class RuleWork(object):
    def __init__(self, config):
        print('model name is rule')
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.model_dir = 'rule'

    def predict_by_rule(self, op='train'):
        examples = [
            json.loads(s) for s in open(os.path.join(self.data_dir, '%s.json' % op)).readlines()]
        predict, element, entity = [], [], []
        for e in examples:
            predict.append(self.predict_one(e['sentence'], [(ee[0], tuple(ee[1])) for ee in e['entity']]))
            # predict.append([])
            entity.append([(ee[0], tuple(ee[1])) for ee in e['entity']])
            element.append(entity[-1] + [(ee[0], tuple(ee[1])) for ee in e['element'][len(entity[-1]):]])
        # return predict
        p, r, f1 = compute_metric(element, predict, entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), 'rule', p, r, f1))
        mp, mr, mf = compute_metric_by_dist(element, predict, entity, [r_[0] for r_ in self.relations])
        print_dist(mp, mr, mf, [r_[0] for r_ in self.relations])

    def predict_one(self, sentence, entity):
        type2index = {tp: [] for tp in self.all_type}
        for i, e in enumerate(entity):
            type2index[e[0]].append(i)
        elem = entity
        at_relation = [x[1] for x in self.relations if x[0] == '@'][0]
        for ops in at_relation:
            r_c = [tuple()]
            for op in ops:
                cur_c = []
                op_idx = []
                for opt in op:
                    op_idx.extend(type2index[opt])
                for m in op_idx:
                    for c in r_c:
                        if len(c) == 0 or entity[c[0]][1][1] < entity[m][1][0]:
                            cur_c.append(c + tuple([m]))
                r_c = list(set(cur_c))
                if not r_c:
                    break
            tc = list(set([('@', c) for c in r_c]))
            elem.extend(tc)
        type2index['@'].extend(list(range(len(entity), len(elem))))
        key_words = {'>': ['大于', '以上', '>', '超过', '高于'],
                     '<': ['小于', '以下', '<', '低于', '少于'],
                     '=': ['等于', '为'],
                     '>=': ['大于等于', '>=', '满', '达到', '达', '不低于', '不少于', '及以上'],
                     '<=': ['小于等于', '<=', '不高于', '不超过', '及以下']}
        other_relations = [x for x in self.relations if x[0] in ['>=', '<=']] + [
            x for x in self.relations if x[0] in ['<', '>']]
        for r in other_relations:
            flag = False
            for w in key_words[r[0]]:
                if w in sentence:
                    flag = True
                    break
            if not flag:
                continue
            for ops in r[1]:
                r_c = [tuple()]
                for op in ops:
                    cur_c = []
                    op_idx = []
                    for opt in op:
                        op_idx.extend(type2index[opt])
                    for m in op_idx:
                        for c in r_c:
                            if len(c) == 0 or entity[c[0]][1][1] < entity[m][1][0]:
                                cur_c.append(c + tuple([m]))
                    r_c = list(set(cur_c))
                    if not r_c:
                        break
                tc = list(set([(r[0], c) for c in r_c]))
                elem.extend(tc)
            if flag:
                break
        return elem


use_roberta = False


class FlatFrameWork(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config.flat_dir, config.flat_name)
        if not use_roberta:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'Flat-%s' % config.flat_name)
        self.relation = json.loads(open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])[
            'relation']
        self.rel2id = {}
        self.id2rel = []
        self.ng_rel = ['no_relation', 'Other']
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
            if r not in self.ng_rel:
                self.rel2id['re-' + r] = len(self.rel2id)
                self.id2rel.append('re-' + r)
        print(len(self.rel2id), list(self.rel2id.keys()))
        self.model = None

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load %s model from epoch = %d' % (model_file, state['epoch']))

    def _setup_train(self):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])

    def get_loader(self, op='train', do_enforce=False):
        from data_utils import NDataSet
        from torch.utils.data import DataLoader

        def convert_entity(raw_words, entity):
            raw_id2n_id = []
            n_c = 0
            raw_id2n_id.append(n_c)
            n_c += len(self.tokenizer.tokenize(raw_words[0]))
            sp_words = self.tokenizer.tokenize(raw_words[0])
            for w in raw_words[1:]:
                raw_id2n_id.append(n_c)
                n_c += len(self.tokenizer.tokenize(w if not use_roberta else (' ' + w)))
                sp_words += self.tokenizer.tokenize(w if not use_roberta else (' ' + w))
                # raw_id2n_id.extend([len(raw_id2n_id)]*len(self.tokenizer.tokenize(w)))
            raw_id2n_id.append(n_c)
            converted_entities = []
            for e in entity:
                converted_entities.append((e[0], (raw_id2n_id[e[1][0]], raw_id2n_id[e[1][1]])))
            if sp_words != self.tokenizer.tokenize(' '.join(raw_words)):
                print('raw: ' + ' '.join(raw_words))
                print('sp: ' + ' '.join(sp_words))
            return converted_entities

        def collate_fn(batch):
            sentence, words, entity, label = [], [], [], []
            for idx, line in enumerate(batch):
                e = eval(line)
                t_sentence = ' '.join(e['token'])
                # t_words = ['[CLS]'] + self.tokenizer.tokenize(t_sentence)
                t_words = self.tokenizer.tokenize(t_sentence)
                # print(t_sentence, t_words)
                t_word_ids = self.tokenizer.convert_tokens_to_ids(t_words)
                r_entity = [(x['name'], x['pos']) for x in [e['h'], e['t']]]
                # t_entity = convert_entity(['[CLS]']+e['token'], r_entity)
                t_entity = convert_entity(e['token'], r_entity)
                sentence.append(t_sentence)
                t_word_pad_ids = t_word_ids[:self.config.max_flat_len] \
                                 + [self.tokenizer.pad_token_id] * (self.config.max_flat_len - len(t_word_ids))
                words.append(t_word_pad_ids)
                entity.append(t_entity)
                if e['relation'] in self.ng_rel:
                    label.append(e['relation'])
                    if op != 'train':
                        continue
                    if do_enforce:
                        sentence.append(t_sentence)
                        words.append(t_word_pad_ids)
                        entity.append(t_entity[::-1])
                        label.append(e['relation'])
                else:
                    t_label = e['relation'][:-7]
                    label.append(t_label if '(e1,e2)' in e['relation'] else ('re-' + t_label))
                    if op != 'train':
                        label[-1] = e['relation']
                        continue
                    if do_enforce:
                        sentence.append(t_sentence)
                        words.append(t_word_pad_ids)
                        entity.append(t_entity[::-1])
                        label.append(t_label if '(e2,e1)' in e['relation'] else ('re-' + t_label))
            return {'sentence': sentence, 'words': words, 'entity': entity, 'label': label}

        _collate_fn = collate_fn
        file = os.path.join(self.data_dir, '%s.txt' % op)
        data_iter = DataLoader(dataset=NDataSet(file), batch_size=self.config.batch_size,
                               shuffle=(op == 'train'), collate_fn=_collate_fn)
        return data_iter

    def compute_label_sim(self, labels):
        import numpy as np

        b = len(labels)
        sim = np.zeros([b, b], dtype=float)
        for i in range(b):
            for j in range(i + 1, b):
                sim[i][j] = 1 \
                    if labels[i] == labels[j] or labels[i] == ('re-' + labels[j]) or ('re-' + labels[i]) == labels[
                    j] else 0
        return sim

    def compute_metric(self, label, predict, op='valid'):
        for i in range(len(predict)):
            if predict[i] not in self.ng_rel:
                predict[i] = (predict[i] + '(e1,e2)') if 're-' not in predict[i] else (predict[i][3:] + '(e2,e1)')
        if op == 'test':
            res_file = os.path.join(self.model_dir, 'result.csv')
            writer = open(res_file, 'w', encoding='utf-8')
            for i in range(len(label)):
                writer.write(','.join([label[i], predict[i]]) + '\n')
            writer.close()
        
        rel2id = {(item[0] + '(e1,e2)') if 're-' not in item[0] else (item[0][3:] + '(e2,e1)'): item[1]
                  for item in self.rel2id.items()}
        TP, FP, FN = [0] * len(self.rel2id), [0] * len(self.rel2id), [0] * len(self.rel2id)
        for i in range(len(label)):
            if label[i] not in self.ng_rel:
                TP[rel2id[label[i]]] += int(label[i] == predict[i])
                FP[rel2id[label[i]]] += int(label[i] != predict[i])
            if predict[i] not in self.ng_rel:
                FN[rel2id[predict[i]]] += int(label[i] != predict[i])
        tp, fp, fn = sum(TP), sum(FP), sum(FN)
        p = tp / (tp + fp) if tp != 0 else 0
        r = tp / (tp + fn) if tp != 0 else 0
        f1 = 2 * p * r / (p + r) if tp != 0 else 0
        return p, r, f1

    def train(self):
        train_iter = self.get_loader(op='train')
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                label_id = [self.rel2id[r] for r in batch['label']]
                label = t.tensor(label_id, dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob, loss = self.model(ids, mask, entity_index, label)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op='valid')
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob, _ = self.model(ids, mask, entity_index)
                prob = prob.softmax(-1).argmax(-1)
                predict.extend(prob.tolist())
                label.extend(batch['label'])
        p, r, f1 = self.compute_metric(label, [self.id2rel[r] for r in predict])
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class FlatBTFrameWork(FlatFrameWork):
    def __init__(self, config):
        super(FlatBTFrameWork, self).__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.flat_dir, config.flat_name)
        if not use_roberta:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'FlatBT-%s' % config.flat_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        self.ng_rel = ['no_relation', 'Other']
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
            if r not in self.ng_rel:
                self.rel2id['re-' + r] = len(self.rel2id)
                self.id2rel.append('re-' + r)
        self.model = FlatBTModel(config.bert_en, config.bert_hidden, len(self.rel2id),
                                 n_head=config.bert_head, trans_layer=config.transformer_layer).to(config.device)

    def train(self):
        train_iter = self.get_loader(op='train')
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                label_id = [self.rel2id[r] for r in batch['label']]
                label = t.tensor(label_id, dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob, loss = self.model(ids, mask, entity_index, label)
                
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op='valid')
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob, _ = self.model(ids, mask, entity_index)
                prob = prob.softmax(-1).argmax(-1)
                predict.extend(prob.tolist())
                label.extend(batch['label'])
        p, r, f1 = self.compute_metric(label, [self.id2rel[r] for r in predict], op=op)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class FlatBLFrameWork(FlatFrameWork):
    def __init__(self, config):
        super(FlatBLFrameWork, self).__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.flat_dir, config.flat_name)
        if not use_roberta:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'FlatBL-%s' % config.flat_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        self.ng_rel = ['no_relation', 'Other']
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
            if r not in self.ng_rel:
                self.rel2id['re-' + r] = len(self.rel2id)
                self.id2rel.append('re-' + r)
        self.model = FlatBLModel(config.bert_en, config.bert_hidden, len(self.rel2id)).to(config.device)

    def _setup_train(self):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])

    def train(self):
        train_iter = self.get_loader(op='train')
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                label_id = [self.rel2id[r] for r in batch['label']]
                label = t.tensor(label_id, dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob, loss = self.model(ids, mask, entity_index, label)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op='valid')
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob, _ = self.model(ids, mask, entity_index)
                prob = prob.softmax(-1).argmax(-1)
                predict.extend(prob.tolist())
                label.extend(batch['label'])
        p, r, f1 = self.compute_metric(label, [self.id2rel[r] for r in predict], op=op)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class FlatLLFrameWork(FlatFrameWork):
    def __init__(self, config):
        super(FlatLLFrameWork, self).__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.flat_dir, config.flat_name)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'FlatLL-%s' % config.flat_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        self.ng_rel = ['no_relation', 'Other']
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
            if r not in self.ng_rel:
                self.rel2id['re-' + r] = len(self.rel2id)
                self.id2rel.append('re-' + r)
        self.model = FlatLLModel(self.tokenizer.vocab_size, config.embedding_size,
                                 config.bert_hidden, relation_types=len(self.rel2id)).to(config.device)

    def _setup_train(self):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.lr}
        ])

    def train(self):
        train_iter = self.get_loader(op='train')
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                label_id = [self.rel2id[r] for r in batch['label']]
                label = t.tensor(label_id, dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                lens = ids.ne(self.tokenizer.pad_token_id).sum(-1).cpu()
                prob, loss = self.model(ids, lens, entity_index, label)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op='valid')
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                lens = ids.ne(self.tokenizer.pad_token_id).sum(-1).cpu()
                prob, _ = self.model(ids, lens, entity_index)
                prob = prob.softmax(-1).argmax(-1)
                predict.extend(prob.tolist())
                label.extend(batch['label'])
        p, r, f1 = self.compute_metric(label, [self.id2rel[r] for r in predict], op=op)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class FlatB(FlatFrameWork):
    def __init__(self, config):
        super(FlatB, self).__init__(config)
        pass


class SFrameWork(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.rel_type = [r[0] for r in self.relations]
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model_dir = os.path.join(config.model_dir, 'SModel')
        self.config.model_dir = self.model_dir
        self.model = None

    def _setup_train(self):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def train(self):
        pass

    def eval(self):
        pass


class BTS(SFrameWork):
    def __init__(self, config):
        super(BTS, self).__init__(config)
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.rel_type = [r[0] for r in self.relations]
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model_dir = os.path.join(config.model_dir, 'BTSModel-%d' % config.trans_type)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.model = BTsModel(config.bert_dir, config.bert_hidden, config.bert_head, config.transformer_layer,
                              self.all_type, self.rel_type, config.trans_type).to(config.device)

    def train(self, train_file=None):
        train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=train_file
        )
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()

                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model(
                    x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                    elements=batch['element'], relations=self.relations,
                    sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight
                )
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1, batch_len,
                        round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = get_data_loader2(op, self.data_dir, batch_size=self.config.batch_size // 2,
                                      tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations
                )

                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class OverlapFM(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config.overlap_dir, config.overlap_name)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'Overlap-%s' % config.overlap_name)
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
        self.rel2id['Other'] = len(self.rel2id)
        self.id2rel.append('Other')
        print('relation type num = ', len(self.rel2id))
        self.model = None

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load %s model from epoch = %d' % (model_file, state['epoch']))

    def _setup_train(self, lr):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def get_loader(self, op='train', do_enforce=False):
        from data_utils import NDataSet
        from torch.utils.data import DataLoader

        def convert_entity(raw_words, entity):
            raw_id2n_id = []
            n_c = 0
            for w in raw_words:
                raw_id2n_id.append(n_c)
                n_c += len(self.tokenizer.tokenize(w))
                # raw_id2n_id.extend([len(raw_id2n_id)]*len(self.tokenizer.tokenize(w)))
            converted_entities = []
            for e in entity:
                converted_entities.append((e[0], (raw_id2n_id[e[1][0] + 1], raw_id2n_id[e[1][1] + 1])))
            return converted_entities

        def collate_fn(batch):
            sentence, words, entity, relation, candidate, candidate_label = [], [], [], [], [], []
            for idx, line in enumerate(batch):
                e = eval(line)
                t_sentence = ' '.join(e['tokens'])
                t_words = ['[CLS]'] + self.tokenizer.tokenize(t_sentence)
                t_word_ids = self.tokenizer.convert_tokens_to_ids(t_words)
                t_relations = e['relation']
                t_entity = []
                for r in t_relations:
                    t_entity.extend([tuple(r['e1'].values()), tuple(r['e2'].values())])
                t_entity_set = sorted(set(t_entity), key=lambda x: x[1])
                t_entity_pair2label = {}
                for r in t_relations:
                    e_pair = (t_entity_set.index(tuple(r['e1'].values())), t_entity_set.index(tuple(r['e2'].values())))
                    t_entity_pair2label[e_pair] = r['label']
                t_candidate, t_label = [], []
                for i in range(len(t_entity_set)):
                    for j in range(len(t_entity_set)):
                        t_candidate.append((i, j))
                        t_label.append(self.rel2id[t_entity_pair2label.get((i, j), 'Other')])
                sentence.append(t_sentence)
                t_word_pad_ids = t_word_ids[:self.config.max_overlap_len] \
                                 + [self.tokenizer.pad_token_id] * (self.config.max_overlap_len - len(t_word_ids))
                words.append(t_word_pad_ids)
                entity.append(convert_entity(['[CLS]'] + e['tokens'], [(en[0], (en[1], en[1])) for en in t_entity_set]))
                candidate.append(t_candidate)
                candidate_label.append(t_label)
                relation.append(
                    [(r['label'], (r['e1']['entity'], r['e2']['entity'])) for i, r in enumerate(e['relation'])])
            return {'sentence': sentence, 'words': words, 'entity': entity,
                    'candidate': candidate, 'label': candidate_label, 'relation': relation}

        _collate_fn = collate_fn
        file = os.path.join(self.data_dir, '%s.txt' % op)
        data_iter = DataLoader(dataset=NDataSet(file), batch_size=self.config.batch_size,
                               shuffle=(op == 'trains'), collate_fn=_collate_fn)
        return data_iter

    def compute_metric(self, label, predict):
        assert len(label) == len(predict)
        cnt, c1, c2 = 0, 0, 0
        for i in range(len(label)):
            # cur_predict = [(self.id2rel[r[0]], r[1]) for r in predict[i]]
            cnt += len(set(label[i]) & set(predict[i]))
            c1 += len(set(label[i]))
            c2 += len(set(predict[i]))
        p = cnt / c2 if cnt != 0 else 0
        r = cnt / c1 if cnt != 0 else 0
        f1 = 2 * p * r / (p + r) if cnt != 0 else 0
        return p, r, f1

    def train(self):
        pass

    def eval(self):
        pass


class OverlapBTFM(OverlapFM):
    def __init__(self, config):
        super(OverlapBTFM, self).__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.overlap_dir, config.overlap_name)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'OverlapBT-%s' % config.overlap_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
        self.rel2id['Other'] = len(self.rel2id)
        self.id2rel.append('Other')
        self.model = OverlapBTModel(config.bert_en, config.bert_hidden, len(self.rel2id),
                                    config.bert_head, config.transformer_layer).to(config.device)

    def train(self):
        train_iter = self.get_loader(op='train')
        batch_len = len(train_iter)
        self._setup_train(self.config.bert_lr)
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model(ids, mask, entity_index, batch['candidate'], batch['label'])
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval('valid')
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                t_predict = self.model.predict(ids, mask, entity_index, batch['candidate'])
                label.extend(batch['relation'])
                predict.extend([[(self.id2rel[r[0]], (batch['entity'][i][r[1][0]][0], batch['entity'][i][r[1][1]][0]))
                                 for r in t_predict[i]] for i in range(len(t_predict))])
        p, r, f1 = self.compute_metric(label, predict)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class OverlapBLFM(OverlapFM):
    def __init__(self, config):
        super(OverlapBLFM, self).__init__(config)
        from models import OverlapBLModel
        self.config = config
        self.data_dir = os.path.join(config.overlap_dir, config.overlap_name)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'OverlapBL-%s' % config.overlap_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
        self.rel2id['Other'] = len(self.rel2id)
        self.id2rel.append('Other')
        self.model = OverlapBLModel(config.bert_en, config.bert_hidden, len(self.rel2id)).to(config.device)

    def train(self):
        train_iter = self.get_loader(op='train')
        batch_len = len(train_iter)
        self._setup_train(self.config.bert_lr)
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model(ids, mask, entity_index, batch['candidate'], batch['label'])
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval('valid')
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                t_predict = self.model.predict(ids, mask, entity_index, batch['candidate'])
                label.extend(batch['relation'])
                predict.extend([[(self.id2rel[r[0]], (batch['entity'][i][r[1][0]][0], batch['entity'][i][r[1][1]][0]))
                                 for r in t_predict[i]] for i in range(len(t_predict))])
        p, r, f1 = self.compute_metric(label, predict)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class OverlapLLFM(OverlapFM):
    def __init__(self, config):
        super(OverlapLLFM, self).__init__(config)
        from models import OverlapLLModel
        self.config = config
        self.data_dir = os.path.join(config.overlap_dir, config.overlap_name)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'OverlapLL-%s' % config.overlap_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
        self.rel2id['Other'] = len(self.rel2id)
        self.id2rel.append('Other')
        self.model = OverlapLLModel(self.tokenizer.vocab_size, config.embedding_size,
                                    config.bert_hidden, len(self.rel2id)).to(config.device)

    def train(self):
        train_iter = self.get_loader(op='train')
        batch_len = len(train_iter)
        self._setup_train(self.config.bert_lr)
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                lens = ids.ne(self.tokenizer.pad_token_id).sum(-1).cpu()
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model(ids, lens, entity_index, batch['candidate'], batch['label'])
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval('valid')
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                lens = ids.ne(self.tokenizer.pad_token_id).sum(-1).cpu()
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                t_predict = self.model.predict(ids, lens, entity_index, batch['candidate'])
                label.extend(batch['relation'])
                predict.extend([[(self.id2rel[r[0]], (batch['entity'][i][r[1][0]][0], batch['entity'][i][r[1][1]][0]))
                                 for r in t_predict[i]] for i in range(len(t_predict))])
        p, r, f1 = self.compute_metric(label, predict)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


def my_job():
    from config import Config
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model_name', type=str, default='all', help='模型类型，多个模型使用|分隔')
    parser.add_argument('--do_k_fold', type=int, default=0, help='k折交叉验证')
    parser.add_argument('--entity_replace', type=int, default=0, help='使用实体替换增强策略')
    args = parser.parse_args()
    model_dict = {'LL': FrameWorkIM, 'BT': FrameWorkTrans, 'LT': FrameWorkLTrans, 'BL': FrameWorkBLM}
    if args.model_name == 'rule':
        config = Config()
        model = RuleWork(config)
        model.predict_by_rule(op=config.valid_prefix)
        return
    model_names = args.model_name.split('-') if args.model_name != 'all' else list(model_dict.keys())
    for model_name in model_names:
        model = model_dict[model_name]
        if args.do_k_fold == 0:
            config = Config()
           
            set_random(config.random_seed)
            fm = model(config=config)
            fm.train()
        else:
            for i in range(args.do_k_fold):  # -1, -1, -1):
                config = Config()
                set_random(config.random_seed)
                print('doing %d of %d fold training' % (i + 1, args.do_k_fold))
                config.data_dir = os.path.join(config.data_dir, '%d-%d' % (args.do_k_fold, i + 1))
                new_train_file = None
                if args.entity_replace != 0:
                    new_train_file = os.path.join(config.data_dir,
                                                  '%s-%s.json' % (config.generated_prefix, config.train_prefix))
                    generate_ne_entity_replace(
                        train_file=os.path.join(config.data_dir, '%s.json' % config.train_prefix),
                        new_file=new_train_file
                    )
                    config.train_prefix = '%s-%s' % (config.generated_prefix, config.train_prefix)
                config.model_dir = os.path.join(config.model_dir, '%d-%d' % (args.do_k_fold, i + 1))
                fm = model(config=config)
                fm.train(train_file=new_train_file)
   


def flat_job():
    from config import Config
    idx1, idx2 = 0, 0
    flat_model = [('FlatBT', FlatBTFrameWork), ('FlatBL', FlatBLFrameWork), ('FlatLL', FlatLLFrameWork)
                  ][idx1:idx2 + 1]
    # flat_model = [FlatBTFrameWork, FlatBLFrameWork, FlatLLFrameWork][idx1:idx2+1]
    # model_name = ['FlatBTFrameWork', 'FlatBLFrameWork', 'FlatLLFrameWork'][idx1:idx2+1]
    lr_list = [5e-5, 1e-5, 5e-6]
    bs_list = [8, 16][1:]
    flat_name = ['kbp37', 'semeval2010']
    for fn_ in range(len(flat_name)):
        for fm_ in range(len(flat_model)):
            for lr_ in range(len(lr_list)):
                for bs_ in range(len(bs_list)):
                    config = Config()
                    set_random(config.random_seed + 2)
                    config.train_epoch = 30
                    config.bert_en = '../../PLM/roberta-base'
                    config.model_dir = 'model-flat'
                    config.flat_name = flat_name[fn_]
                    config.lr, config.bert_lr = lr_list[lr_], lr_list[lr_]
                    config.batch_size = bs_list[bs_]
                    print('train model = %s | flat name = %s | lr = %.6f | bs = %d' % (
                        flat_model[fm_][0], flat_name[fn_], lr_list[lr_], bs_list[bs_]
                    ))
                    fm = flat_model[fm_][1](config)
                    fm.train()


def deal_flat_out():
    log_file = 'E:\master\python\RE\\relation\PLM\flat-r2.out'
    out_file = 'E:\master\python\RE\\relation\PLM\flat-r2.csv'
    lines = open(log_file, 'r', encoding='utf-8').readlines()
    data_name, model_name, lr, bs = [0] * 4
    writer = open(out_file, 'w', encoding='utf-8')
    writer.write(','.join(['dataset', 'model', 'lr', 'bs', 'p', 'r', 'f']) + '\n')
    for i in range(len(lines)):
        if 'train model' in lines[i]:
            words = lines[i].strip().split()
            data_name, model_name, lr, bs = words[3], words[8], words[12], words[16]
        elif 'test result' in lines[i]:
            words = lines[i].strip().split()
            p, r, f = words[8][:-1], words[11][:-1], words[14][:-1]
            writer.write(','.join([data_name, model_name, lr, bs, p, r, f]) + '\n')
    writer.close()


def overlap_job():
    from config import Config
    idx1, idx2 = 0, 2
    overlap_model = [OverlapBTFM, OverlapBLFM, OverlapLLFM][idx1: idx2 + 1]
    model_name = ['OverlapBTFM', 'OverlapBLFM', 'OverlapLLFM'][idx1: idx2 + 1]
    lr_list = [1e-5, 5e-6]
    for i, m in enumerate(overlap_model):
        for lr in lr_list:
            config = Config()
            set_random(config.random_seed)
            config.batch_size = 16
            config.model_dir = 'model-overlap'
            print('#' * 10 + ' ' * 5 + 'train model %s with lr = %f' % (model_name[i], lr))
            if 'LL' in model_name[i]:
                config.lr = lr
            else:
                config.bert_lr = lr
            fm = m(config)
            fm.train()


def my_job_s():
    """"""
    from config import Config
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model_name', type=str, default='all', help='Model Type')
    parser.add_argument('--do_k_fold', type=int, default=0, help='k-cross validation')
    parser.add_argument('--entity_replace', type=int, default=0, help='Entity Enhancement')

    args = parser.parse_args()
    model_dict = {'LL': FrameWorkIM, 'BT': BTS, 'LT': FrameWorkLTrans, 'BL': FrameWorkBLM}
    if args.model_name == 'rule':
        config = Config()
        model = RuleWork(config)
        model.predict_by_rule(op=config.valid_prefix)
        return
    model_names = args.model_name.split('-') if args.model_name != 'all' else list(model_dict.keys())
    for model_name in model_names:
        model = model_dict[model_name]
        if args.do_k_fold == 0:
            config = Config()
            set_random(config.random_seed)
            fm = model(config=config)
            fm.train()
        else:
            for i in range(args.do_k_fold):  # -1, -1, -1):
                config = Config()
                set_random(config.random_seed)
                print('doing %d of %d fold training' % (i + 1, args.do_k_fold))
                config.data_dir = os.path.join(config.data_dir, '%d-%d' % (args.do_k_fold, i + 1))
                if args.entity_replace != 0:
                    generate_ne_entity_replace(
                        train_file=os.path.join(config.data_dir, '%s.json' % config.train_prefix),
                        new_file=os.path.join(config.data_dir, '%s-%s.json' % (config.generated_prefix,
                                                                               config.train_prefix))
                    )
                    config.train_prefix = '%s-%s' % (config.generated_prefix, config.train_prefix)
                config.model_dir = os.path.join(config.model_dir, '%d-%d' % (args.do_k_fold, i + 1))
                fm = model(config=config)
                fm.train()


def my_few_job():
    """"""
    from config import Config
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model_name', type=str, default='all', help='Model Type')
    parser.add_argument('--do_k_fold', type=int, default=0, help='k-cross validation')
    parser.add_argument('--entity_replace', type=int, default=0, help='Entity Enhancement')
    parser.add_argument('--few_ratio', type=int, default=10, help='sample percentage')
    args = parser.parse_args()
    model_dict = {'LL': FrameWorkIM, 'LT': FrameWorkLTrans, 'BL': FrameWorkBLM, 'BT': FrameWorkTrans}
    model_names = args.model_name.split('-') if args.model_name != 'all' else list(model_dict.keys())
    ratio_list = [10, 20, 30, 50][2:3]
    lr_list = [1e-5, 5e-6]
    for model_name in model_names:
        for ratio in ratio_list:
            for lr in lr_list:
                print('train %s with data ratio = %d and lr = %f' % (model_name, ratio, lr))
                model = model_dict[model_name]
                config = Config()
                set_random(config.random_seed)
                config.lr, config.bert_lr = lr, lr
                config.model_dir = os.path.join(config.model_dir, 'FS/%d' % ratio)
                config.p_step = 4
                config.batch_size = 4
                fm = model(config)
                fm.train(train_file=os.path.join(config.data_dir, '%s-%d.json' % (config.few_labeled_prefix, ratio)))
         
    cw_list = [0.1, 0.2, 0.3]
    for model_name in ['BT']:
        for cw in cw_list:
            for ratio in ratio_list:
                for lr in lr_list:
                    print('train contrastive %d %s with data ratio = %d and lr = %f' % (
                        int(cw * 10), model_name, ratio, lr))
                    model = model_dict[model_name]
                    config = Config()
                    set_random(config.random_seed)
                    config.cont_weight = cw
                    config.lr, config.bert_lr = lr, lr
                    config.model_dir = os.path.join(config.model_dir, 'FS/%d' % ratio)
                    config.p_step = 4
                    config.batch_size = 8
                    fm = model(config)
                    fm.train(
                        train_file=os.path.join(config.data_dir, '%s-%d.json' % (config.few_labeled_prefix, ratio)))


def my_one_job():
    from config import Config
    lr_list = [1e-5, 5e-6]
    cw_list = [0, 0.1, 0.2, 0.3]
    bs_list = [4, 8, 16]
    for cw in cw_list:
        for lr in lr_list:
            for bs in bs_list:
                for k in range(1, 6):
                    print('train fold %d model on lr = %.6f, cont-weight = %.2f, batch-size = %d' % (k, lr, cw, bs))
                    config = Config()
                    config.cont_weight = cw
                    config.lr = lr
                    config.batch_size = bs
                    config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
                    config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
                    set_random(config.random_seed)
                    fm = FrameWorkIM(config=config)
                    fm.train()


def my_cw_job():
    from config import Config
    import numpy as np
    # cw_list = [0.1, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    cw_list = [round(x, 2) for x in np.arange(0.55, 1, 0.05)]
    for cw in cw_list:
        for i in range(5):
            config = Config()
            config.cont_weight = cw
            config.batch_size = 4
            set_random(config.random_seed)
            print('doing %d of 5 fold training with cont weight = %.3f' % (i + 1, config.cont_weight))
            config.data_dir = os.path.join(config.data_dir, '5-%d' % (i + 1))
            config.model_dir = os.path.join(config.model_dir, '5-%d' % (i + 1))
            fm = FrameWorkTrans(config=config)
            fm.train()


def my_all_job():
    from config import Config
    nest_model = [('INN', FrameWorkIM), ('BERT+DAG-LSTM', FrameWorkBLM), ('BERT+Transformer', FrameWorkTrans),
                  ('SCL-nestedRE', FrameWorkTrans)]
    # nest_model = [('INN', FrameWorkIM), ('INN-BERT', FrameWorkBLM)]
    flat_model = [('INN', FlatLLFrameWork), ('BERT+DAG-LSTM', FlatBLFrameWork), ('BERT+Transformer', FlatBTFrameWork)]
    overlap_model = [('INN', OverlapLLFM), ('INN-BERT', OverlapBLFM)]
    # nest_model = [('SCL-nestedRE', FrameWorkTrans)]
    # 
    do_w_policy = True
    bs = [8]  # 
    bert_lr = [1e-5]  # 
    lr = [2e-5]
    random_seeds = [111, 66, 1234, 999, 6789]  # 
    cont_weight = [i / 100 for i in range(0, 100, 5)]
    # cont_weight = [0.1]
    for model_idx in range(len(nest_model)):
        for bs_idx in range(len(bs)):
            for lr_idx in range(len(lr)):
                for k in range(0, 1):
                    for random_seed in random_seeds[1:2]:
                        for cw in cont_weight:
                            if not do_w_policy:
                                continue
                            if model_idx not in {3}:
                                continue
                            config = Config()
                            config.train_epoch = 40
                            config.random_seed = random_seed
                            config.device = "cuda:1"
                            
                            config.cont_weight, config.lr, config.bert_lr, config.batch_size = \
                                0, lr[lr_idx], bert_lr[lr_idx], bs[bs_idx]
                           
                            config.cont_weight = cw
                            
                            config.model_dir = 'figure-6-5'
                            config.data_dir = os.path.join(config.data_dir, '5-%d' % (k + 1))
                            config.model_dir = os.path.join(config.model_dir, '5-%d' % (k + 1))
                            set_random(config.random_seed)
                            print(
                                'Model Introduction: %s, bs = %d, bert_lr = %.6f, lr = %.6f, fold = %d, use_scheme = %d, cont_weight = %.2f, random = %d' % (
                                    nest_model[model_idx][0], bs[bs_idx], bert_lr[lr_idx], lr[lr_idx], k + 1,
                                    config.use_schema, config.cont_weight, random_seed))
                            fm = nest_model[model_idx][1](config)
                            fm.train()

    # low-data percentage
    do_few_policy = False
    bs = [2, 4, 8]
    ratio = [50]
    for ratio_idx in range(len(ratio)):
        for model_idx in range(len(nest_model)):
            for bs_idx in range(len(bs)):
                for lr_idx in range(len(lr)):
                    for random_seed in random_seeds:
                        for cw in cont_weight:
                            if not do_few_policy:
                                continue
                            if model_idx < 3:
                                continue
                            print('Model Introduction: %s, low-data ratio = %d, bs = %d, lr = %.6f, bert_lr = %.6f, random_seed = %d' % (
                                nest_model[model_idx][0], ratio[ratio_idx], bs[bs_idx], lr[lr_idx], bert_lr[lr_idx], random_seed))
                            # try:
                            config = Config()
                            config.device = "cuda:0"
                            config.random_seed = random_seed
                            config.train_epoch = 40
                            config.cont_weight, config.lr, config.bert_lr, config.batch_size = 0, lr[lr_idx], bert_lr[lr_idx], bs[bs_idx]
                            if nest_model[model_idx][0] == 'SCL-nestedRE':
                                config.cont_weight = cw
                            config.model_dir = os.path.join(config.model_dir, 'FS/%d' % ratio[ratio_idx])
                            set_random(config.random_seed)
                            fm = nest_model[model_idx][1](config)
                            fm.train(train_file=os.path.join(config.data_dir, '%s-%d.json' % (
                                config.few_labeled_prefix, ratio[ratio_idx])))
                            # except:
                            #     print('error occurred')

    # flat
    do_flat = False
    flat_name = ['kbp37', 'semeval2010']
    bs = [2, 4, 8, 16]  # 
    lr = [1e-5]  # 
    bert_lr = [2e-5]  # 
    for flat_idx in range(len(flat_name)):
        for model_idx in range(len(flat_model)):
            for bs_idx in range(len(bs)):
                for lr_idx in range(len(lr)):
                    for random_seed in random_seeds:
                        if not do_flat:
                            continue
                        if model_idx != 2 or flat_idx != 1:
                            continue
                        config = Config()
                        config.random_seed = random_seed
                        
                        set_random(random_seed)
                        config.device = "cuda:0"
                        config.cont_weight, config.lr, config.bert_lr, config.batch_size = 0, lr[lr_idx], bert_lr[lr_idx], bs[bs_idx]
                        print('Model Introduction: %s, flat = %s, bs = %d, lr = %.6f, bert_lr = %.6f, random_seed = %d' % (
                            flat_model[model_idx][0], flat_name[flat_idx], bs[bs_idx], lr[lr_idx], bert_lr[lr_idx], random_seed))
                        config.model_dir = os.path.join(config.model_dir, 'flat')
                        config.flat_name = flat_name[flat_idx]
                        fm = flat_model[model_idx][1](config)
                        fm.train()

    # overlap
    do_overlap = False
    overlap_name = ['NYT', 'WebNLG']
    bs = [8]
    lr = [1e-5]
    for overlap_idx in range(len(overlap_name)):
        for model_idx in range(len(flat_model)):
            for bs_idx in range(len(bs)):
                for lr_idx in range(len(lr)):
                    for random_seed in random_seeds:
                        if not do_overlap:
                            continue
                        if overlap_idx == 0:
                            continue
                        print('Model Introduction: %s, overlap = %s, bs = %d, lr = %.6f, random = %d' % (
                            overlap_model[model_idx][0], overlap_name[overlap_idx], bs[bs_idx], lr[lr_idx],
                            random_seed))
                        config = Config()
                        config.random_seed = random_seed
                        set_random(config.random_seed)
                        config.device = 'cuda:1'
                        config.cont_weight, config.lr, config.bert_lr, config.batch_size = \
                            0, lr[lr_idx], lr[lr_idx], bs[bs_idx]
                        config.model_dir = os.path.join(config.model_dir, 'overlap')
                        config.overlap_name = overlap_name[overlap_idx]
                        fm = overlap_model[model_idx][1](config)
                        fm.train()
                       


def do_roberta_job():
    from config import Config
    bs_list = [8, 16][1:]
    lr_list = [1e-5, 5e-6]
    cw_list = [0, 0.1, 0.2, 0.3]
    nest_model = [('BL', FrameWorkBLM), ('BT', FrameWorkTrans)][1:]
    for m_ in range(len(nest_model)):
        for lr_ in range(len(lr_list)):
            for bs_ in range(len(bs_list)):
                for cw_ in range(len(cw_list)):
                    for k in range(1, 6):
                        print('train model = %s | fold = %d | lr = %.6f | cont-weight = %.2f | batch-size = %d' % (
                            nest_model[m_][0], k, lr_list[lr_], cw_list[cw_], bs_list[bs_]))
                        config = Config()
                        config.train_epoch = 30
                        config.cont_weight = cw_list[cw_]
                        config.lr = lr_list[lr_]
                        config.bert_lr = lr_list[lr_]
                        config.batch_size = bs_list[bs_]
                        config.bert_dir = '../../PLM/ch-roberta'
                        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
                        config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
                        set_random(config.random_seed)
                        fm = nest_model[m_][1](config)
                        fm.train()


def deal_robert_out():
    """"""
    log_file = 'E:\master\python\RE\\relation\PLM\position.out'
    out_file = 'E:\master\python\RE\\relation\PLM\position.csv'
    lines = open(log_file, 'r', encoding='utf-8').readlines()
    layer_gold_num = [394, 209, 61, 7]
    layer_g = [392, 198, 52, 5]
    model_name, fold, lr, cw, bs = [0] * 5
    writer = open(out_file, 'w', encoding='utf-8')
    writer.write(','.join(['model', 'bs', 'lr', 'cw', 'fold', 'p', 'r', 'f',
                           'p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'r4', 'f1', 'f2', 'f3', 'f4',
                           'gp', 'gr', 'gf', 'gp1', 'gp2', 'gp3', 'gp4', 'gr1', 'gr2', 'gr3', 'gr4',
                           'gf1', 'gf2', 'gf3', 'gf4', 'f34', 'gf34']) + '\n')
    for i in range(len(lines)):
        if 'train model' in lines[i]:
            words = lines[i].strip().split()
            model_name, fold, lr, cw, bs = words[3], words[7], words[11], words[15], words[19]
        elif 'test result' in lines[i]:
            words = lines[i].strip().split()
            p, r, f = words[8][:-1], words[11][:-1], words[14]
            p1, p2, p3, p4 = [lines[i + 17 + i_].strip().split()[-1] for i_ in range(4)]
            r1, r2, r3, r4 = [lines[i + 43 + i_].strip().split()[-1] for i_ in range(4)]
            f1, f2, f3, f4 = [lines[i + 69 + i_].strip().split()[-1] for i_ in range(4)]
            gp1, gp2, gp3, gp4 = [lines[i + 105 + i_].strip().split()[-1] for i_ in range(4)]
            gp = lines[i + 115].strip().split()[-1]
            gr1, gr2, gr3, gr4 = [lines[i + 131 + i_].strip().split()[-1] for i_ in range(4)]
            gr = lines[i + 141].strip().split()[-1]
            gf1, gf2, gf3, gf4 = [lines[i + 157 + i_].strip().split()[-1] for i_ in range(4)]
            gf = lines[i + 167].strip().split()[-1]
            cor34 = float(r3) * layer_gold_num[2] + float(r4) * layer_gold_num[3]
            gcor34 = float(gr3) * layer_g[2] + float(gr4) * layer_g[3]
            pre3 = 0 if float(p3) == 0 else float(r3) * layer_gold_num[2] / float(p3)
            pre4 = 0 if float(p4) == 0 else float(r4) * layer_gold_num[3] / float(p4)
            pre34 = pre3 + pre4
            gpre3 = 0 if float(gp3) == 0 else float(gr3) * layer_g[2] / float(gp3)
            gpre4 = 0 if float(gp4) == 0 else float(gr4) * layer_g[3] / float(gp4)
            gpre34 = gpre3 + gpre4
            f34, gf34 = 2 * cor34 / (pre34 + sum(layer_gold_num[2:])), 2 * gcor34 / (gpre34 + sum(layer_g[2:]))
            writer.write(','.join([model_name, bs, lr, cw, fold, p, r, f, p1, p2, p3, p4, r1, r2, r3, r4,
                                   f1, f2, f3, f4, gp, gr, gf, gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4,
                                   gf1, gf2, gf3, gf4, str(f34), str(gf34)]) + '\n')
            if int(fold) % 5 == 0:
                writer.write('\n')
    writer.close()


def do_position_job():
    from config import Config
    bs_list = [4, 8][1:]
    lr_list = [1e-5, 5e-6]
    cw_list = [0, 0.1, 0.2, 0.3]
    nest_model = [('BT', FrameWorkTrans)]
    for m_ in range(len(nest_model)):
        for lr_ in range(len(lr_list)):
            for bs_ in range(len(bs_list)):
                for cw_ in range(len(cw_list)):
                    for k in range(1, 6):
                        print('train model = %s | fold = %d | lr = %.6f | cont-weight = %.2f | batch-size = %d' % (
                            nest_model[m_][0], k, lr_list[lr_], cw_list[cw_], bs_list[bs_]))
                        config = Config()
                        config.train_epoch = 30
                        config.cont_weight = cw_list[cw_]
                        config.lr = lr_list[lr_]
                        config.bert_lr = lr_list[lr_]
                        config.batch_size = bs_list[bs_]
                        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
                        config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
                        set_random(config.random_seed)
                        fm = nest_model[m_][1](config)
                        fm.train()


def get_pipeline_loader1(batch_size, word2id=None, max_seq_len=64, data_file=None):
    from torch.utils.data import DataLoader
    from data_utils import NDataSet

    def collate_fn(batch):
        sentence, sen_len, raw_entity, new_entity, element, sid = [], [], [], [], [], []
        b_dict = {}
        for idx, line in enumerate(batch):
            item = json.loads(line)
            t_id = item['sid']
            t_sent = [word2id[w] if w in word2id.keys() else word2id['[UNK]'] for w in item['sentence']]
            t_len = len(item['sentence'])
            t_raw_entity = [(e[0], tuple(e[1])) for e in item['raw_entity']]
            t_new_entity = [(e[0], tuple(e[1])) for e in item['new_entity']]
            t_element = [(e[0], tuple(e[1])) for e in item['element'][len(t_raw_entity):]]
            b_dict[idx] = (t_sent, t_len, t_raw_entity, t_new_entity, t_raw_entity + t_element, t_id)
        items = sorted(b_dict.items(), key=lambda x: x[1][1], reverse=True)
        for k, v in items:
            sentence.append(v[0])
            sen_len.append(v[1])
            raw_entity.append(v[2])
            new_entity.append(v[3])
            element.append(v[4])
            sid.append(v[5])
        batch = {'ids': [t.tensor(s) for s in sentence], 'sent_len': sen_len,
                 'raw_entity': raw_entity, 'new_entity': new_entity, 'element': element, 'sid': sid}
        return batch

    _collate_fn = collate_fn
    data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=batch_size,
                           shuffle=False, collate_fn=_collate_fn, num_workers=0)
    return data_iter


def get_pipeline_loader2(batch_size=4, tokenizer=None, max_seq_len=64, data_file=None):
    from torch.utils.data import DataLoader
    from data_utils import NDataSet

    def convert_entity(words, entities):
        """"""
        converted_entities = []
        c_idx2w_idx = []  # character idx -> word idx
        for i in range(len(words)):
            c_idx2w_idx.extend([i] * len(words[i]))
        for e in entities:
            converted_entities.append((e[0], (c_idx2w_idx[e[1][0] + 5], c_idx2w_idx[e[1][1] + 5])))
        return converted_entities

    def collate_fn(batch):
        sentence, raw_entity, new_entity, element, sid = [], [], [], [], []
        for idx, line in enumerate(batch):
            item = json.loads(line)
            t_id = item['sid']
            t_words = ['[CLS]'] + tokenizer.tokenize(item['sentence'])
            t_word_ids = tokenizer.convert_tokens_to_ids(t_words)[:max_seq_len]
            t_word_ids = t_word_ids + [tokenizer.pad_token_id] * (max_seq_len - len(t_words))
            try:
                t_raw_entity = convert_entity(t_words, item['raw_entity'])
                t_new_entity = convert_entity(t_words, item['new_entity'])
            except:
                print(line)
            t_element = t_raw_entity + [(e[0], tuple(e[1])) for e in item['element'][len(t_raw_entity):]]
            sentence.append(t_word_ids)
            raw_entity.append(t_raw_entity)
            new_entity.append(t_new_entity)
            element.append(t_element)
            sid.append(t_id)
        return {'ids': sentence, 'raw_entity': raw_entity, 'new_entity': new_entity, 'element': element, 'sid': sid}

    _collate_fn = collate_fn
    data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=batch_size,
                           shuffle=False, collate_fn=_collate_fn, num_workers=0)
    return data_iter


def compute_pipeline_metric(all_elements, all_predict, all_raw_entity, all_new_entity):
    def get_entity_rep(predict, entity):
        p2er = [e for e in entity]
        for elem in predict:
            if elem in entity:
                continue
            p2er.append((elem[0], (p2er[elem[1][0]][1], p2er[elem[1][1]][1])))
        return p2er

    def get_correct_num(elements, raw_entity, predict, new_entity):
        gold_rep = get_entity_rep(elements, raw_entity)[len(raw_entity):]
        pred_rep = get_entity_rep(predict, new_entity)[len(new_entity):]
        correct = list(set(gold_rep) & set(pred_rep))
        return len(correct)

    print('compute pipeline metric')
    assert len(all_elements) == len(all_predict) == len(all_raw_entity) == len(all_new_entity)
    correct, predict_num, gold_num = 0, 0, 0
    for i in range(len(all_elements)):
        correct += get_correct_num(all_elements[i], all_raw_entity[i], all_predict[i], all_new_entity[i])
        predict_num += len(all_predict[i])
        gold_num += (len(all_elements[i]) - len(all_raw_entity[i]))
    if not predict_num:
        predict_num = 1
    if not gold_num:
        gold_num = 1
    p = round(correct / predict_num, 5)
    r = round(correct / gold_num, 5)
    f1 = round(2 * p * r / (p + r), 5) if p and r else 0
    return p, r, f1


def compute_auc(all_candidates, all_prob, all_elements):
    from sklearn import metrics
    """"""
    label, prob = [], []
    assert len(all_candidates) == len(all_prob) == len(all_elements)
    # 
    for l in range(6):
        for i_ in range(len(all_candidates)):
            assert len(all_candidates[i_]) == len(all_elements[i_])
            for l_ in range(len(all_candidates[i_][l:l + 1])):
                for j_ in range(len(all_candidates[i_][l + l_])):
                    prob.append(all_prob[i_][l + l_][j_])
                    if all_candidates[i_][l + l_][j_] in all_elements[i_][l + l_]:
                        label.append(1)
                    else:
                        label.append(0)
        print('gold count {} | {} '.format(sum(label), len(label)))
        print('[INFO] {} | {} layers\' auc result: {}'.format(
            datetime.datetime.now(), l + 1, metrics.roc_auc_score(label, prob)))
    
    return metrics.roc_auc_score(label, prob)


def get_pipeline_file():
    from config import Config
    config = Config()
    type_map = {tp.upper(): tp for tp in config.entity_type}
    raw_file = 'data\\test.json'
    new_res_file = 'data\\ner-res\\RTX3090_embedding-ctb_bz32_lstm768_crf_wplinear0.05_ce_fgm_dpr0.5\\policy_22-04-11-10-04-03.txt'
    pipeline_file = 'data\\pipeline-1.json'
    raw_lines = open(raw_file, 'r', encoding='utf-8').readlines()
    new_lines = open(new_res_file, 'r', encoding='utf-8').readlines()
    new_entity = []
    # print(len(new_lines))
    for i_ in range(4, len(new_lines), 4):
        new_entity.append([new_lines[i_ - 2].strip(), eval(new_lines[i_][6:])])
    writer = open(pipeline_file, 'w', encoding='utf-8')
    for i_ in range(len(raw_lines)):
        item = json.loads(raw_lines[i_])
        # print(i_)
        t_new_entity = [(type_map[e[0]], (e[1], e[2])) for e in new_entity[i_][1]]
        writer.write(json.dumps({'sid': item['sid'], 'sentence': item['sentence'],
                                 'raw_entity': item['entity'], 'new_entity': t_new_entity,
                                 'element': item['element']}, ensure_ascii=False) + '\n')
    writer.close()


def do_pipeline_job():
    from config import Config
    
    do_LL = False
    for k in range(1, 6):
        if not do_LL:
            continue
        print('train model = %s | fold = %d | lr = %.6f | cont-weight = %.2f | batch-size = %d' % (
            'LL', k, 1e-5, 0, 4
        ))
        config = Config()
        config.train_epoch = 30
        config.cont_weight = 0
        config.lr, config.bert_lr = 1e-5, 1e-5
        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
        config.model_dir = os.path.join('pipeline', '%d-%d' % (5, k))
        set_random(config.random_seed)
        fm = FrameWorkIM(config)
        fm.train()
        fm.load_model(os.path.join(fm.config.model_dir, 'model.pt'))
    # BL
    do_BL = False
    for k in range(1, 6):
        if not do_BL:
            continue
        print('train model = %s | fold = %d | lr = %.6f | cont-weight = %.2f | batch-size = %d' % (
            'BL', k, 1e-5, 0, 4
        ))
        config = Config()
        config.train_epoch = 30
        config.cont_weight = 0
        config.batch_size = 4
        config.lr, config.bert_lr = 1e-5, 1e-5
        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
        config.model_dir = os.path.join('pipeline', '%d-%d' % (5, k))
        set_random(config.random_seed)
        fm = FrameWorkBLM(config)
        fm.train()
        fm.load_model(os.path.join(fm.config.model_dir, 'model.pt'))
    # BT-c
    do_BT = True
    for k in range(1, 6):
        if not do_BT:
            continue
        print('train model = %s | fold = %d | lr = %.6f | cont-weight = %.2f | batch-size = %d' % (
            'BT', k, 5e-6, 0.2, 4
        ))
        config = Config()
        config.train_epoch = 30
        config.cont_weight = 0.1
        config.batch_size = 4
        config.device = "cuda:1"
        config.lr, config.bert_lr = 5e-6, 5e-6
        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
        config.model_dir = os.path.join('pipeline', '%d-%d' % (5, k))
        set_random(config.random_seed)
        fm = FrameWorkTrans(config)
        fm.train()
        fm.load_model(os.path.join(fm.config.model_dir, 'model.pt'))
        fm.eval_pipeline(os.path.join('data', 'pipeline-1.json'))
        fm.eval_pipeline(os.path.join('data', 'pipeline-2.json'))


if __name__ == '__main__':
    
    my_all_job()
    
