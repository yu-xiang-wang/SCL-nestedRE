# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : fewshot.py
# @Software: PyCharm


import os
from models import IModel, ITransformer, ILTransformer, IBLModel
from data_utils import get_data_loader1, get_data_loader2, load_vocab, get_type_info, generate_raw_weak_data
from data_utils import save_predict, analysis_relation_distribution, generate_ne_entity_replace
from data_utils import generate_ne_entity_replace2, generate_n_way_k_shot
from utils import set_random, compute_metric, calc_running_avg_loss, compute_metric_by_dist, print_dist
from utils import save_config, compute_metric_by_dist_layers, compute_similarity_examples, compute_metric_relation
from utils import compute_metric_relation_guided
import datetime
import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer
import argparse
import copy
import json
import time
import random


class SelfTrainWorkLL(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.word2id, self.id2word = load_vocab(config.vocab_file)
        self.model_dir = os.path.join(config.model_dir, 'FS/LL')
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

        self.model_tea = IModel(vocab_size=config.vocab_size, embedding_size=config.embedding_size,
                                lstm_size=config.hidden_size, dag_size=config.hidden_size * 2,
                                relation_types=len(self.relations)).to(config.device)
        self.model_stu = IModel(vocab_size=config.vocab_size, embedding_size=config.embedding_size,
                                lstm_size=config.hidden_size, dag_size=config.hidden_size * 2,
                                relation_types=len(self.relations)).to(config.device)
        self.teach_step = 0
        generate_raw_weak_data(self.data_dir, ratio=self.config.few_ratio)

    def _setup_train(self):
        params = self.model_stu.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.lr}
        ])

    def save_model(self, teach_step, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model_stu.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model-%d.pt' % teach_step))

    def load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model_stu.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def exchange_model(self):
        self.model_tea = copy.deepcopy(self.model_stu).to('cpu')
        # for param in self.model_tea.parameters():
        #     param.requires_grad = False
        self.model_stu = IModel(vocab_size=self.config.vocab_size, embedding_size=self.config.embedding_size,
                                lstm_size=self.config.hidden_size, dag_size=self.config.hidden_size * 2,
                                relation_types=len(self.relations)).to(self.config.device)
        self._setup_train()

    def initial(self):
        """"""
        raw_train_iter = get_data_loader1(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size, word2id=self.word2id,
            data_file=os.path.join(
                self.data_dir, '%s-%d.json' % (
                    self.config.few_labeled_prefix, int(self.config.few_ratio * self.config.few_scale)))
        )
        batch_len = len(raw_train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                time.sleep(0.003)
                self.model_stu.train()
                self.optimizer.zero_grad()
                # data prepare
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len, entity, element = batch['sent_len'], batch['entity'], batch['element']
                entity_type = [[e[0] for e in se] for se in entity]
                entity_index = [[e[1] for e in se] for se in entity]

                loss = self.model_stu(
                    x=sentence, x_lens=sentence_len, entity_index=entity_index, entity_type=entity_type,
                    elements=element, relations=self.relations, all_type=self.all_type)
                if loss.item():
                    loss.backward()
                else:
                    print(batch['sid'])
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())

                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.teach_step += 1

    def train(self):
        """"""
        raw_train_iter = get_data_loader1(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size, word2id=self.word2id,
            data_file=os.path.join(
                self.data_dir, '%s-%d.json' % (
                    self.config.few_labeled_prefix, int(self.config.few_ratio * self.config.few_scale)))
        )
        raw_len = len(raw_train_iter)
        for st in range(5):
            self.exchange_model()
            running_avg_loss = None
            max_f1, stop_step = -1, 0

            self.generate_weak_file()
            weak_train_iter = get_data_loader1(
                self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size, word2id=self.word2id,
                data_file=os.path.join(
                    self.data_dir, '%s-%d.json' % (
                        self.config.few_tea_prefix, int(self.config.few_ratio * self.config.few_scale)))
            )
            weak_len = len(weak_train_iter)
            for e in range(1, self.config.train_epoch + 1):
                # train on raw train iter
                for idx, batch in enumerate(raw_train_iter):
                    time.sleep(0.003)
                    self.model_stu.train()
                    self.optimizer.zero_grad()
                    # data prepare
                    sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                    sentence_len, entity, element = batch['sent_len'], batch['entity'], batch['element']
                    entity_type = [[e[0] for e in se] for se in entity]
                    entity_index = [[e[1] for e in se] for se in entity]

                    loss = self.model_stu(
                        x=sentence, x_lens=sentence_len, entity_index=entity_index, entity_type=entity_type,
                        elements=element, relations=self.relations, all_type=self.all_type)
                    if loss.item():
                        loss.backward()
                    else:
                        print(batch['sid'])
                    self.optimizer.step()
                    running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                    if (idx + 1) % self.config.p_step == 0:
                        print('[INFO] {} | Epoch : {}/{} | raw process:{}/{} | train_loss : {}'.format(
                            datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                            raw_len, round(running_avg_loss, 5)
                        ))

               
                p, r, f1 = self.eval(op=self.config.valid_prefix)
                if f1 > max_f1:
                    self.save_model(self.teach_step, e, f1)
                    max_f1 = f1
                    stop_step = 0
                else:
                    stop_step += 1
                if stop_step >= self.config.early_stop:
                    print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                    break
            self.teach_step += 1

    def generate_weak_file(self):
        """"""
        weak_file = os.path.join(
            self.data_dir, '%s-%d.json' % (
                self.config.few_unlabeled_prefix, int(self.config.few_ratio * self.config.few_scale)))
        weak_iter = get_data_loader1(
            self.config.valid_prefix, self.data_dir, batch_size=self.config.batch_size, word2id=self.word2id,
            data_file=weak_file)
        all_predict = []
        self.model_tea.eval()
        with t.no_grad():
            for idx, batch in enumerate(weak_iter):
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len, entity, element = batch['sent_len'], batch['entity'], batch['element']
                entity_type = [[e[0] for e in se] for se in entity]
                entity_index = [[e[1] for e in se] for se in entity]
                predict = self.model_tea.predict_scratch(
                    x=sentence, x_lens=sentence_len, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations, all_type=self.all_type
                )
                all_predict.extend(predict)
        examples = [json.loads(line) for line in open(weak_file, 'r', encoding='utf-8').readlines()]
        assert len(examples) == len(all_predict)
        for i in range(len(examples)):
            examples[i]['element'] = examples[i]['entity'] + all_predict[i]
        label_file = os.path.join(
            self.data_dir, '%s-%d.json' % (
                self.config.few_tea_prefix, int(self.config.few_ratio * self.config.few_scale)))
        writer = open(label_file, 'w', encoding='utf-8')
        for e in examples:
            writer.write(json.dumps(e, ensure_ascii=False) + '\n')
        writer.close()

    def eval(self, op='valid'):
        """"""
        valid_iter = get_data_loader1(op, self.data_dir, self.config.batch_size, word2id=self.word2id)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len, entity, element = batch['sent_len'], batch['entity'], batch['element']
                entity_type = [[e[0] for e in se] for se in entity]
                entity_index = [[e[1] for e in se] for se in entity]
                predict = self.model_stu.predict_scratch(
                    x=sentence, x_lens=sentence_len, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations, all_type=self.all_type)
               
                all_predict.extend(predict)
                all_elem.extend(element)
                all_entity.extend(entity)
     
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
      
        return p, r, f1


class SelfTrainWorkBT(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.few_id = int(self.config.few_ratio * self.config.few_scale)
        self.model_dir = os.path.join(config.model_dir, 'FS/BT-%d-%d' % (config.trans_type, self.few_id))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        generate_raw_weak_data(self.data_dir, ratio=config.few_ratio)
        self.teach_step = 0

        self.model_stu = ITransformer(config.bert_dir, config.max_sent_len, config.bert_hidden,
                                      config.bert_head, config.transformer_layer,
                                      self.all_type, config.trans_type).to(config.device)
        self.relation_weight = {r[0]: 1 for r in self.relations}

    def _setup_train(self):
        params = self.model_stu.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])

    def save_model(self, teach_step, epoch, f1, prefix=None):
        state = {
            'epoch': epoch,
            'model': self.model_stu.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(
            self.model_dir, '%s-model-%d.pt' % (prefix, teach_step) if prefix else 'model-%d.pt' % teach_step))

    def load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model_stu.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def exchange_model(self):
        self.model_stu = ITransformer(self.config.bert_dir, self.config.max_sent_len, self.config.bert_hidden,
                                      self.config.bert_head, self.config.transformer_layer,
                                      self.all_type, self.config.trans_type).to(self.config.device)
        self._setup_train()

    def train_weak(self):
        """"""
        print('[INFO] {} | Job Description do weak training of ratio {}'.format(
            datetime.datetime.now(), self.config.few_ratio))

        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=os.path.join(self.data_dir, '%s-%d.json' % (self.config.few_labeled_prefix, self.few_id))
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train()
        # stage 1
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                time.sleep(0.003)
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        p, r, f1, mp, mr, mf = self.eval(op=self.config.test_prefix)
        self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
        self.teach_step += 1

        # stage 2
        stage2 = self.config.do_weak_label
        if stage2:
            print('[INFO] {} | stage2: do weak training'.format(datetime.datetime.now()))
            max_f1, stop_step = -1, 0
            flag2 = False
            self.generate_weak_file()
            weak_train_iter = get_data_loader2(
                self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size // 2,
                tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
                data_file=os.path.join(self.model_dir, '%s-%d.json' % (self.config.few_tea_prefix, self.few_id))
            )
            weak_len = len(weak_train_iter)
            for e in range(1, self.config.train_epoch + 1):
                for idx, batch in enumerate(weak_train_iter):
                    self.model_stu.train()
                    self.optimizer.zero_grad()
                    sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                    mask = sentence.ne(self.tokenizer.pad_token_id)
                    entity_type = [[e[0] for e in se] for se in batch['entity']]
                    entity_index = [[e[1] for e in se] for se in batch['entity']]

                    # loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                    #                       elements=batch['element'], relations=self.relations)
                    loss = self.model_stu.forward_weight(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, relation_weight=self.relation_weight)
                    if loss:
                        loss.backward()
                    else:
                        print(batch['sid'])
                    clip_grad_norm_(self.model_stu.parameters(), 0.5)
                    self.optimizer.step()
                    running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                    if (idx + 1) % (self.config.p_step // self.config.few_ratio) == 0:
                        print('[INFO] {} | Epoch : {}/{} | raw process:{}/{} | train_loss : {}'.format(
                            datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                            weak_len, round(running_avg_loss, 5)
                        ))
                p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
                # self.scheduler.step()
                if f1 > max_f1:
                    flag2 = True
                    self.save_model(self.teach_step, e, f1)
                    max_f1 = f1
                    stop_step = 0
                    assert len(self.relations) + 1 == len(mp[-1])
                    # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
                else:
                    stop_step += 1
                if stop_step >= self.config.early_stop:
                    print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                    break
            if flag2:
                self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
            self.eval(op=self.config.test_prefix)
        self.teach_step += 1

        # stage 3
        print('[INFO] {} | stage3: do fine tune training'.format(datetime.datetime.now()))
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                time.sleep(0.003)
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)
        self.teach_step += 1

    def initial(self):
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=os.path.join(
                self.data_dir, '%s-%d.json' % (
                    self.config.few_labeled_prefix, int(self.config.few_ratio * self.config.few_scale)))
        )
        batch_len = len(raw_train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                time.sleep(0.003)
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break


    def train(self):
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        print('[INFO] {} | teach step = {}'.format(datetime.datetime.now(), self.teach_step))
        self.eval(op=self.config.test_prefix)
        self.teach_step += 1
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=os.path.join(
                self.data_dir, '%s-%d.json' % (
                    self.config.few_labeled_prefix, int(self.config.few_ratio * self.config.few_scale)))
        )
        raw_len = len(raw_train_iter)
        for st in range(5):
            running_avg_loss = None
            max_f1, stop_step = -1, 0

            self.generate_weak_file()
            self.exchange_model()
            weak_train_iter = get_data_loader2(
                self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
                tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
                data_file=os.path.join(
                    self.model_dir, '%s-%d.json' % (
                        self.config.few_tea_prefix, int(self.config.few_ratio * self.config.few_scale)))
            )
            weak_len = len(weak_train_iter)
            for e in range(1, self.config.train_epoch + 1):
                # train on raw train iter
                for idx, batch in enumerate(raw_train_iter):
                    time.sleep(0.003)
                    self.model_stu.train()
                    self.optimizer.zero_grad()
                    sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                    mask = sentence.ne(self.tokenizer.pad_token_id)
                    entity_type = [[e[0] for e in se] for se in batch['entity']]
                    entity_index = [[e[1] for e in se] for se in batch['entity']]

                    loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                          elements=batch['element'], relations=self.relations)
                    if loss:
                        loss.backward()
                    else:
                        print(batch['sid'])
                    clip_grad_norm_(self.model_stu.parameters(), 0.5)
                    self.optimizer.step()
                    running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                    if (idx + 1) % self.config.p_step == 0:
                        print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                            datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                            raw_len, round(running_avg_loss, 5)
                        ))
                # train by teacher model
                for idx, batch in enumerate(weak_train_iter):
                    self.model_stu.train()
                    self.optimizer.zero_grad()
                    sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                    mask = sentence.ne(self.tokenizer.pad_token_id)
                    entity_type = [[e[0] for e in se] for se in batch['entity']]
                    entity_index = [[e[1] for e in se] for se in batch['entity']]

                    loss = self.model_stu.forward_weight(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, relation_weight=self.relation_weight)
                    if loss:
                        loss.backward()
                    else:
                        print(batch['sid'])
                    clip_grad_norm_(self.model_stu.parameters(), 0.5)
                    self.optimizer.step()
                    running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                    if (idx + 1) % (self.config.p_step * 10) == 0:
                        print('[INFO] {} | Epoch : {}/{} | raw process:{}/{} | train_loss : {}'.format(
                            datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                            weak_len, round(running_avg_loss, 5)
                        ))

                p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
                if f1 > max_f1:
                    self.save_model(self.teach_step, e, f1)
                    max_f1 = f1
                    stop_step = 0
                    assert len(self.relations) + 1 == len(mp[-1])
                    self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
                else:
                    stop_step += 1
                if stop_step >= self.config.early_stop:
                    print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                    break
            self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
            print('[INFO] {} | teach step = {}'.format(datetime.datetime.now(), self.teach_step))
            self.eval(op=self.config.test_prefix)
            self.teach_step += 1

    def eval(self, op='valid'):
        valid_iter = get_data_loader2(op, self.data_dir, batch_size=self.config.batch_size,
                                      tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model_stu.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])

        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        mp, mr, mf = compute_metric_by_dist(all_elem, all_predict, all_entity, [r[0] for r in self.relations])
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        return p, r, f1, mp, mr, mf

    def generate_weak_file(self):
        weak_file = os.path.join(self.data_dir, '%s-%d.json' % (self.config.few_unlabeled_prefix, self.few_id))
        weak_iter = get_data_loader2(
            self.config.valid_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=weak_file)
        all_predict = []
        with t.no_grad():
            for idx, batch in enumerate(weak_iter):
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model_stu.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations)
                all_predict.extend(predict)
        examples = [json.loads(line) for line in open(weak_file, 'r', encoding='utf-8').readlines()]
        assert len(examples) == len(all_predict)
        for i in range(len(examples)):
            examples[i]['element'] = examples[i]['entity'] + all_predict[i]
        label_file = os.path.join(self.model_dir, '%s-%d.json' % (self.config.few_tea_prefix, self.few_id))
        writer = open(label_file, 'w', encoding='utf-8')
        for e in examples:
            writer.write(json.dumps(e, ensure_ascii=False) + '\n')
        writer.close()

    def train_by_entity_replace(self):
        print('[INFO] {} | Job Description do entity replace and weak training of ratio {}'.format(
            datetime.datetime.now(), self.config.few_ratio))
        labeled_file = os.path.join(self.config.data_dir, '%s-%d.json' % (self.config.few_labeled_prefix, self.few_id))
        unlabeled_file = os.path.join(
            self.config.data_dir, '%s-%d.json' % (self.config.few_unlabeled_prefix, self.few_id))
        new_file = os.path.join(
            self.config.data_dir,
            '%s%d-%s-%d.json' % (
                self.config.generated_prefix, self.config.generate_type, self.config.few_labeled_prefix, self.few_id))
        if self.config.generate_type == 2:
            generate_ne_entity_replace2(labeled_file, unlabeled_file, new_file)
        else:
            generate_ne_entity_replace(labeled_file, new_file)
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=new_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train()
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1, prefix=self.config.generated_prefix)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(
            os.path.join(self.model_dir, '%s-model-%d.pt' % (self.config.generated_prefix, self.teach_step)))
        self.eval(op=self.config.test_prefix)

    def train_weak_by_entity_replace(self):
        """"""
        print('[INFO] {} | Job Description do entity replace and weak training of ratio {}'.format(
            datetime.datetime.now(), self.config.few_ratio))
        labeled_file = os.path.join(self.config.data_dir, '%s-%d.json' % (self.config.few_labeled_prefix, self.few_id))
        unlabeled_file = os.path.join(
            self.config.data_dir, '%s-%d.json' % (self.config.few_unlabeled_prefix, self.few_id))
        new_file = os.path.join(
            self.config.data_dir,
            '%s%d-%s-%d.json' % (
                self.config.generated_prefix, self.config.generate_type, self.config.few_labeled_prefix, self.few_id))
        if self.config.generate_type == 2:
            generate_ne_entity_replace2(labeled_file, unlabeled_file, new_file)
        else:
            generate_ne_entity_replace(labeled_file, new_file)
        weak_teach_file = os.path.join(self.config.model_dir, '%s-%d.json' % (self.config.few_tea_prefix, self.few_id))
        model_prefix = 'w-%s' % self.config.generated_prefix

        # stage1
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=new_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train()
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                time.sleep(0.003)
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1, prefix=model_prefix)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, '%s-model-%d.pt' % (model_prefix, self.teach_step)))
        p, r, f1, mp, mr, mf = self.eval(op=self.config.test_prefix)
        self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
        self.teach_step += 1

        stage2 = self.config.do_weak_label
        if stage2:
            print('[INFO] {} | stage2: do weak training'.format(datetime.datetime.now()))
            max_f1, stop_step = -1, 0
            flag2 = False
            self.generate_weak_file()
            weak_train_iter = get_data_loader2(
                self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size // 2,
                tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=weak_teach_file
            )
            weak_len = len(weak_train_iter)
            for e in range(1, self.config.train_epoch + 1):
                for idx, batch in enumerate(weak_train_iter):
                    self.model_stu.train()
                    self.optimizer.zero_grad()
                    sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                    mask = sentence.ne(self.tokenizer.pad_token_id)
                    entity_type = [[e[0] for e in se] for se in batch['entity']]
                    entity_index = [[e[1] for e in se] for se in batch['entity']]

                    # loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                    #                       elements=batch['element'], relations=self.relations)
                    loss = self.model_stu.forward_weight(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, relation_weight=self.relation_weight)
                    if loss:
                        loss.backward()
                    else:
                        print(batch['sid'])
                    clip_grad_norm_(self.model_stu.parameters(), 0.5)
                    self.optimizer.step()
                    running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                    if (idx + 1) % (self.config.p_step // self.config.few_ratio) == 0:
                        print('[INFO] {} | Epoch : {}/{} | raw process:{}/{} | train_loss : {}'.format(
                            datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                            weak_len, round(running_avg_loss, 5)
                        ))
                p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
                # self.scheduler.step()
                if f1 > max_f1:
                    flag2 = True
                    self.save_model(self.teach_step, e, f1, prefix=model_prefix)
                    max_f1 = f1
                    stop_step = 0
                    assert len(self.relations) + 1 == len(mp[-1])
                    # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
                else:
                    stop_step += 1
                if stop_step >= self.config.early_stop:
                    print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                    break
            if flag2:
                self.load_model(os.path.join(self.model_dir, '%s-model-%d.pt' % (model_prefix, self.teach_step)))
            self.eval(op=self.config.test_prefix)
        self.teach_step += 1

        # stage 3
        print('[INFO] {} | stage3: do fine tune training'.format(datetime.datetime.now()))
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                time.sleep(0.003)
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1, prefix=model_prefix)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, '%s-model-%d.pt' % (model_prefix, self.teach_step)))
        self.eval(op=self.config.test_prefix)
        self.teach_step += 1


class FewShotBT(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        # self.labels = generate_n_way_k_shot(os.path.join(self.data_dir, '%s.json' % self.config.train_prefix))
        # all_type, relations = get_type_info(config.type_file)
        # re_types = [r[0] for r in relations]
        # self.all_type = list(filter(lambda x: x not in re_types, all_type)) + self.labels
        # self.relations = list(filter(lambda x: x[0] in self.labels, relations))
        self.all_type, self.relations = get_type_info(config.type_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.relations = list(filter(lambda x: x[0] in n_label, self.relations))
        # print(self.relations)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

        self.model_dir = os.path.join(
            config.model_dir, 'FS/N%d-K%d/BT-%d-%d' % (self.config.few_n, self.config.few_k,
                                                       config.trans_type, config.cont_weight * 10))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.teach_step = 0

        self.model_stu = ITransformer(
            config.bert_dir, config.max_sent_len, config.bert_hidden, config.bert_head, config.transformer_layer,
            self.all_type, config.trans_type).to(config.device)
        self.relation_weight = {r[0]: 1 for r in self.relations}

    def _setup_train(self, lr=1e-5):
        params = self.model_stu.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def save_model(self, teach_step, epoch, f1, prefix=None):
        state = {
            'epoch': epoch,
            'model': self.model_stu.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(
            self.model_dir, '%s-model-%d.pt' % (prefix, teach_step) if prefix else 'model-%d.pt' % teach_step))

    def load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model_stu.load_state_dict(state['model'])
        print('load %s model from epoch = %d' % (model_file, state['epoch']))

    def eval(self, op='valid'):
        op_file = os.path.join(
            self.config.data_dir, '%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, op))
        valid_iter = get_data_loader2(
            op, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=op_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model_stu.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations
                )
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r[0] for r in self.relations])
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        return p, r, f1, mp, mr, mf1

    def train(self):
        print('[INFO] {} | Job Description do training of {} way {} shot'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k
        ))
        raw_train_file = os.path.join(
            self.data_dir, '%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=raw_train_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.bert_lr)
        # stage1
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations,
                                      sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)

    def train_weak(self):
        """n way k shot """
        print('[INFO] {} | Job Description do weak training of {} way {} shot'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k
        ))
        raw_train_file = os.path.join(
            self.data_dir, '%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=raw_train_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.lr)
        # stage1
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        p, r, f1, mp, mr, mf = self.eval(op=self.config.test_prefix)
        self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
        self.teach_step += 1

        # stage 2
        stage2 = self.config.do_weak_label
        self._setup_train(self.config.bert_lr)
        if stage2:
            print('[INFO] {} | stage2: do weak training'.format(datetime.datetime.now()))
            max_f1, stop_step = -1, 0
            flag2 = False
            self.generate_weak_file()
            weak_label_file = os.path.join(
                self.model_dir, '%d-way-%d-shot-tea.json' % (self.config.few_n, self.config.few_k))
            weak_train_iter = get_data_loader2(
                self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size // 2,
                tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
                data_file=weak_label_file
            )
            weak_len = len(weak_train_iter)
            for e in range(1, self.config.train_epoch + 1):
                for idx, batch in enumerate(weak_train_iter):
                    self.model_stu.train()
                    self.optimizer.zero_grad()
                    sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                    mask = sentence.ne(self.tokenizer.pad_token_id)
                    entity_type = [[e[0] for e in se] for se in batch['entity']]
                    entity_index = [[e[1] for e in se] for se in batch['entity']]
                    loss = self.model_stu.forward_weight(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, relation_weight=self.relation_weight)
                    if loss:
                        loss.backward()
                    else:
                        print(batch['sid'])
                    clip_grad_norm_(self.model_stu.parameters(), 0.5)
                    self.optimizer.step()
                    running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                    if (idx + 1) % (self.config.p_step // self.config.few_ratio) == 0:
                        print('[INFO] {} | Epoch : {}/{} | raw process:{}/{} | train_loss : {}'.format(
                            datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                            weak_len, round(running_avg_loss, 5)
                        ))
                p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
                if f1 > max_f1:
                    flag2 = True
                    self.save_model(self.teach_step, e, f1)
                    max_f1 = f1
                    stop_step = 0
                    assert len(self.relations) + 1 == len(mp[-1])
                    # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
                else:
                    stop_step += 1
                if stop_step >= self.config.early_stop:
                    print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                    break
            if flag2:
                self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
            self.eval(op=self.config.test_prefix)
        self.teach_step += 1

        # stage 3
        print('[INFO] {} | stage3: do fine tune training'.format(datetime.datetime.now()))
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)

    def generate_weak_file(self):
        weak_file = os.path.join(
            self.data_dir, '%d-way-%d-shot-unlabeled.json' % (self.config.few_n, self.config.few_k))
        weak_iter = get_data_loader2(
            self.config.valid_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=weak_file
        )
        all_predict = []
        with t.no_grad():
            for idx, batch in enumerate(weak_iter):
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model_stu.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations
                )
                all_predict.extend(predict)
        examples = [json.loads(line) for line in open(weak_file, 'r', encoding='utf-8').readlines()]
        assert len(examples) == len(all_predict)
        for i in range(len(examples)):
            examples[i]['element'] = examples[i]['entity'] + all_predict[i]
        label_file = os.path.join(self.model_dir, '%d-way-%d-shot-tea.json' % (self.config.few_n, self.config.few_k))
        writer = open(label_file, 'w', encoding='utf-8')
        for e in examples:
            writer.write(json.dumps(e, ensure_ascii=False) + '\n')
        writer.close()

    def train_by_entity_replace(self):
        print('[INFO] {} | Job Description do entity replace {} way {} shot'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k))
        labeled_file = os.path.join(self.data_dir, '%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        unlabeled_file = os.path.join(
            self.data_dir, '%d-way-%d-shot-unlabeled.json' % (self.config.few_n, self.config.few_k))
        new_file = os.path.join(
            self.data_dir, '%s%d-%d-way-%d-shot.json' % (
                self.config.generated_prefix, self.config.generate_type, self.config.few_n, self.config.few_k))
        if self.config.generate_type == 2:
            generate_ne_entity_replace2(labeled_file, unlabeled_file, new_file)
        else:
            generate_ne_entity_replace(labeled_file, new_file)
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=new_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.lr)
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1, prefix=self.config.generated_prefix)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(
            os.path.join(self.model_dir, '%s-model-%d.pt' % (self.config.generated_prefix, self.teach_step)))
        self.eval(op=self.config.test_prefix)

    def train_weak_by_entity_replace(self):
        print('[INFO] {} | Job Description do entity replace and weak training for {} way {} shot'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k))
        labeled_file = os.path.join(self.data_dir, '%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        unlabeled_file = os.path.join(
            self.data_dir, '%d-way-%d-shot-unlabeled.json' % (self.config.few_n, self.config.few_k))
        new_file = os.path.join(
            self.data_dir, '%s%d-%d-way-%d-shot.json' % (
                self.config.generated_prefix, self.config.generate_type, self.config.few_n, self.config.few_k))
        if self.config.generate_type == 2:
            generate_ne_entity_replace2(labeled_file, unlabeled_file, new_file)
        else:
            generate_ne_entity_replace(labeled_file, new_file)
        model_prefix = 'w-%s' % self.config.generated_prefix
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=new_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.lr)
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1, prefix=model_prefix)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, '%s-model-%d.pt' % (model_prefix, self.teach_step)))
        p, r, f1, mp, mr, mf = self.eval(op=self.config.test_prefix)
        self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
        self.teach_step += 1

        stage2 = self.config.do_weak_label
        self._setup_train(self.config.bert_lr)
        if stage2:
            print('[INFO] {} | stage2: do weak training'.format(datetime.datetime.now()))
            max_f1, stop_step = -1, 0
            flag2 = False
            self.generate_weak_file()
            weak_label_file = os.path.join(
                self.model_dir, '%d-way-%d-shot-tea.json' % (self.config.few_n, self.config.few_k))
            weak_train_iter = get_data_loader2(
                self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size // 2,
                tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=weak_label_file
            )
            weak_len = len(weak_train_iter)
            for e in range(1, self.config.train_epoch + 1):
                for idx, batch in enumerate(weak_train_iter):
                    self.model_stu.train()
                    self.optimizer.zero_grad()
                    sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                    mask = sentence.ne(self.tokenizer.pad_token_id)
                    entity_type = [[e[0] for e in se] for se in batch['entity']]
                    entity_index = [[e[1] for e in se] for se in batch['entity']]

                    loss = self.model_stu.forward_weight(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, relation_weight=self.relation_weight)
                    if loss:
                        loss.backward()
                    else:
                        print(batch['sid'])
                    clip_grad_norm_(self.model_stu.parameters(), 0.5)
                    self.optimizer.step()
                    running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                    if (idx + 1) % (self.config.p_step // self.config.few_ratio) == 0:
                        print('[INFO] {} | Epoch : {}/{} | raw process:{}/{} | train_loss : {}'.format(
                            datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                            weak_len, round(running_avg_loss, 5)
                        ))
                p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
                if f1 > max_f1:
                    flag2 = True
                    self.save_model(self.teach_step, e, f1, prefix=model_prefix)
                    max_f1 = f1
                    stop_step = 0
                    assert len(self.relations) + 1 == len(mp[-1])
                    # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
                else:
                    stop_step += 1
                if stop_step >= self.config.early_stop:
                    print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                    break
            if flag2:
                self.load_model(os.path.join(self.model_dir, '%s-model-%d.pt' % (model_prefix, self.teach_step)))
            self.eval(op=self.config.test_prefix)
        self.teach_step += 1

        # stage 3
        print('[INFO] {} | stage3: do fine tune training'.format(datetime.datetime.now()))
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1, prefix=model_prefix)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
                # self.relation_weight = {r[0]: mp[-1][i] for i, r in enumerate(self.relations)}
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, '%s-model-%d.pt' % (model_prefix, self.teach_step)))
        self.eval(op=self.config.test_prefix)


class FewShotBL(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.relations = list(filter(lambda x: x[0] in n_label, self.relations))
        # print(self.relations)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model_dir = os.path.join(
            config.model_dir, 'FS/N%d-K%d/BL-%d' % (self.config.few_n, self.config.few_k, self.config.cont_weight * 10))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.teach_step = 0

        self.model_stu = IBLModel(config.bert_dir, config.bert_hidden, len(self.relations)).to(config.device)
        self.relation_weight = {r[0]: 1 for r in self.relations}

    def _setup_train(self, lr=1e-5):
        params = self.model_stu.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def save_model(self, teach_step, epoch, f1, prefix=None):
        state = {
            'epoch': epoch,
            'model': self.model_stu.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(
            self.model_dir, '%s-model-%d.pt' % (prefix, teach_step) if prefix else 'model-%d.pt' % teach_step))

    def load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model_stu.load_state_dict(state['model'])
        print('load %s model from epoch = %d' % (model_file, state['epoch']))

    def eval(self, op='valid'):
        op_file = os.path.join(
            self.config.data_dir, '%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, op))
        valid_iter = get_data_loader2(
            op, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=op_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model_stu.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations, all_type=self.all_type
                )
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r[0] for r in self.relations])
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        return p, r, f1, mp, mr, mf1

    def train(self):
        print('[INFO] {} | Job Description do training of {} way {} shot'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k
        ))
        raw_train_file = os.path.join(
            self.data_dir, '%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=raw_train_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.bert_lr)
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations, all_type=self.all_type,
                                      sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)


class FewShotLL(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.word2id, self.id2word = load_vocab(config.vocab_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.relations = list(filter(lambda x: x[0] in n_label, self.relations))
        # print(self.relations)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model_dir = os.path.join(
            config.model_dir, 'FS/N%d-K%d/LL-%d' % (self.config.few_n, self.config.few_k, config.cont_weight * 10))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.teach_step = 0

        self.model_stu = IModel(self.tokenizer.vocab_size, config.embedding_size,
                                config.hidden_size, 2 * config.hidden_size, len(self.relations)).to(config.device)
        self.relation_weight = {r[0]: 1 for r in self.relations}

    def _setup_train(self, lr=1e-5):
        params = self.model_stu.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def save_model(self, teach_step, epoch, f1, prefix=None):
        state = {
            'epoch': epoch,
            'model': self.model_stu.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(
            self.model_dir, '%s-model-%d.pt' % (prefix, teach_step) if prefix else 'model-%d.pt' % teach_step))

    def load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model_stu.load_state_dict(state['model'])
        print('load %s model from epoch = %d' % (model_file, state['epoch']))

    def eval(self, op='valid'):
        op_file = os.path.join(
            self.config.data_dir, '%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, op))
        
        valid_iter = get_data_loader1(op, self.data_dir, self.config.batch_size,
                                      word2id=self.word2id, data_file=op_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_predict, guide_elem = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len = batch['sent_len']
                
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model_stu.predict_scratch(
                    x=sentence, x_lens=sentence_len, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations, all_type=self.all_type)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r[0] for r in self.relations])
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        return p, r, f1, mp, mr, mf1

    def train(self):
        print('[INFO] {} | Job Description do training of {} way {} shot'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k
        ))
        raw_train_file = os.path.join(
            self.data_dir, '%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
       
        raw_train_iter = get_data_loader1(self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
                                          word2id=self.word2id, data_file=raw_train_file)
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.lr)
        # stage1
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len = batch['sent_len']
                
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model_stu(x=sentence, x_lens=sentence_len, entity_index=entity_index,
                                      entity_type=entity_type, elements=batch['element'],
                                      relations=self.relations, all_type=self.all_type,
                                      sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1, mp, mr, mf = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
                assert len(self.relations) + 1 == len(mp[-1])
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)


class FewShotLow(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i + config.few_rel_idx][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.target_relations = list(n_label)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self._set_model()

    def _set_model(self):
        self.model_dir = None
        self.model_stu = None
        pass

    def save_model(self, teach_step, epoch, f1, prefix=None):
        state = {
            'epoch': epoch,
            'model': self.model_stu.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(
            self.model_dir, '%s-model-%d.pt' % (prefix, teach_step) if prefix else 'model-%d.pt' % teach_step))

    def _setup_train(self, lr=1e-5):
        params = self.model_stu.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model_stu.load_state_dict(state['model'])
        print('load %s model from epoch = %d' % (model_file, state['epoch']))

    def train(self):
        pass

    def eval(self, op='valid'):
        pass


class FewShotLowBT(FewShotLow):
    def _set_model(self):
        self.model_dir = os.path.join(
            self.config.model_dir,
            'FS/low-N%d-K%d/BT-%d' % (self.config.few_n, self.config.few_k, self.config.cont_weight * 10))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.teach_step = 0
        self.model_stu = ITransformer(
            self.config.bert_dir, self.config.max_sent_len, self.config.bert_hidden, self.config.bert_head,
            self.config.transformer_layer, self.all_type, self.config.trans_type
        ).to(self.config.device)

    def train(self):
        print('[INFO] {} | Job Description: do model LowBT training of {} way {} shot on low relation'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k
        ))
        raw_train_file = os.path.join(
            self.data_dir, 'low-%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=raw_train_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.bert_lr)
        # stage1
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations,
                                      sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        op_file = os.path.join(
            self.config.data_dir, 'low-%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, op))
        valid_iter = get_data_loader2(
            op, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=op_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                
                predict, gold, _, _ = self.model_stu.predict_guided(
                    x=sentence, x_mask=mask, entity_index=entity_index, elements=batch['element'],
                    entity_type=entity_type, relations=self.relations
                )
                all_predict.extend(predict)
                all_elem.extend(gold)
                all_entity.extend(batch['entity'])
        # p, r, f1 = compute_metric_relation(all_elem, all_predict, all_entity, self.target_relations)
        p, r, f1 = compute_metric_relation_guided(all_elem, all_predict, all_entity, self.target_relations)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        
        return p, r, f1


class FewShotLowBL(FewShotLow):
    def _set_model(self):
        self.model_dir = os.path.join(
            self.config.model_dir,
            'FS/low-N%d-K%d/BL-%d' % (self.config.few_n, self.config.few_k, self.config.cont_weight * 10))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.teach_step = 0
        self.model_stu = IBLModel(
            self.config.bert_dir, self.config.bert_hidden, len(self.relations)).to(self.config.device)

    def train(self):
        print('[INFO] {} | Job Description: do model LowBL training of {} way {} shot on low relation'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k
        ))
        raw_train_file = os.path.join(
            self.data_dir, 'low-%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        raw_train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
            data_file=raw_train_file
        )
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.bert_lr)
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model_stu(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                      elements=batch['element'], relations=self.relations, all_type=self.all_type,
                                      sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        op_file = os.path.join(
            self.config.data_dir, 'low-%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, op))
        valid_iter = get_data_loader2(
            op, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=op_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict, gold, _, _ = self.model_stu.predict_guided(
                    x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                    elements=batch['element'], relations=self.relations, all_type=self.all_type
                )
                all_predict.extend(predict)
                all_elem.extend(gold)
                all_entity.extend(batch['entity'])
        p, r, f1 = compute_metric_relation_guided(all_elem, all_predict, all_entity, self.target_relations)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class FewShotLowLL(FewShotLow):
    def _set_model(self):
        self.model_dir = os.path.join(
            self.config.model_dir,
            'FS/low-N%d-K%d/LL-%d' % (self.config.few_n, self.config.few_k, self.config.cont_weight * 10))
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.word2id, self.id2word = load_vocab(self.config.vocab_file)
        self.teach_step = 0
        self.model_stu = IModel(
            self.tokenizer.vocab_size, self.config.hidden_size,
            self.config.hidden_size, 2 * self.config.hidden_size, len(self.relations)
        ).to(self.config.device)

    def train(self):
        print('[INFO] {} | Job Description: do model LowLL training of {} way {} shot on low relation'.format(
            datetime.datetime.now(), self.config.few_n, self.config.few_k
        ))
        raw_train_file = os.path.join(
            self.data_dir, 'low-%d-way-%d-shot-train.json' % (self.config.few_n, self.config.few_k))
        # raw_train_iter = get_data_loader2(
        #     self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
        #     tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len,
        #     data_file=raw_train_file
        # )
        raw_train_iter = get_data_loader1(self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
                                          word2id=self.word2id, data_file=raw_train_file)
        raw_batch_len = len(raw_train_iter)
        self._setup_train(self.config.lr)
        # stage1
        print('[INFO] {} | stage1: do supervised training'.format(datetime.datetime.now()))
        print('*' * 20)
        self.teach_step += 1
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(raw_train_iter):
                self.model_stu.train()
                self.optimizer.zero_grad()
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len = batch['sent_len']
                # sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                # sentence_len = sentence.ne(self.tokenizer.pad_token_id).sum(-1).cpu()
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                loss = self.model_stu(x=sentence, x_lens=sentence_len, entity_index=entity_index,
                                      entity_type=entity_type, elements=batch['element'],
                                      relations=self.relations, all_type=self.all_type,
                                      sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)
                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model_stu.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        raw_batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(self.teach_step, e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self.load_model(os.path.join(self.model_dir, 'model-%d.pt' % self.teach_step))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        op_file = os.path.join(
            self.config.data_dir, 'low-%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, op))
        
        valid_iter = get_data_loader1(op, self.data_dir, self.config.batch_size,
                                      word2id=self.word2id, data_file=op_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model_stu.eval()
        all_predict, all_elem, all_entity = [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                sentence = pad_sequence(batch['ids'], batch_first=True, padding_value=0).to(self.config.device)
                sentence_len = batch['sent_len']
               
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict, gold, _, _ = self.model_stu.predict_guided(
                    x=sentence, x_lens=sentence_len, entity_index=entity_index, entity_type=entity_type,
                    elements=batch['element'], relations=self.relations, all_type=self.all_type
                )
                all_predict.extend(predict)
                all_elem.extend(gold)
                all_entity.extend(batch['entity'])
        p, r, f1 = compute_metric_relation_guided(all_elem, all_predict, all_entity, self.target_relations)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


def my_job():
    from config import Config
    config = Config()
    set_random(config.random_seed)
    
    few_ratio = [0.1, 0.2, 0.3, 0.5]
    for r in few_ratio:
        config = Config()
        set_random(config.random_seed)
        config.few_ratio = r
        model = SelfTrainWorkBT(config)
        
        model.train_weak_by_entity_replace()


def my_nk_job():
    from config import Config
    few_k = [1, 5, 10][:]
    model_dict = {'BT': FewShotBT, 'BL': FewShotBL, 'LL': FewShotLL,
                  'BTLow': FewShotLowBT, 'BLLow': FewShotLowBL, 'LLLow': FewShotLowLL}
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model_name', type=str, default='all', help='Model Type')
    args = parser.parse_args()
    model_names = args.model_name.split('-') if args.model_name != 'all' else list(model_dict.keys())
    lr_list = [1e-4, 5e-5, 1e-5, 5e-6]
    for k in few_k:
        for name in model_names:
            for lr in lr_list:
                config = Config()
                config.lr, config.bert_lr = lr, lr
                config.few_k = k
                config.p_step = 10
                set_random(config.random_seed)
                print('train %s with lr = %f for %d way % shot' % (name, lr, config.few_n, config.few_k))
                model = model_dict[name](config)
                model.train()
            
    for cw in [0.1, 0.2, 0.3]:
        for k in few_k:
            for name in ['BT', 'BTLow']:
                for lr in lr_list:
                    config = Config()
                    config.lr, config.bert_lr = lr, lr
                    config.few_k = k
                    config.p_step = 10
                    config.cont_weight = cw
                    set_random(config.random_seed)
                    print('train contrastive %d %s with lr = %f for %d way % shot' % (
                        int(cw * 10), name, lr, config.few_n, config.few_k))
                    model = model_dict[name](config)
                    model.train()


def my_all_job():
    from config import Config
    high_model = [('INN', FewShotLL), ('INN-BERT', FewShotBL)]
    low_model = [('INN', FewShotLowLL), ('INN-BERT', FewShotLowBL), ('BERT+Transformer', FewShotLowBT), ('SCL-nestedRE', FewShotLowBT)]
    bs = [2, 4, 8]
    lr = [1e-4, 5e-5, 1e-5, 5e-6]
    few_k = [5, 10]
    random_seeds = [66, 111, 1234, 999, 6789]

    # 
    for model_idx in range(len(high_model)):
        for bs_idx in range(len(bs)):
            for lr_idx in range(len(lr)):
                for k_idx in range(len(few_k)):
                    for random_seed in random_seeds:
                        continue
                        # if model_idx == 0:
                        #     continue
                        config = Config()

                        config.random_seed = random_seed
                        config.device = "cuda:1"

                        print('Model Introduction: %s high relation 5 way %d shot, bs = %d, lr = %.6f, random = %d' % (
                            high_model[model_idx][0], few_k[k_idx], bs[bs_idx], lr[lr_idx], random_seed))
                        config.random_seed = 1234
                        config.cont_weight, config.lr, config.bert_lr = 0, lr[lr_idx], lr[lr_idx]
                        config.batch_size, config.few_k = bs[bs_idx], few_k[k_idx]
                        config.p_step = 10
                        set_random(config.random_seed)
                        model = high_model[model_idx][1](config)
                        model.train()

    # 
    for model_idx in range(len(low_model)):
        for bs_idx in range(len(bs)):
            for lr_idx in range(len(lr)):
                for k_idx in range(len(few_k)):
                    for random_seed in random_seeds:
                        # continue
                        config = Config()

                        config.random_seed = random_seed
                        config.device = "cuda:0"

                        print('Model Introduction: %s low relation 5 way %d shot, bs = %d, lr = %.6f, random = %d' % (
                            low_model[model_idx][0], few_k[k_idx], bs[bs_idx], lr[lr_idx], random_seed))
                        config.cont_weight, config.lr, config.bert_lr = 0, lr[lr_idx], lr[lr_idx]
                        if low_model[model_idx][0] == 'SCL-nestedRE':
                            config.cont_weight = 0.1
                        config.batch_size, config.few_k = bs[bs_idx], few_k[k_idx]
                        config.p_step = 10
                        set_random(config.random_seed)
                        model = low_model[model_idx][1](config)
                        model.train()


if __name__ == '__main__':
    my_all_job()
