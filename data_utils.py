# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : data_utils.py
# @Software: PyCharm


from torch.utils.data import Dataset, DataLoader
import os
import json
from config import Config
import torch as t
import random
import numpy as np
from utils import str_q2b, set_random


config = Config()


def prepare_data():

    files = os.listdir(config.labeled_dir) 
    examples = []  
    for file in files:
        line = open(os.path.join(config.labeled_dir, file), 'r', encoding='utf-8').readlines()[0]
        examples.extend([e['answer'] for e in json.loads(line)])
    print(len(examples))
    print(examples[0])
    all_sent = [''.join(w['word'] for w in e['words']) for e in examples]
    all_sid = [e['info']['sid'] for e in examples]
    print(len(set(all_sent)), len(set(all_sid)))

    all_words, all_entity, all_relation = [], [], []
    max_wl = 0
    for i, e in enumerate(examples):
        word = [w['word'] for w in e['words']]
        max_wl = max(max_wl, len(word))
        all_words.extend(word)
        all_entity.extend([r['type'] for r in e['entities']])
        all_relation.extend(en['type'] for en in e['relations'])
    print(len(set(all_words)), '\n', 'entity:', set(all_entity),
          '\n', 'relation:', set(all_relation), '\n', 'max sentence len = ', max_wl)
    # 2
    all_words = list(set(all_words))
    new_vocab_file = os.path.join(config.data_dir, 'vocab.txt')
    v_writer = open(new_vocab_file, 'w', encoding='utf-8')
    v_writer.write('\n'.join(all_words))
    v_writer.close()

    # 3
    new_sys_file = os.path.join(config.data_dir, 'new-sys-labeled.json')
    new_examples = [json.loads(x) for x in open(new_sys_file, 'r', encoding='utf-8').readlines()
                    ] if os.path.exists(new_sys_file) else []
    for i in range(len(new_examples)):
        new_examples[i]['entity'] = [(change_type(ent[0]), ent[1]) for ent in new_examples[i]['entity']]
    sids = [e['sid'] for e in new_examples]
    text = set([str_q2b(e['sentence']) for e in new_examples])
    print(len(text))
    # new_examples = []
    # error_count = 0
    lens = []
    null_count, re_count = 0, 0
    for i, e in enumerate(examples):
        word = [w['word'] for w in e['words']]
        sentence = str_q2b(''.join(word))
        # print(sentence)
        entity = [(en['id'], change_type(en['type']), ''.join([word[int(w[1:])] for w in en['tokens']]),
                  [int(en['tokens'][0][1:]), int(en['tokens'][-1][1:])])
                  for en in e['entities']]  # (id, type, words, words_index)
        relation = [(r['id'], r['type'], r['operands']) for r in e['relations']]  # (id, type, ops)
        element_id2index = {elem[0]: i for i, elem in enumerate(entity+relation)}  # map id to idx

        ne = dict()
        ne['sid'] = e['info']['sid']
        if sentence in text:
            re_count += 1
            continue
        # ne['ids'] = []
        n_en = [(en[1], tuple(en[3])) for en in entity]  # entity: (type, (start, end))
        ne['entity'] = n_en
        n_re = [(r[1], tuple(element_id2index[elem] for elem in r[2])) for r in relation]  # relation: (type, (op1, op2)
        if not len(n_re):  
            null_count += 1
            continue
        ne['element'] = n_en + n_re
        sids.append(ne['sid'])
        ne['sentence'] = sentence
        lens.append(len(sentence))
        text.add(sentence)
         
        new_examples.append(ne)
    print('null count = ', null_count, 'repeat count = ', re_count)
    # 4
    set_random(config.random_seed)
    print('shuffle the data set')
    random.shuffle(new_examples)
    print('processed examples = ', len(new_examples))
    print('max sentence length = ', max(lens))
    # fold = [8, 1, 1]
    train_file = os.path.join(config.data_dir, 'train.json')
    valid_file = os.path.join(config.data_dir, 'valid.json')
    test_file = os.path.join(config.data_dir, 'test.json')
    # 
    test_len = len(new_examples) 
    test_e = new_examples[-test_len:]
    train_len = ((len(new_examples)-test_len)
    train_e, valid_e = new_examples[:train_len], new_examples[train_len:-test_len]
    # split_len = int(round(len(new_examples) / sum(fold)))
    # test_e = new_examples[sum(fold[0:2])*split_len:]
    for f, e in zip([train_file, valid_file, test_file], [train_e, valid_e, test_e]):
        writer = open(f, 'w', encoding='utf-8')
        for ne in e:
            writer.write(json.dumps(ne, ensure_ascii=False)+'\n')
        writer.close()


def get_error_data():
    """"""
    # 1
    files = os.listdir(config.labeled_dir)  
    examples = [] 
    for file in files:
        line = open(os.path.join(config.labeled_dir, file), 'r', encoding='utf-8').readlines()[0]
        examples.extend([e['answer'] for e in json.loads(line)])

    new_examples, abnormal_examples = [], []
    sids = set()
    error_count = 0
    lens = []
    text = set()
    # all_type, relations = get_type_info(config.type_file)
    # re_type = [r[0] for r in relations]
    for i, e in enumerate(examples):
        word = [w['word'] for w in e['words']]
        sentence = str_q2b(''.join(word))
        # print(sentence)
        entity = [(en['id'], change_type(en['type']), ''.join([word[int(w[1:])] for w in en['tokens']]),
                   [int(en['tokens'][0][1:]), int(en['tokens'][-1][1:])])
                  for en in e['entities']]  # (id, type, words, words_index)
        relation = [(r['id'], r['type'], r['operands']) for r in e['relations']]  # (id, type, ops)
        element_id2index = {elem[0]: i for i, elem in enumerate(entity + relation)}  # map id to idx

        ne = dict()
        ne['sid'] = e['info']['sid']

        ne['sentence'] = sentence
        lens.append(len(sentence))
        # ne['ids'] = []
        n_en = [(en[1], tuple(en[3])) for en in entity]  # entity: (type, (start, end))
        ne['entity'] = n_en
        n_re = [(r[1], tuple(element_id2index[elem] for elem in r[2])) for r in relation]  # relation: (type, (op1, op2)
        if not len(n_re):  
            sids.add(ne['sid'])
            continue
        ne['element'] = n_en + n_re
        # ne, flag = parse_example(ne)
        flag = error_data(ne)
        if flag or ne['sid'] in sids or ne['sentence'] in text:
            # print(ne['sid'])
            sids.add(ne['sid'])
            text.add(ne['sentence'])
            abnormal_examples.append(ne)
            error_count += 1
            continue  
        sids.add(ne['sid'])
        text.add(ne['sentence'])
        # for r in ne['element']:
        #     if r[0] == '@' and ne['element'][r[1][1]][0] in re_type and ne['element'][r[1][1]][0] != '@':
        #         print(ne['sid'])
        new_examples.append(ne)
    # 
    raw_files = [os.path.join('../JNRE/data/split', 'sp%d.json' % i) for i in range(1, 16)]
    for f in raw_files:
        raw_ex = json.loads(open(f, 'r', encoding='utf-8').readlines()[0])
        for e in raw_ex:
            if e['info']['sid'] not in sids:
                error_count += 1
                abnormal_examples.append({
                    'sid': e['info']['sid'], 'sentence': str_q2b(''.join([w['word'] for w in e['words']])),
                    'entity': [], 'element': []
                })

    print('abnormal examples = ', error_count, len(new_examples))
    ab_file = os.path.join(config.data_dir, 'abnormal.json')
    writer = open(ab_file, 'w', encoding='utf-8')
    for e in abnormal_examples:
        writer.write(json.dumps(e, ensure_ascii=False)+'\n')
    writer.close()


def convert_new_example(example):
    """"""
    ne = {'sid': example['sid'], 'sentence': example['sentence'], 'entity': example['entity']}
    ne_element = example['element'][:len(example['entity'])]
    for elem in example['element'][len(example['entity']):]:
        if elem[0] == '@':
            right = ne_element[elem[1][1]][0]
            if '@' in right:
                ne_element.append([right, elem[1]])
            else:
                ne_element.append(['@%s' % right, elem[1]])
        else:
            ne_element.append(elem)
    ne['element'] = ne_element
    return ne


def error_data(example):
    """"""
    flag = False
    element = example['element']

    for i in range(len(element)):  # (type, (idx1, idx2)) | (type, (op1, op2))
        if element[i][0] == '@' and element[element[i][1][0]][0] == '@' and element[element[i][1][1]][0] not in config.relation_type:
            # element[i] = (element[i][0], (element[i][1][1], element[i][1][0]))
            flag = True
    # 
    key_words = ['达到', '达', '满']
    for w in key_words:
        if w in example['sentence']:
            flag = True
            break
    return flag


def generate_k_fold_data(fold):
    """train: valid: test = 2*k-2: 1 : 1"""
    # 1
    files = os.listdir(config.labeled_dir)  
    examples = [] 
    for file in files:
        line = open(os.path.join(config.labeled_dir, file), 'r', encoding='utf-8').readlines()[0]
        examples.extend([e['answer'] for e in json.loads(line)])
    # 3
    new_sys_file = os.path.join(config.data_dir, 'new-sys-labeled.json')
    new_examples = [json.loads(x) for x in open(new_sys_file, 'r', encoding='utf-8').readlines()
                    ] if os.path.exists(new_sys_file) else []
    sids = [e['sid'] for e in new_examples]
    text = set([e['sentence'] for e in new_examples])
    # error_count = 0
    for i, e in enumerate(examples):
        word = [w['word'] for w in e['words']]
        sentence = str_q2b(''.join(word))
        # print(sentence)
        entity = [(en['id'], change_type(en['type']), ''.join([word[int(w[1:])] for w in en['tokens']]),
                   [int(en['tokens'][0][1:]), int(en['tokens'][-1][1:])])
                  for en in e['entities']]  # (id, type, words, words_index)
        relation = [(r['id'], r['type'], r['operands']) for r in e['relations']]  # (id, type, ops)
        element_id2index = {elem[0]: i for i, elem in enumerate(entity + relation)}  # map id to idx

        ne = dict()
        ne['sid'] = e['info']['sid']
        if ne['sid'] in sids or sentence in text:
            continue
        sids.append(ne['sid'])
        ne['sentence'] = sentence
        # ne['ids'] = []
        n_en = [(en[1], tuple(en[3])) for en in entity]  # entity: (type, (start, end))
        ne['entity'] = n_en
        n_re = [(r[1], tuple(element_id2index[elem] for elem in r[2])) for r in
                relation]  
        if not len(n_re):  
            continue
        ne['element'] = n_en + n_re
        # ne, flag = parse_example(ne)
        # if flag:
        #     print(ne['sid'])
        #     error_count += 1
        #     continue  
        text.add(sentence)
        new_examples.append(ne)
    print('all examples = %d' % len(new_examples))
    set_random(config.random_seed)
    print('shuffle the data set')
    random.shuffle(new_examples)
    split_len = int(round(len(new_examples) / (2*fold)))
    for k in range(fold):
        data_dir = os.path.join(config.data_dir, '%d-%d' % (fold, k+1))
        os.makedirs(data_dir, exist_ok=True)
        valid_start, test_start, test_end = 2*k*split_len, (2*k+1)*split_len, 2*(k+1)*split_len
        train_e = new_examples[0:valid_start]+new_examples[test_end:]
        valid_e = new_examples[valid_start:test_start]
        test_e = new_examples[test_start:test_end]
        train_file, valid_file, test_file = [os.path.join(data_dir, '%s.json' % m) for m in ['train', 'valid', 'test']]
        for f, e in zip([train_file, valid_file, test_file], [train_e, valid_e, test_e]):
            writer = open(f, 'w', encoding='utf-8')
            for ne in e:
                writer.write(json.dumps(ne, ensure_ascii=False) + '\n')
            writer.close()


def generate_k_fold_data2(fold):
    """"""
    # 
    train_f, valid_f = os.path.join(config.data_dir, 'train.json'), os.path.join(config.data_dir, 'valid.json')
    examples = [json.loads(x) for x in open(train_f, 'r', encoding='utf-8').readlines()
                ] + [json.loads(x) for x in open(valid_f, 'r', encoding='utf-8').readlines()]
    split_len = len(examples) // fold + 1
    test_examples = [
        json.loads(x) for x in open(os.path.join(config.data_dir, 'test.json'), 'r', encoding='utf-8').readlines()]
    for k in range(fold):
        data_dir = os.path.join(config.data_dir, '%d-%d' % (fold, k+1))
        os.makedirs(data_dir, exist_ok=True)
        valid_start, valid_end = k*split_len, (k+1)*split_len
        train_e = examples[0:valid_start]+examples[valid_end:]
        valid_e = examples[valid_start:valid_end]
        train_f, valid_f, test_f = [os.path.join(data_dir, '%s.json' % m) for m in ['train', 'valid', 'test']]
        for f, e in zip([train_f, valid_f, test_f], [train_e, valid_e, test_examples]):
            writer = open(f, 'w', encoding='utf-8')
            for ne in e:
                writer.write(json.dumps(ne, ensure_ascii=False) + '\n')
            writer.close()


def change_type(t_name):
    if t_name == 'Other':
        return 'Base'
    elif t_name == 'OtherV':
        return 'BaseV'
    return t_name


def generate_type():
    file = os.path.join(config.data_dir, 'type.json')
    entity_type = ['Main', 'Main-q', 'Labor', 'Labor-q', 'Service', 'Place',
                   'Rate', 'RateV', 'Fund', 'FundV', 'Time', 'TimeV', 'Base', 'BaseV']
    relation_type = [['>', [[['Fund', '@', '+', '-'], ['Fund', '@', 'FundV', 'RateV']],
                            [['Rate', '@', '+', '-', '/'], ['Rate', '@', 'RateV']],
                            [['Labor', '@', '+', '-'], ['Labor', '@', 'BaseV', 'RateV']],
                            [['Time', '@', '+', '-'], ['Time', '@', 'TimeV']],
                            [['Base', '@', '+', '-'], ['Base', '@', 'BaseV', 'RateV']]]],
                     ['+', [[['Fund', '@', '+', '-'], ['Fund', '@']],
                            [['Labor', '@', '+', '-'], ['Labor', '@']],
                            [['Base', '@', '+', '-'], ['Base', '@']],
                            [['Time', '@', '+', '-'], ['Time', '@']],
                            [['Rate', '@', '+', '-'], ['Rate', '@']],
                            [['Main', '@', '+', '-'], ['Main', '@']],
                            [['Service', '@', '+', '-'], ['Service', '@']],
                            [['Main', '@', '+', '-'], ['Main', '@']]]],
                     ['has', [[['Main', '@'], ['Main-q', 'Labor', 'Service', 'Base']],
                              [['Labor', '@'], ['Labor-q', 'Base']],
                              [['Service', '@'], ['Base']],
                              [['Base', '@'], ['Base']]]],
                     ['and', [[['>', '>=', 'and'], ['<', '<=']]]],
                     ['@', [[['Main-q', 'Place', 'TimeV', 'Time', '>', '>=', '<', '<=', '=', 'and'], ['Main', '@']],
                            [['Labor-q', 'Place', 'TimeV', 'Time', 'Main', 'Service', 'Fund', '>', '>=', '<', '<=', '=', 'and'], ['Labor', '@']],
                            [['Main', 'Place', 'TimeV', 'Time', '>', '>=', '<', '<=', '=', 'and'], ['Service', '@']],
                            [['Main', 'Service', 'Place', 'Base'], ['Time', '@']],
                            [['Main', 'Place', 'TimeV', 'Time', 'Labor', 'Service', 'Base', '>', '>=', '<', '<=', '=', 'and'], ['Base', '@']],
                            [['Main', 'Service', 'Labor', 'Base', 'Place', 'TimeV', 'Time', '>', '>=', '<', '<=', '=', 'and'], ['Fund', '@']],
                            [['Main', 'Service', 'Labor', 'Base', 'Place', 'TimeV', 'Time', 'Fund', '>', '>=', '<', '<=', '=', 'and'], ['Rate', '@']]]]
                     ]
    type_data = {'entity': entity_type,
                 'relation': [['>', relation_type[0][1]],
                              ['>=', relation_type[0][1]],
                              ['<', relation_type[0][1]],
                              ['<=', relation_type[0][1]],
                              ['=', relation_type[0][1]],
                              ['+', relation_type[1][1]],
                              ['-', relation_type[1][1]],
                              ['/', relation_type[1][1]],
                              ['has', relation_type[2][1]],
                              ['and', relation_type[3][1]],
                              ['@', relation_type[4][1]]]}
    writer = open(file, 'w', encoding='utf-8')
    writer.write(json.dumps(type_data, ensure_ascii=False) + '\n')
    writer.close()


def generate_type2():
    file = os.path.join(config.data_dir, 'type2.json')
    entity_type = ['Main', 'Main-q', 'Labor', 'Labor-q', 'Service', 'Place',
                   'Rate', 'RateV', 'Fund', 'FundV', 'Time', 'TimeV', 'Base', 'BaseV']
    relation_type = [['>', [[['Fund', '@Fund', '+', '-'], ['Fund', '@Fund', 'FundV', 'RateV']],
                            [['Rate', '@Rate', '+', '-', '/'], ['Rate', '@Rate', 'RateV']],
                            [['Labor', '@Labor', '+', '-'], ['Labor', '@Labor', 'BaseV']],
                            [['Time', '@Time', '+', '-'], ['Time', '@Time', 'TimeV']],
                            [['Base', '@Base', '+', '-'], ['Base', '@Base', 'BaseV', 'RateV']]]],
                     ['+', [[['Fund', '@Fund', '+', '-'], ['Fund', '@Fund']],
                            [['Labor', '@Labor', '+', '-'], ['Labor', '@Labor']],
                            [['Base', '@Base', '+', '-'], ['Base', '@Base']],
                            [['Time', '@Time', '+', '-'], ['Time', '@Time']],
                            [['Rate', '@Rate', '+', '-'], ['Rate', '@Rate']],
                            [['Service', '@Service', '+', '-'], ['Service', '@Service']],
                            [['Main', '@Main', '+', '-'], ['Main', '@Main']]]],
                     ['has', [[['Main', '@Main'], ['Main-q', 'Labor', 'Service', 'Base']],
                              [['Labor', '@Labor'], ['Labor-q', 'Base']],
                              [['Service', '@Service'], ['Base']],
                              [['Base', '@Base'], ['Base']]]],
                     ['and', [[['>', '>=', 'and'], ['<', '<=']]]],
                     ['@Main', [
                         [['Main-q', 'Place', 'TimeV', 'Time', '>', '>=', '<', '<=', '=', 'and'],
                          ['Main', '@Main']]]],
                     ['@Labor', [
                         [['Labor-q', 'Place', 'TimeV', 'Time', 'Fund', '@Fund', 'Main', '@Main', 'Service', '@Service', '>', '>=', '<', '<=', '=', 'and'],
                          ['Labor', '@Labor']]]],
                     ['@Service', [
                         [['Main', '@Main', 'Place', 'TimeV', 'Time', '>', '>=', '<', '<=', '=', 'and'],
                          ['Service', '@Service']]]],
                     ['@Time', [
                         [['Main', '@Main', 'Place', 'Service', '@Service', 'Base', '@Base'],
                          ['Time', '@Time']]]],
                     ['@Base', [
                         [['Main', '@Main', 'Place', 'TimeV', 'Time', 'Labor', '@Labor', 'Service', '@Service', 'Base', '@Base', '>', '>=', '<', '<=', '=', 'and'],
                          ['Base', '@Base']]]],
                     ['@Fund', [
                         [['Main', '@Main', 'Service', '@Service', 'Labor', '@Labor', 'Base', '@Base', 'Place', 'TimeV', 'Time', '>', '>=', '<', '<=', '=','and'],
                          ['Fund', '@Fund']]]],
                     ['@Rate', [
                         [['Main', '@Main', 'Service', '@Service', 'Labor', '@Labor', 'Base', '@Base', 'Place', 'TimeV', 'Time', 'Fund', '@Fund', '>', '>=', '<', '<=', '=', '+', '-', 'and'],
                          ['Rate', '@Rate']]]]
                     ]
    type_data = {'entity': entity_type,
                 'relation': [['>', relation_type[0][1]],
                              ['>=', relation_type[0][1]],
                              ['<', relation_type[0][1]],
                              ['<=', relation_type[0][1]],
                              ['=', relation_type[0][1]],
                              ['+', relation_type[1][1]],
                              ['-', relation_type[1][1]],
                              ['/', relation_type[1][1]],
                              ['has', relation_type[2][1]],
                              ['and', relation_type[3][1]],
                              ['@Main', relation_type[4][1]],
                              ['@Labor', relation_type[5][1]],
                              ['@Service', relation_type[6][1]],
                              ['@Time', relation_type[7][1]],
                              ['@Base', relation_type[8][1]],
                              ['@Fund', relation_type[9][1]],
                              ['@Rate', relation_type[10][1]]]}
    writer = open(file, 'w', encoding='utf-8')
    writer.write(json.dumps(type_data, ensure_ascii=False) + '\n')
    writer.close()


def parse_example(example):
    """"""
    element = example['element']
    changed = False
    # 
    for i in range(len(element)):  # (type, (idx1, idx2)) | (type, (op1, op2))
        if element[i][0] == '@' and element[element[i][1][0]][0] == '@' and element[element[i][1][1]][0] not in config.relation_type:
            element[i] = (element[i][0], (element[i][1][1], element[i][1][0]))
            changed = True
    example['element'] = element
    return example, changed


def load_vocab(file):
    """"""
    lines = open(file, 'r', encoding='utf-8').readlines()
    words = config.extra_words + [w.strip() for w in lines]
    word2id = {w: i for i, w in enumerate(words)}
    return word2id, words


def get_type_info(file):
    line = open(file, 'r', encoding='utf-8').readlines()[0]
    content = json.loads(line)
    entity, relation = content['entity'], content['relation']
    all_type = entity + [r[0] for r in relation]
    # relations = [(r[0], ) for r in relation]
    return all_type, relation


class NDataSet(Dataset):
    def __init__(self, file, max_line=1000000):
        super(NDataSet, self).__init__()
        self.data = []
        with open(file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                self.data.append(line)
                if idx >= max_line:
                    break

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_data_loader1(op='train', data_dir='', batch_size=4, word2id=None, data_file=None, max_seq_len=64):
    def collate_fn(batch):
        """
        generate data from sentence of batch
        :param batch: sentences for json
        :return: data for model
        """
        sentence, sen_len, entity, element, sid = [], [], [], [], []
        b_dict = {}
        for idx, line in enumerate(batch):
            item = json.loads(line)
            # rl = len(item['entity'])
            # item['entity'] = sorted(list(set([(e[0], tuple(e[1])) for e in item['entity']])))
            # item['element'] = item['entity'] + list(set([(el[0], tuple(el[1])) for el in item['element'][rl:]]))
            t_id = item['sid']
            t_sent = [word2id[w] if w in word2id.keys() else word2id['[UNK]'] for w in item['sentence']]
            # t_sent = t_sent[:max_seq_len] + [word2id['[PAD]']] * (max_seq_len-len(t_sent))
            t_len = len(item['sentence'])
            t_entity = [(e[0], tuple(e[1])) for e in item['entity']]
            t_element = [(e[0], tuple(e[1])) for e in item['element'][len(t_entity):]]
            b_dict[idx] = (t_sent, t_len, t_entity, t_entity+t_element, t_id)
        items = sorted(b_dict.items(), key=lambda x: x[1][1], reverse=True)
        for k, v in items:
            sentence.append(v[0])
            sen_len.append(v[1])
            entity.append(v[2])
            element.append(v[3])
            sid.append(v[4])
        batch = {
            'ids': [t.tensor(s) for s in sentence],
            # 'ids': sentence,
            'sent_len': sen_len,
            'entity': entity,
            'element': element,
            'sid': sid
        }
        return batch

    _collate_fn = collate_fn
    file = os.path.join(data_dir, '%s.json' % op) if data_file is None else data_file
    if op == 'train':
        print('shuffle data set...')
        data_iter = DataLoader(dataset=NDataSet(file), batch_size=batch_size,
                               shuffle=True, collate_fn=_collate_fn, num_workers=0)
    else:
        data_iter = DataLoader(dataset=NDataSet(file), batch_size=batch_size,
                               shuffle=False, collate_fn=_collate_fn, num_workers=0)
    return data_iter


def test_data_loader():
    w2id, _ = load_vocab(config.vocab_file)
    dl = get_data_loader1(data_dir=config.data_dir, word2id=w2id)
    dl = iter(dl)
    print(len(dl))
    # print(list(dl)[0]['ids'])
    for i, b in enumerate(dl):
        print(i, b['sid'])
        break


def get_data_loader2(op='train', data_dir='', batch_size=4, tokenizer=None, max_seq_len=64, data_file=None):
    """"""
    def convert_entity(words, entities):
        """"""
        converted_entities = []
        c_idx2w_idx = []  # character idx -> word idx
        for i in range(len(words)):
            c_idx2w_idx.extend([i]*len(words[i]))
        for e in entities:
            converted_entities.append((e[0], (c_idx2w_idx[e[1][0]+5], c_idx2w_idx[e[1][1]+5])))
        return converted_entities

    def collate_fn(batch):
        """
        generate data from sentence of batch
        :param batch: sentences for json
        :return: data for model
        """
        sentence, entity, element, sid = [], [], [], []
        for idx, line in enumerate(batch):
            item = json.loads(line)
            # rl = len(item['entity'])
            # item['entity'] = sorted(list(set([(e[0], tuple(e[1])) for e in item['entity']])))
            # item['element'] = item['entity'] + list(set([(el[0], tuple(el[1])) for el in item['element'][rl:]]))
            t_id = item['sid']
            t_words = [config.extra_words[-1]] + tokenizer.tokenize(item['sentence'])
            # t_words = [w for w in item['sentence']]
            t_word_ids = tokenizer.convert_tokens_to_ids(t_words)[:max_seq_len]
            t_word_ids = t_word_ids + [tokenizer.pad_token_id]*(max_seq_len-len(t_words))
            try:
                t_entity = convert_entity(t_words, item['entity'])
            except:
                print(line)
            # t_entity = [(e[0], tuple(e[1])) for e in item['entity']]
            t_element = t_entity + [(e[0], tuple(e[1])) for e in item['element'][len(t_entity):]]
            sentence.append(t_word_ids)
            entity.append(t_entity)
            element.append(t_element)
            sid.append(t_id)
        return {'ids': sentence, 'entity': entity, 'element': element, 'sid': sid}

    _collate_fn = collate_fn
    file = os.path.join(data_dir, '%s.json' % op) if data_file is None else data_file
    if op == 'train':
        print('shuffle data set...')
        data_iter = DataLoader(dataset=NDataSet(file), batch_size=batch_size,
                               shuffle=True, collate_fn=_collate_fn, num_workers=0)
    else:
        data_iter = DataLoader(dataset=NDataSet(file), batch_size=batch_size,
                               shuffle=False, collate_fn=_collate_fn, num_workers=0)
    return data_iter


def has_candidate(candidates):
    """
    check weather new candidate generated
    :param candidates:
    :return:
    """
    for i in range(len(candidates)):
        if len(candidates[i]):
            return True
    return False


def generate_re_type2id(relations):
    """
    generate a map: relation name -> idx
    :param relations: list of (type name, (ops))
    :return: map
    """
    relation_names = [r[0] for r in relations]
    type2id = {r: i for i, r in enumerate(relation_names)}
    return type2id


def generate_type2index(all_type, entity_types):
    """
    map type to idx list
    :param all_type: key
    :param entity_types: entity type by index in each sentence
    :return: type2indexes dict
    """
    b = len(entity_types)
    type2index = []
    for i in range(b):
        cur_t2i = {tp: [[]] for tp in all_type}
        for j, tp in enumerate(entity_types[i]):
            cur_t2i[tp][0].append(j)
        type2index.append(cur_t2i)
    return type2index


def generate_candidate1(relations, type2index, pre_candidates):
    """
    generate candidate relation for cur layer from last layer output
    :param relations: list of (r-type, ([operands1, operands2,...]))
    :param type2index: elem type -> elem index in sentence
    :param pre_candidates: generated candidates in past, [(r_type, (op_idx, ...))]
    :return: candidates [(r_type, (op_idx,...))], labels [1, 0, ...]
    """
    candidates = []
    for r in relations:
        for op_s in r[1]:
            r_c = [tuple()]
            # print(op_s)
            for op in op_s:
                cur_c = []
                op_idx = []
                for opt in op:
                    op_idx.extend(type2index[opt])
                for m in op_idx:
                    for c in r_c:
                        cur_c.append(c+tuple([m]))
                r_c = list(set(cur_c))
                if not r_c:
                    break
            tc = list(filter(lambda x: x not in candidates and x[1][0] != x[1][1], list(set([(r[0], c) for c in r_c]))))
            if len(tc) > 1000:
                continue
            candidates.extend(sorted(tc))
            candidates = list(filter(lambda x: x not in pre_candidates, candidates))
    return candidates


def generate_candidate2(relations, type2index):
    """"""
    candidates = []
    if not config.use_schema:
        # print("not use schema")
        op1_idx, op2_idx = [], []
        for k in type2index.keys():
            op1_idx.extend(type2index[k][-1])
            for idx in type2index[k]:
                op2_idx.extend(idx)
        for idx1 in op1_idx:
            for idx2 in op2_idx:
                for r in relations:
                    c = (r[0], (idx1, idx2))
                    candidates.append(c)
                    if len(candidates) > 500:
                        return list(sorted(set(candidates)))
        op1_idx, op2_idx = [], []
        for k in type2index.keys():
            op2_idx.extend(type2index[k][-1])
            for idx in type2index[k][:-1]:
                op1_idx.extend(idx)
        for idx1 in op1_idx:
            for idx2 in op2_idx:
                for r in relations:
                    c = (r[0], (idx1, idx2))
                    candidates.append(c)
                    if len(candidates) > 500:
                        return list(sorted(set(candidates)))
        return list(sorted(set(candidates)))
    for r in relations:
        # r_c = []
        for op_s in r[1]:
            op1, op2 = op_s[0], op_s[1]
            op1_idx, op2_idx = [], []
            for opt in op1:
                op1_idx.extend(type2index[opt][-1])
            for opt in op2:
                for idx in type2index[opt]:
                    op2_idx.extend(idx)
            for idx1 in op1_idx:
                for idx2 in op2_idx:
                    c = (r[0], (idx1, idx2))
                    if idx1 != idx2:
                        candidates.append(c)
                        if len(candidates) > 5000:
                            return list(sorted(set(candidates)))
            op1_idx, op2_idx = [], []
            for opt in op1:
                for idx in type2index[opt][:-1]:
                    op1_idx.extend(idx)
            for opt in op2:
                op2_idx.extend(type2index[opt][-1])
            for idx1 in op1_idx:
                for idx2 in op2_idx:
                    c = (r[0], (idx1, idx2))
                    if idx1 != idx2:
                        candidates.append(c)
                        if len(candidates) > 5000:
                            return list(sorted(set(candidates)))
            # if len(r_c) > 2000:
            #     break
    return list(sorted(set(candidates)))


def generate_candidate3(relations, type2index):
    """"""
    candidates = []
    if not config.use_schema:
        op1_idx, op2_idx = [], []
        for k in type2index.keys():
            op1_idx.extend(type2index[k][-1])
            for idx in type2index[k]:
                op2_idx.extend(idx)
        for idx1 in op1_idx:
            for idx2 in op2_idx:
                c = (idx1, idx2)
                candidates.append(c)
                if len(candidates) > 500:
                    return list(sorted(set(candidates)))
        op1_idx, op2_idx = [], []
        for k in type2index.keys():
            op2_idx.extend(type2index[k][-1])
            for idx in type2index[k][:-1]:
                op1_idx.extend(idx)
        for idx1 in op1_idx:
            for idx2 in op2_idx:
                c = (idx1, idx2)
                candidates.append(c)
                if len(candidates) > 500:
                    return list(sorted(set(candidates)))
        return list(sorted(set(candidates)))
    left, right = [], []
    for r in relations:
        for ops in r[1]:
            left.extend(ops[0])
            right.extend(ops[1])
    left = sorted(list(set(left)))
    right = sorted(list(set(right)))

    left_idx, right_idx = [], []
    for opt in left:
        left_idx.extend(type2index[opt][-1])
    for opt in right:
        for idx in type2index[opt]:
            right_idx.extend(idx)
    for idx1 in left_idx:
        for idx2 in right_idx:
            if idx1 != idx2:
                candidates.append((idx1, idx2))
    left_idx, right_idx = [], []
    for opt in left:
        for idx in type2index[opt][:-1]:
            left_idx.extend(idx)
    for opt in right:
        right_idx.extend(type2index[opt][-1])
    for idx1 in left_idx:
        for idx2 in right_idx:
            if idx1 != idx2:
                candidates.append((idx1, idx2))
    return sorted(candidates)


def analysis_relation():
    """"""
    files = ['train.json', 'valid.json', 'test.json']
    examples = []
    for file in files:
        lines = open(os.path.join(config.data_dir, file), 'r', encoding='utf-8').readlines()
        examples.extend([json.loads(line) for line in lines])
    
    _, relation = get_type_info(config.type_file)
    relation_types = [r[0] for r in relation]
    all_element, all_entity = [], []
    for item in examples:
        all_entity.append([(e[0], tuple(e[1])) for e in item['entity']])
        all_element.append([(e[0], tuple(e[1])) for e in item['element']])
    dist = analysis_relation_distribution(all_element, all_entity, relation_types,
                                          os.path.join(config.data_dir, 'labeled_analysis.csv'))


def analysis_relation_distribution(element, entity, relation_types, out_file=None):
    """"""
    """type     1   2   ... sum
       l1       n11 n12 ... n1
       l2       n21 n22 ... n2
    """
    n = len(element)
    dist = [[0]*len(relation_types) for _ in range(10)]
    re2id = {r: i for i, r in enumerate(relation_types)}
    for i in range(n):
        layer_num = [0]*len(element[i])
        for j in range(len(entity[i]), len(element[i])):
            layer_num[j] = 1+max(layer_num[element[i][j][1][0]], layer_num[element[i][j][1][1]])
            dist[layer_num[j]][re2id[element[i][j][0]]] += 1
    if out_file is None:
        print("#relation distribution#")
        print('\t\t'.join(['type'] + relation_types + ['sum']))
        for i in range(len(dist)):
            print('\t\t'.join(['l%d' % i] + [str(x) for x in dist[i]] + [str(sum(dist[i]))]))
        print('\t\t'.join(['sum'] + [str(sum(np.array(dist)[:, i])) for i in range(len(relation_types))]))
    else:
        writer = open(out_file, 'w', encoding='utf-8', newline='')
        writer.write(','.join(['type'] + relation_types + ['sum'])+'\n')
        for i in range(len(dist)):
            writer.write(','.join(['l%d' % i] + [str(x) for x in dist[i]] + [str(sum(dist[i]))])+'\n')
        writer.write(','.join(['sum'] + [str(sum(np.array(dist)[:, i])) for i in range(len(relation_types))]))
        writer.close()
        print('write analysis result into %s' % out_file)
    return dist


def save_predict(result, examples, file):
    """
    
    :param result: 
    :param examples: 
    :param file: 
    :return:
    """
    assert len(result) == len(examples)
    writer = open(file, 'w', encoding='utf-8', newline='')
    for r, e in zip(result, examples):
        ne = {'sid': e['sid'], 'entity': e['entity'], 'element': e['entity']+r}
        writer.write(json.dumps(ne, ensure_ascii=False)+'\n')
    writer.close()


def get_all_entity(train_file):
    """"""
    lines = open(train_file, 'r', encoding='utf-8').readlines()
    examples = [json.loads(line) for line in lines]
    all_type, relation = get_type_info(config.type_file)
    entity_type = all_type[:-len(relation)]
    entity2mention = {e: set() for e in entity_type}
    for item in examples:
        for e in item['entity']:
            entity2mention[e[0]].add(item['sentence'][e[1][0]:e[1][1]+1])
    entity2mention = {item[0]: list(item[1]) for item in entity2mention.items()}
    return entity2mention


def generate_ne_entity_replace(train_file, new_file, gt=1, change_prob=0.5):
    """
    
    :param train_file: 
    :param new_file: 
    :param gt: 
    :param change_prob: 
    :return:
    """
    import copy
    if os.path.exists(new_file):
        return
    # set_random(config.random_seed)
    random.seed(config.random_seed)
    # change_prob, gt = 0.5, 1
    lines = open(train_file, 'r', encoding='utf-8').readlines()
    examples = [json.loads(line) for line in lines]
    all_type, relation = get_type_info(config.type_file)
    entity_type = all_type[:-len(relation)]
    entity2mention = {e: set() for e in entity_type}
    #     examples[i]['element'] = examples[i]['entity'] + examples[i]['element'][rl:]
    for item in examples:
        for e in item['entity']:
            entity2mention[e[0]].add(item['sentence'][e[1][0]:e[1][1] + 1])
    entity2mention = {item[0]: sorted(list(item[1])) for item in entity2mention.items()}

    new_examples = []
    for tn in range(1, gt+1):
        for item in examples:
            if len(item['element']) == len(item['entity']):  #
                continue
            new_item = copy.deepcopy(item)

            entity_len, offset = len(item['entity']), 0
            new_sentence = new_item['sentence']
            sort_entity = sorted([[i, e] for i, e in enumerate(item['entity'])], key=lambda x: x[1][1][0])
            for i in range(entity_len):
                if random.random() > change_prob:
                    new_entity = random.choice(entity2mention[sort_entity[i][1][0]])
                    new_sentence = new_sentence[:sort_entity[i][1][1][0] + offset] + new_entity + new_sentence[
                                                                                                  sort_entity[i][1][1][
                                                                                                      1] + 1 + offset:]
                    new_item['entity'][sort_entity[i][0]][1] = [sort_entity[i][1][1][0] + offset,
                                                                sort_entity[i][1][1][0] + offset + len(new_entity) - 1]
                    new_item['element'][sort_entity[i][0]][1] = new_item['entity'][sort_entity[i][0]][1]
                    offset += len(new_entity) - (sort_entity[i][1][1][1] - sort_entity[i][1][1][0] + 1)
                else:
                    new_item['entity'][sort_entity[i][0]][1] = [sort_entity[i][1][1][0] + offset,
                                                                sort_entity[i][1][1][1] + offset]
                    new_item['element'][sort_entity[i][0]][1] = new_item['entity'][sort_entity[i][0]][1]
            new_item['sentence'] = new_sentence
            # new_item['sid'] = str(10000*tn + int(item['sid']))
            new_item['sid'] = 'er-%s' % item['sid']
            if new_item != item:
                new_examples.append(new_item)
    new_examples.extend(examples)
    random.shuffle(new_examples)
    # new_file = os.path.join(config.data_dir, '%s-%s.json' % (config.generated_prefix, config.train_prefix))
    writer = open(new_file, 'w', encoding='utf-8', newline='')
    for e in new_examples:
        writer.write(json.dumps(e, ensure_ascii=False)+'\n')
    writer.close()
    print('save generated %d examples into %s' % (len(new_examples)-len(examples), new_file))


def generate_ne_entity_replace2(file1, file2, new_file, gt=1, change_prob=0.5):
    """
    :param file1: 
    :param file2: 
    :param new_file: 
    :param gt: 
    :param change_prob: 
    :return:
    """
    import copy
    if os.path.exists(new_file):
        return
    random.seed(config.random_seed)
    examples1 = [json.loads(line) for line in open(file1, 'r', encoding='utf-8').readlines()]
    examples2 = [json.loads(line) for line in open(file2, 'r', encoding='utf-8').readlines()]
    all_type, relation = get_type_info(config.type_file)
    entity_type = all_type[:-len(relation)]
    entity2mention = {e: set() for e in entity_type}
    
    for item in examples1+examples2:
        for e in item['entity']:
            entity2mention[e[0]].add(item['sentence'][e[1][0]:e[1][1]+1])
    entity2mention = {item[0]: sorted(list(item[1])) for item in entity2mention.items()}
    new_examples = []
    new_num = int(gt*len(examples1))
    while len(new_examples) < new_num:
        cur_idx = random.randint(0, len(examples1)-1)
        # 
        cur_ex = examples1[cur_idx]
        new_ex = copy.deepcopy(cur_ex)
        # 
        entity_len, offset = len(cur_ex['entity']), 0
        new_sentence = new_ex['sentence']
        sort_entity = sorted([[i, e] for i, e in enumerate(cur_ex['entity'])], key=lambda x: x[1][1][0])
        for i in range(entity_len):
            if random.random() > change_prob:
                new_entity = random.choice(entity2mention[sort_entity[i][1][0]])
                new_sentence = new_sentence[:sort_entity[i][1][1][0] + offset] + new_entity + new_sentence[
                                                                                                  sort_entity[i][1][1][
                                                                                                      1] + 1 + offset:]
                new_ex['entity'][sort_entity[i][0]][1] = [sort_entity[i][1][1][0] + offset,
                                                          sort_entity[i][1][1][0] + offset + len(new_entity) - 1]
                new_ex['element'][sort_entity[i][0]][1] = new_ex['entity'][sort_entity[i][0]][1]
                offset += len(new_entity)-(sort_entity[i][1][1][1]-sort_entity[i][1][1][0]+1)
            else:
                new_ex['entity'][sort_entity[i][0]][1] = [sort_entity[i][1][1][0] + offset,
                                                          sort_entity[i][1][1][1] + offset]
                new_ex['element'][sort_entity[i][0]][1] = new_ex['entity'][sort_entity[i][0]][1]
        new_ex['sentence'] = new_sentence
        if new_ex != cur_ex:
            new_ex['sid'] = 'er-%s' % cur_ex['sid']
            new_examples.append(new_ex)
    new_examples.extend(examples1)
    random.shuffle(new_examples)
    writer = open(new_file, 'w', encoding='utf-8')
    for e in new_examples:
        writer.write(json.dumps(e, ensure_ascii=False)+'\n')
    writer.close()
    print('save generated %d examples into %s' % (len(new_examples) - len(examples1), new_file))


def generate_merge_example(train_file, new_file, gt=1, g_prob=0.5):
    """"""
    import copy
    # set_random(config.random_seed)
    random.seed(config.random_seed)
    # change_prob, gt = 0.5, 1
    # 
    lines = open(train_file, 'r', encoding='utf-8').readlines()
    examples = [json.loads(line) for line in lines]
    max_sent_len = max([len(e['sentence']) for e in examples])
    print('max sentence length is %d' % max_sent_len)
    examples = sorted(examples, key=lambda x: len(x['sentence']))
    new_examples, new_hash = [], set()
    raw_num, new_num = len(examples), gt*len(examples)
    len2eid = [-1]*(max_sent_len+1)
    for i in range(len(examples)):
        len2eid[len(examples[i]['sentence'])] = i
    for i in range(1, max_sent_len):
        if len2eid[i] == -1:
            len2eid[i] = len2eid[i-1]
    while len(new_examples) < new_num:
        ei = random.randint(0, raw_num-1)
        j_l = min(config.max_sent_len-1-len(examples[ei]['sentence']), max_sent_len)
        if len2eid[j_l] == -1:
            continue
        ej = random.randint(0, len2eid[j_l])
        if (ei, ej) not in new_hash and (ej, ei) not in new_hash:
            new_hash.add((ei, ej))
            new_sentence = examples[ei]['sentence']+'。'+examples[ej]['sentence']
            new_sid = '%s-%s' % (examples[ei]['sid'], examples[ej]['sid'])
            len1 = len(examples[ei]['sentence'])+1
            new_entity = examples[ei]['entity'] + [[e[0], [e[1][0]+len1, e[1][1]+len1]] for e in examples[ej]['entity']]
            i_layer, j_layer = [0]*len(examples[ei]['entity']), [0]*len(examples[ej]['entity'])
            for i, e in enumerate(examples[ei]['element'][len(examples[ei]['entity']):]):
                i_layer.append(max(i_layer[e[1][0]], i_layer[e[1][1]])+1)
            for i, e in enumerate(examples[ej]['element'][len(examples[ej]['entity']):]):
                j_layer.append(max(j_layer[e[1][0]], j_layer[e[1][1]])+1)
            new_element = examples[ei]['entity']+examples[ej]['entity']
            index_i, index_j = len(examples[ei]['entity']), len(examples[ej]['entity'])
            i_r2n, j_r2n = list(range(index_i)), list(range(index_i, index_i+index_j))
            while index_i < len(i_layer) or index_j < len(j_layer):
                while index_i < len(i_layer):
                    elem_i = examples[ei]['element'][index_i]
                    new_element.append([elem_i[0], [i_r2n[elem_i[1][0]], i_r2n[elem_i[1][1]]]])
                    i_r2n.append(len(new_element)-1)
                    index_i += 1
                    if index_i >= len(i_layer) or i_layer[index_i] != i_layer[index_i-1]:
                        break
                while index_j < len(j_layer):
                    elem_j = examples[ej]['element'][index_j]
                    new_element.append([elem_j[0], [j_r2n[elem_j[1][0]], j_r2n[elem_j[1][1]]]])
                    j_r2n.append(len(new_element)-1)
                    index_j += 1
                    if index_j >= len(j_layer) or j_layer[index_j] != j_layer[index_j-1]:
                        break
            new_examples.append({'sid': new_sid, 'sentence': new_sentence,
                                 'entity': new_entity, 'element': new_element})
    new_examples.extend(examples)
    random.shuffle(new_examples)
    # new_file = os.path.join(config.data_dir, '%s-%s.json' % (config.generated_prefix, config.train_prefix))
    writer = open(new_file, 'w', encoding='utf-8', newline='')
    for e in new_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()
    print('save generated %d examples into %s' % (len(new_examples) - len(examples), new_file))


def convert_1to2(v1_file, v2_dir, v2_num=3):
    """"""
    lines = open(v1_file, 'r', encoding='utf-8').readlines()
    examples = [json.loads(line) for line in lines]
    # all_type, relation = get_type_info(config.type_file)
    # type2id = {tp: i for i, tp in enumerate(all_type)}
    new_examples = []
    text = []
    for e in examples:
        new_id = e['sid']
        new_text = e['sentence']
        if new_text in text:
            continue
        text.append(new_text)
        new_entity = [{'key': 'e%s-%d-%d' % (new_id, ent[1][0], ent[1][1]), # 'entityID': type2id[ent[0]],
                       'entityType': ent[0], 'start': ent[1][0], 'end': ent[1][1]} for ent in e['entity']]
        el = len(new_entity)
        # new_relation = [{'key': 'r%s-%d-%d' % (new_id, rel[1][0], rel[1][1]), 'relationID': type2id[rel[0]],
        #                  'relationType': rel[0]
        #                  } for rel in e['element'][len(new_entity):]]
        new_relation = []
        for rel in e['element'][len(new_entity):]:
            new_relation.append({
                'key': 'r%s-%d-%d' % (new_id, rel[1][0], rel[1][1]), # 'relationID': type2id[rel[0]],
                'relationType': rel[0],
                'operand1': new_entity[rel[1][0]]['key'] if rel[1][0] < el else new_relation[rel[1][0]-el],
                'operand2': new_entity[rel[1][1]]['key'] if rel[1][1] < el else new_relation[rel[1][1]-el]
            })
        new_examples.append({'id': new_id, 'text': new_text, 'meta': {}, 'tags': {}})
                             # 'tags': {'entities': new_entity, 'relations': new_relation}})
    os.makedirs(v2_dir, exist_ok=True)
    print(len(new_examples))
    split_num = len(new_examples)//v2_num + 1
    for i in range(1, v2_num+1):
        v2_file = os.path.join(v2_dir, '%d.jsonl' % i)
        writer = open(v2_file, 'w', encoding='utf-8', newline='')
        for e in new_examples[split_num*(i-1):split_num*i]:
            writer.write(json.dumps(e, ensure_ascii=False) + '\n')
        writer.close()
        print('convert %s into %s' % (v1_file, v2_file))


def convert_2to1(v2_dir, v1_file):
    """"""
    files = os.listdir(v2_dir)
    examples = []
    for f in files:
        file = os.path.join(v2_dir, f)
        examples.extend([json.loads(x) for x in open(file, 'r', encoding='utf-8').readlines()])
    new_examples = []
    new_sid = set()
    for e in examples:
        ne = {'sid': str(e['id']), 'sentence': e['text']}
        entity = e['tags']['entities']
        ne_entity = [(ent['entityType'], (ent['start'], ent['end']-1)) for ent in entity]
        if str(e['id']) in new_sid or len(ne_entity) == 0:
            continue
        entity_num = len(entity)
        key2idx = {ent['key']: i for i, ent in enumerate(entity)}
        relations = e['tags']['relations']
        ne_relations = []
        try:
            for i, r in enumerate(relations):
                key2idx[r['key']] = i + entity_num
                ne_relations.append((r['relationType'], (key2idx[r['operand1']], key2idx[r['operand2']])))
        except:
            print(e['id'], e['text'])
            continue
        ne['entity'] = ne_entity
        ne['element'] = ne_entity+ne_relations
        new_examples.append(ne)
        new_sid.add(str(e['id']))
    print(len(new_examples))
    writer = open(v1_file, 'w', encoding='utf-8')
    for e in new_examples:
        writer.write(json.dumps(e, ensure_ascii=False)+'\n')
    writer.close()


def generate_raw_weak_data(data_dir, ratio=0.1):
    """"""
    set_random(config.random_seed+3)
    _, relations = get_type_info(config.type_file)
    new_file = os.path.join(data_dir, '%s-%d.json' % (config.few_labeled_prefix, int(ratio * config.few_scale)))
    weak_file = os.path.join(data_dir, '%s-%d.json' % (config.few_unlabeled_prefix, int(ratio * config.few_scale)))
    if os.path.exists(new_file):
        return

    train_file = os.path.join(data_dir, '%s.json' % config.train_prefix)
    train_examples = [json.loads(line) for line in open(train_file, 'r', encoding='utf-8').readlines()]
    train_len = len(train_examples)
    new_index = set()
    relation_type = [r[0] for r in relations]
    new_types = set()
    for i, e in enumerate(train_examples):
        e_r = [elem[0] for elem in e['element']][len(e['entity']):]
        if not set(e_r).issubset(new_types):
            new_index.add(i)
            new_types = new_types | set(e_r)
        if len(new_types) == len(relation_type):
            break
    while len(new_index) < train_len * ratio:
        ci = random.randint(0, train_len - 1)
        if ci not in new_index:
            new_index.add(ci)
    writer_new = open(new_file, 'w', encoding='utf-8')
    writer_weak = open(weak_file, 'w', encoding='utf-8')
    for i, e in enumerate(train_examples):
        if i in new_index:
            writer_new.write(json.dumps(e, ensure_ascii=False) + '\n')
        else:
            writer_weak.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer_new.close()
    writer_weak.close()


def generate_n_way_k_shot(train_dir, n=5, k=1):
    import copy
    set_random(config.random_seed)
    train_file = os.path.join(train_dir, 'train.json')
    examples = [json.loads(line) for line in open(train_file, 'r', encoding='utf-8').readlines()]
    res_examples = []
    _, relations = get_type_info(config.type_file)
    # step1
    label2num = {r[0]: 0 for r in relations}
    for e in examples:
        for r in e['element'][len(e['entity']):]:
            label2num[r[0]] += 1
    label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
    # step2
    label_count = {label2num[i][0]: 0 for i in range(n)}
    n_label = set(label_count.keys())
    print(label2num)
    fail_count = 100000
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in res_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label).issubset(n_label):
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                new_label_count[l] += 1
                if new_label_count[l] > 3*k:
                    can_add = False
                    break
            if can_add:
                label_count = new_label_count
                res_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < k:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(res_examples))
    nk_file = os.path.join(config.data_dir, '%d-way-%d-shot-train.json' % (n, k))
    writer = open(nk_file, 'w', encoding='utf-8')
    for e in res_examples:
        writer.write(json.dumps(e, ensure_ascii=False)+'\n')
    writer.close()

    # step3
    test_examples = []
    label_count = {label2num[i][0]: 0 for i in range(n)}
    fail_count = 100000
    kt = k * 5
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in test_examples + res_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label).issubset(n_label):
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                new_label_count[l] += 1
                if new_label_count[l] > 3 * kt:
                    can_add = False
                    break
            if can_add:
                label_count = new_label_count
                test_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < kt:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(test_examples))
    nk_test_file = os.path.join(config.data_dir, '%d-way-%d-shot-test.json' % (n, k))
    writer = open(nk_test_file, 'w', encoding='utf-8')
    for e in test_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()

    # step4
    valid_examples = []
    label_count = {label2num[i][0]: 0 for i in range(n)}
    fail_count = 100000
    kv = k * 5
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in test_examples + res_examples + valid_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label).issubset(n_label):
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                new_label_count[l] += 1
                if new_label_count[l] > 3 * kv:
                    can_add = False
                    break
            if can_add:
                label_count = new_label_count
                valid_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < kv:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(valid_examples))
    nk_valid_file = os.path.join(config.data_dir, '%d-way-%d-shot-valid.json' % (n, k))
    writer = open(nk_valid_file, 'w', encoding='utf-8')
    for e in valid_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()
    return list(n_label)


def generate_n_way_k_shot2(train_dir, n=5, k=1, m=3, test_k=30):
    import copy
    set_random(config.random_seed+1)
    train_file = os.path.join(train_dir, 'train.json')
    examples = [json.loads(line) for line in open(train_file, 'r', encoding='utf-8').readlines()]
    _, relations = get_type_info(config.type_file)
    # step1
    label2num = {r[0]: 0 for r in relations}
    for e in examples:
        for r in e['element'][len(e['entity']):]:
            label2num[r[0]] += 1
    label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
    # step2
    label_count = {label2num[i][0]: 0 for i in range(n)}
    n_label = set(label_count.keys())
    print(label2num)
    fail_count = 100000
    test_examples = []
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in test_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label).issubset(n_label):
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                new_label_count[l] += 1
                if new_label_count[l] > m * test_k:
                    can_add = False
                    break
            if can_add:
                label_count = new_label_count
                test_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < test_k:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(test_examples))
    nk_test_file = os.path.join(config.data_dir, '%d-way-%d-shot-test.json' % (n, k))
    writer = open(nk_test_file, 'w', encoding='utf-8')
    for e in test_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()
    # step3
    set_random(config.random_seed+25)
    train_examples = []
    label_count = {label2num[i][0]: 0 for i in range(n)}
    fail_count = 100000
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in train_examples+test_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label).issubset(n_label):
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                new_label_count[l] += 1
                if new_label_count[l] > m * k:
                    can_add = False
                    break
            if can_add:
                label_count = new_label_count
                train_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < k:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(train_examples))
    nk_file = os.path.join(config.data_dir, '%d-way-%d-shot-train.json' % (n, k))
    writer = open(nk_file, 'w', encoding='utf-8')
    for e in train_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()

    # step4
    valid_examples = []
    label_count = {label2num[i][0]: 0 for i in range(n)}
    fail_count = 100000
    kv = k * 2
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in test_examples + train_examples + valid_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label).issubset(n_label):
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                new_label_count[l] += 1
                if new_label_count[l] > 3 * kv:
                    can_add = False
                    break
            if can_add:
                label_count = new_label_count
                valid_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < kv:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(valid_examples))
    nk_valid_file = os.path.join(config.data_dir, '%d-way-%d-shot-valid.json' % (n, k))
    writer = open(nk_valid_file, 'w', encoding='utf-8')
    for e in valid_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()


def generate_n_way_k_shot_low(train_dir, n=5, k=1, n_idx=3, m=3):
    import copy
    set_random(config.random_seed+2)
    train_file = os.path.join(train_dir, 'train.json')
    examples = [json.loads(line) for line in open(train_file, 'r', encoding='utf-8').readlines()]
    _, relations = get_type_info(config.type_file)
    # step1
    label2num = {r[0]: 0 for r in relations}
    for e in examples:
        for r in e['element'][len(e['entity']):]:
            label2num[r[0]] += 1
    label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
    # step2
    res_examples = []
    label_count = {label2num[i+n_idx][0]: 0 for i in range(n)}
    n_label = set(label_count.keys())
    print(label2num)
    fail_count = 100000
    train_label = set()
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in res_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label) & n_label:
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                if l in n_label:
                    new_label_count[l] += 1
                    if new_label_count[l] > m*k:
                        can_add = False
                        break
            if can_add:
                train_label = train_label.union(set(cur_label))
                label_count = new_label_count
                res_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < k:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(res_examples))
    nk_file = os.path.join(config.data_dir, 'low-%d-way-%d-shot-train.json' % (n, k))
    writer = open(nk_file, 'w', encoding='utf-8')
    for e in res_examples:
        writer.write(json.dumps(e, ensure_ascii=False)+'\n')
    writer.close()

    # step3
    test_examples = []
    label_count = {label2num[i+n_idx][0]: 0 for i in range(n)}
    fail_count = 100000
    kt = k*4
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in test_examples+res_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label) & n_label:
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                if l not in train_label:
                    can_add = False
                    break
                if l in n_label:
                    new_label_count[l] += 1
                    if new_label_count[l] > m*kt:
                        can_add = False
                        break
            if can_add:
                label_count = new_label_count
                test_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < kt:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(test_examples))
    nk_test_file = os.path.join(config.data_dir, 'low-%d-way-%d-shot-test.json' % (n, k))
    writer = open(nk_test_file, 'w', encoding='utf-8')
    for e in test_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()

    # step4
    valid_examples = []
    label_count = {label2num[i + n_idx][0]: 0 for i in range(n)}
    fail_count = 100000
    kv = k * 2
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in test_examples+res_examples+valid_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label) & n_label:
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                if l not in train_label:
                    can_add = False
                    break
                if l in n_label:
                    new_label_count[l] += 1
                    if new_label_count[l] > m*kv:
                        can_add = False
                        break
            if can_add:
                label_count = new_label_count
                valid_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < kv:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(valid_examples))
    nk_valid_file = os.path.join(config.data_dir, 'low-%d-way-%d-shot-valid.json' % (n, k))
    writer = open(nk_valid_file, 'w', encoding='utf-8')
    for e in valid_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()


def generate_n_way_k_shot_low2(train_dir, n=5, k=1, n_idx=3, m=3, test_k=20):
    import copy
    set_random(config.random_seed+13)
    train_file = os.path.join(train_dir, 'train.json')
    examples = [json.loads(line) for line in open(train_file, 'r', encoding='utf-8').readlines()]
    _, relations = get_type_info(config.type_file)
    # step1
    label2num = {r[0]: 0 for r in relations}
    for e in examples:
        for r in e['element'][len(e['entity']):]:
            label2num[r[0]] += 1
    label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
    # step2
    test_examples = []
    label_count = {label2num[i + n_idx][0]: 0 for i in range(n)}
    n_label = set(label_count.keys())
    print(label2num)
    fail_count = 100000
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in test_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label) & n_label:
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                if l in n_label:
                    new_label_count[l] += 1
                    if new_label_count[l] > m * test_k:
                        can_add = False
                        break
            if can_add:
                label_count = new_label_count
                test_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < test_k:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(test_examples))
    nk_test_file = os.path.join(config.data_dir, 'low-%d-way-%d-shot-test.json' % (n, k))
    writer = open(nk_test_file, 'w', encoding='utf-8')
    for e in test_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()
    # step3
    set_random(config.random_seed+22)
    train_examples = []
    label_count = {label2num[i + n_idx][0]: 0 for i in range(n)}
    fail_count = 100000
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in train_examples+test_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label) & n_label:
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                if l in n_label:
                    new_label_count[l] += 1
                    if new_label_count[l] > m * k:
                        can_add = False
                        break
            if can_add:
                label_count = new_label_count
                train_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < k:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(train_examples))
    nk_file = os.path.join(config.data_dir, 'low-%d-way-%d-shot-train.json' % (n, k))
    writer = open(nk_file, 'w', encoding='utf-8')
    for e in train_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()
    # step4
    valid_examples = []
    label_count = {label2num[i + n_idx][0]: 0 for i in range(n)}
    fail_count = 100000
    kv = k * 2
    while fail_count > 0:
        ce = random.choice(examples)
        if ce not in train_examples+test_examples+valid_examples:
            cur_label = [x[0] for x in ce['element'][len(ce['entity']):]]
            if not set(cur_label) & n_label:
                continue
            new_label_count = copy.deepcopy(label_count)
            can_add = True
            for l in cur_label:
                if l in n_label:
                    new_label_count[l] += 1
                    if new_label_count[l] > m * kv:
                        can_add = False
                        break
            if can_add:
                label_count = new_label_count
                valid_examples.append(ce)
            else:
                fail_count -= 1
                continue
            can_end = True
            for item in label_count.items():
                if item[1] < k:
                    can_end = False
                    break
            if can_end:
                break
    print(fail_count, len(valid_examples))
    nk_valid_file = os.path.join(config.data_dir, 'low-%d-way-%d-shot-valid.json' % (n, k))
    writer = open(nk_valid_file, 'w', encoding='utf-8')
    for e in valid_examples:
        writer.write(json.dumps(e, ensure_ascii=False) + '\n')
    writer.close()


def generate_shot_weak_data(train_dir, n=5, k=1):
    set_random(config.random_seed)
    train_file = os.path.join(train_dir, 'train.json')
    examples = [json.loads(line) for line in open(train_file, 'r', encoding='utf-8').readlines()]
    _, relations = get_type_info(config.type_file)
    label2num = {r[0]: 0 for r in relations}
    for e in examples:
        for r in e['element'][len(e['entity']):]:
            label2num[r[0]] += 1
    label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
    print(label2num)
    label_count = {label2num[i][0]: 0 for i in range(n)}
    n_label = set(label_count.keys())
    used_examples = []
    stages = ['train', 'valid', 'test']
    files = [os.path.join(train_dir, tf) for tf in ['%d-way-%d-shot-%s.json' % (n, k, m) for m in stages]]
    for f in files:
        used_examples.extend([json.loads(line) for line in open(f, 'r', encoding='utf-8').readlines()])
    count = 0
    other_examples = []
    for e in examples:
        if e not in used_examples:
            count += 1
            cur_label = [x[0] for x in e['element'][len(e['entity']):]]
            if not set(cur_label).issubset(n_label):
                continue
            else:
                other_examples.append(e)
    print(count, len(used_examples), len(examples), len(other_examples))
    unlabeled_file = os.path.join(train_dir, '%d-way-%d-shot-unlabeled.json' % (n, k))
    writer = open(unlabeled_file, 'w', encoding='utf-8')
    for e in other_examples:
        writer.write(json.dumps(e, ensure_ascii=False)+'\n')
    writer.close()


if __name__ == '__main__':
    
    generate_n_way_k_shot_low2(config.data_dir, 5, 1)
    generate_n_way_k_shot_low2(config.data_dir, 5, 5)
    generate_n_way_k_shot_low2(config.data_dir, 5, 10)
