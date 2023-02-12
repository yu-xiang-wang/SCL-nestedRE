# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : utils.py
# @Software: PyCharm


from torch.utils.data import Dataset, DataLoader
import os
import json
import torch as t
import random
import numpy as np
import pandas as pd


def set_random(seed):
    random.seed(seed)
    t.manual_seed(seed)
    np.random.seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def str_q2b(ustring):
    """"""
    r_string = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        r_string += chr(inside_code)
    return r_string


def get_entity_rep(predict, entity):
    p2er = [e for e in entity]
    for elem in predict:
        if elem in entity:  
            continue
        p2er.append((elem[0], (p2er[elem[1][0]][1], p2er[elem[1][1]][1]))) 
    return p2er


def get_correct_num(elements, predict, entity):
    """"""
    gold_rep = get_entity_rep(elements, entity)  
    pred_rep = get_entity_rep(predict, entity)  
    correct = list(set(gold_rep) & set(pred_rep))
    return len(correct)-len(entity)


def compute_metric(all_elements, all_predict, all_entity):
    assert len(all_elements) == len(all_predict) == len(all_entity) 
    correct, predict_num, gold_num = 0, 0, 0  
    for i in range(len(all_elements)):
        correct += get_correct_num(all_elements[i], all_predict[i], all_entity[i])
        predict_num += len(all_predict[i])
        gold_num += (len(all_elements[i])-len(all_entity[i]))
    if not predict_num:
        predict_num = 1
    if not gold_num:
        gold_num = 1
    p = round(correct/predict_num, 5)
    r = round(correct/gold_num, 5)
    f1 = round(2*p*r/(p+r), 5) if p and r else 0
    return p, r, f1


def compute_metric_relation(all_elements, all_predict, all_entity, target_relation):
    def get_correct_num_t(elements, predict, entity):
        gold_rep = get_entity_rep(elements, entity)
        pred_rep = get_entity_rep(predict, entity)
        t_correct = list(set(gold_rep) & set(pred_rep))
        return sum([1 if x[0] in target_relation else 0 for x in t_correct])
    assert len(all_elements) == len(all_predict) == len(all_entity)
    correct, predict_num, gold_num = 0, 0, 0
    for i in range(len(all_elements)):
        correct += get_correct_num_t(all_elements[i], all_predict[i], all_entity[i])
        predict_num += sum([1 if x[0] in target_relation else 0 for x in all_predict[i]])
        gold_num += sum([1 if x[0] in target_relation else 0 for x in all_elements[i][len(all_entity[i]):]])
    if not predict_num:
        predict_num = 1
    if not gold_num:
        gold_num = 1
    p = round(correct/predict_num, 5)
    r = round(correct/gold_num, 5)
    f1 = round(2*p*r/(p+r), 5) if p and r else 0
    return p, r, f1


def compute_metric_relation_guided(all_elements, all_predict, all_entity, target_relation):
    assert len(all_elements) == len(all_predict) == len(all_entity)
    correct_num, predict_num, gold_num = 0, 0, 0
    for i in range(len(all_elements)):
        assert len(all_elements[i]) == len(all_predict[i])
        for j in range(len(all_elements[i])):
            cor = list(set(all_elements[i][j]) & set(all_predict[i][j]))
            correct_num += sum([1 if x[0] in target_relation else 0 for x in cor])
            predict_num += sum([1 if x[0] in target_relation else 0 for x in all_predict[i][j]])
            gold_num += sum([1 if x[0] in target_relation else 0 for x in all_elements[i][j]])
    if not predict_num:
        predict_num = 1
    if not gold_num:
        gold_num = 1
    p = round(correct_num/predict_num, 5)
    r = round(correct_num/gold_num, 5)
    f1 = round(2*p*r/(p+r), 5) if p and r else 0
    return p, r, f1


def relation_filtration_study(all_elements, all_predict, all_entity, relation_types):
    rel2id = {r: i for i, r in enumerate(relation_types)}
    error_num = [0] * len(relation_types)
    ground_truth = [0] * len(relation_types)
    for i in range(len(all_elements)):
        entity_len = len(all_entity[i])
        gold_rep = get_entity_rep(all_elements[i], all_entity[i])[entity_len:]
        pred_rep = get_entity_rep(all_predict[i], all_entity[i])[entity_len:]
        error = set(gold_rep) - (set(gold_rep) & set(pred_rep))
        for e in error:
            r_id = rel2id[e[0]]
            error_num[r_id] += 1
        for g in gold_rep:
            ground_truth[rel2id[g[0]]] += 1
    id2rel = relation_types
    print('filtration from scratch')
    print(id2rel)
    print('truth', ground_truth)
    print('error', error_num)


def compute_metric_by_dist(all_elements, all_predict, all_entity, relation_types):
    assert len(all_elements) == len(all_predict) == len(all_entity)
    re2id = {r: i for i, r in enumerate(relation_types)}
    gold_dist = [[0]*len(relation_types) for _ in range(10)]
    pred_dist = [[0]*len(relation_types) for _ in range(10)]
    cor_dist = [[0]*len(relation_types) for _ in range(10)]
    for i in range(len(all_elements)):
        entity_len = len(all_entity[i])
        gold_rep = get_entity_rep(all_elements[i], all_entity[i])[entity_len:]
        pred_rep = get_entity_rep(all_predict[i], all_entity[i])[entity_len:]
        gold_layer = [0]*len(all_elements[i])
        for j in range(entity_len, len(all_elements[i])):
            gold_layer[j] = 1+max(gold_layer[all_elements[i][j][1][0]], gold_layer[all_elements[i][j][1][1]])
        pred_elem = all_entity[i]+all_predict[i]
        pred_layer = [0]*len(pred_elem)
        for j in range(entity_len, len(pred_elem)):
            pred_layer[j] = 1+max(pred_layer[pred_elem[j][1][0]], pred_layer[pred_elem[j][1][1]])
        cor = list(set(gold_rep) & set(pred_rep))
        for e in cor:
            r_id = re2id[e[0]]
            cor_dist[gold_layer[gold_rep.index(e)+entity_len]][r_id] += 1
        for e, l in zip(gold_rep, gold_layer[entity_len:]):
            gold_dist[l][re2id[e[0]]] += 1
        for e, l in zip(pred_rep, pred_layer[entity_len:]):
            pred_dist[l][re2id[e[0]]] += 1
    for i in range(len(gold_dist)):
        gold_dist[i].append(sum(gold_dist[i]))
        pred_dist[i].append(sum(pred_dist[i]))
        cor_dist[i].append(sum(cor_dist[i]))
    gold_dist.append(list(np.sum(gold_dist, 0)))
    pred_dist.append(list(np.sum(pred_dist, 0)))
    cor_dist.append(list(np.sum(cor_dist, 0)))
    gold_dist, pred_dist, cor_dist = np.array(gold_dist), np.array(pred_dist), np.array(cor_dist)
    gold_dist[gold_dist < 1] = 1
    pred_dist[pred_dist < 1] = 1
    p, r, f1 = cor_dist / pred_dist, cor_dist / gold_dist, 2*cor_dist / (pred_dist+gold_dist)
    return p, r, f1


def relation_filtration_study_guide(all_elements, all_predict, relation_types):
    rel2id = {r: i for i, r in enumerate(relation_types)}
    error_num = [0] * len(relation_types)
    ground_truth = [0] * len(relation_types)
    for i in range(len(all_elements)):
        assert len(all_elements[i]) == len(all_predict[i])
        for j in range(len(all_elements[i])):
            error = set(all_elements[i][j]) - (set(all_elements[i][j]) & set(all_predict[i][j]))
            for e in error:
                r_id = rel2id[e[0]]
                error_num[r_id] += 1
            for g in all_elements[i][j]:
                ground_truth[rel2id[g[0]]] += 1
    id2rel = relation_types
    print('filtration guided')
    print(id2rel)
    print('truth', ground_truth)
    print('error', error_num)


def compute_metric_by_dist_layers(all_elements, all_predict, relation_types):
    assert len(all_elements) == len(all_predict)
    re2id = {r: i for i, r in enumerate(relation_types)}
    gold_dist = [[0] * len(relation_types) for _ in range(10)]
    pred_dist = [[0] * len(relation_types) for _ in range(10)]
    cor_dist = [[0] * len(relation_types) for _ in range(10)]
    for i in range(len(all_elements)):  
        assert len(all_elements[i]) == len(all_predict[i])
        for j in range(len(all_elements[i])): 
            for k in range(len(all_elements[i][j])): 
                gold_dist[j][re2id[all_elements[i][j][k][0]]] += 1
            for k in range(len(all_predict[i][j])):  
                pred_dist[j][re2id[all_predict[i][j][k][0]]] += 1
            cor = list(set(all_elements[i][j]) & set(all_predict[i][j]))  
            for k in range(len(cor)):
                cor_dist[j][re2id[cor[k][0]]] += 1
    for i in range(len(gold_dist)):
        gold_dist[i].append(sum(gold_dist[i]))
        print(i, ": ", gold_dist[i][-1])
        pred_dist[i].append(sum(pred_dist[i]))
        cor_dist[i].append(sum(cor_dist[i]))
    gold_dist.append(list(np.sum(gold_dist, 0)))
    pred_dist.append(list(np.sum(pred_dist, 0)))
    cor_dist.append(list(np.sum(cor_dist, 0)))
    gold_dist, pred_dist, cor_dist = np.array(gold_dist), np.array(pred_dist), np.array(cor_dist)
    gold_dist[gold_dist < 1] = 1
    pred_dist[pred_dist < 1] = 1
    p, r, f1 = cor_dist / pred_dist, cor_dist / gold_dist, 2 * cor_dist / (pred_dist + gold_dist)
    return p, r, f1


def print_dist(p, r, f1, relation_types, out_file=None):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    p_df = pd.DataFrame(p, columns=relation_types+['sum'], index=['l%d' % i for i in range(len(p)-1)]+['sum'])
    r_df = pd.DataFrame(r, columns=relation_types+['sum'], index=['l%d' % i for i in range(len(p)-1)]+['sum'])
    f_df = pd.DataFrame(f1, columns=relation_types+['sum'], index=['l%d' % i for i in range(len(p)-1)]+['sum'])
    # print('precision dist\n', p_df, '\nrecall dist\n', r_df, '\nf1 dist\n', f_df)
    if out_file is not None:
        print(f"write to {out_file}")
        with pd.ExcelWriter(out_file) as writer:
            p_df.to_excel(writer, sheet_name='precision')
            r_df.to_excel(writer, sheet_name='recall')
            f_df.to_excel(writer, sheet_name='f1-score')


def calc_running_avg_loss(running_avg_loss, loss):
    decay = 0.99
    if running_avg_loss is None:
        return loss
    else:
        return min(running_avg_loss * decay + (1-decay) * loss, 12)


def save_config(config, file):
    writer = open(file, 'w', encoding='utf-8')
    names = [v for v in dir(config) if not v.startswith('_')]
    for v in names:
        writer.write('%s = %s\n' % (v, str(eval('config.%s' % v))))
    writer.close()


def print_config(config):
    names = [v for v in dir(config) if not v.startswith('_')]
    for v in names:
        print('%s = %s' % (v, str(eval('config.%s' % v))))


def merge_result(model_dir, k_fold=None, res_file='metric.xlsx'):
    metrics = ['precision', 'recall', 'f1-score']
    all_data, all_index = [[], [], []], [[], [], []]  # p, r, f
    relations = None
    if k_fold is None:
        models = os.listdir(model_dir)

        for m in models:
            f = os.path.join(os.path.join(model_dir, m), res_file)
            df = pd.read_excel(f, sheet_name=None, engine='openpyxl', index_col='Unnamed: 0')
            for i, name in enumerate(metrics):
                if relations is None:
                    relations = df[name].keys().tolist()
                all_data[i].extend(df[name].values.tolist())
                all_index[i].extend(['%s-%s' % (m, x) for x in df[name].index.tolist()])
        
    else:
        # all_data, all_index = [[], [], []], [[], [], []]  # p, r, f
        for k in range(1, k_fold+1):
            k_dir = os.path.join(model_dir, '%d-%d' % (k_fold, k))
            models = os.listdir(k_dir)
            for m in models:
                f = os.path.join(os.path.join(k_dir, m), res_file)
                df = pd.read_excel(f, sheet_name=None, engine='openpyxl', index_col='Unnamed: 0')
                for i, name in enumerate(metrics):
                    if relations is None:
                        relations = df[name].keys().tolist()
                    all_data[i].extend(df[name].values.tolist())
                    all_index[i].extend(['%s-%d-%d-%s' % (m, k_fold, k, x) for x in df[name].index.tolist()])
    merged_file = os.path.join(model_dir, res_file)
    with pd.ExcelWriter(merged_file) as writer:
        for i, name in enumerate(metrics):
            df = pd.DataFrame(all_data[i], columns=relations, index=all_index[i])
            df.to_excel(writer, sheet_name=name)


def compute_similarity_structure(example1, example2):
    entity1, element1 = example1['entity'], example1['element']
    entity2, element2 = example2['entity'], example2['element']
    M, N1, N2 = 0, len(element1), len(element2)
    el1, el2 = len(entity1), len(entity2)
    
    layer1, layer2 = [0]*N1, [0]*N2
    for i in range(len(entity1), len(element1)):
        layer1[i] = 1+max(layer1[element1[i][1][0]], layer1[element1[i][1][1]])
    for i in range(len(entity2), len(element2)):
        layer2[i] = 1 + max(layer2[element2[i][1][0]], layer2[element2[i][1][1]])
    max1, max2 = max(layer1), max(layer2)
    elem1_idx_by_layer, elem2_idx_by_layer = [[] for _ in range(max1+1)], [[] for _ in range(max2+1)]
    elem1_idx_by_layer[0], elem2_idx_by_layer[0] = [e[0] for e in entity1], [e[0] for e in entity2]
    for idx, l in enumerate(layer1[el1:]):
        elem1_idx_by_layer[l].append(
            (element1[idx+el1][0], (element1[element1[idx+el1][1][0]][0], element1[element1[idx+el1][1][1]][0])))
    for idx, l in enumerate(layer2[el2:]):
        elem2_idx_by_layer[l].append(
            (element2[idx+el2][0], (element2[element2[idx+el2][1][0]][0], element2[element2[idx+el2][1][1]][0])))
    for i in range(min(max1, max2)+1):
        elem = set(elem1_idx_by_layer[i]) & set(elem2_idx_by_layer[i])
        for e in elem:
            M += min(elem1_idx_by_layer[i].count(e), elem2_idx_by_layer[i].count(e))
    p, r = M/max(N1, 1), M/max(1, N2)
    return 2*p*r/(p+r) if p > 0 else 0


def compute_similarity_examples(examples):
    b = len(examples['entity'])
    sim = np.zeros([b, b], dtype=float)
    for i in range(b):
        for j in range(i+1, b):
            try:
                sim[i][j] = compute_similarity_structure(
                    {'entity': examples['entity'][i], 'element': examples['element'][i]},
                    {'entity': examples['entity'][j], 'element': examples['element'][j]}
                )
            except:
                pass
                
    return sim


def t_compute_similarity_structure(train_file):

    examples = [json.loads(line) for line in open(train_file, 'r', encoding='utf-8').readlines()]
    
    le = len(examples)
    m_sim = [0] * le
    idx = [-1] * le
    for i in range(le):
        for j in range(i+1, le):
            sim = compute_similarity_structure(examples[i], examples[j])
            if sim > m_sim[i]:
                m_sim[i] = sim
                idx[i] = j
            if sim > m_sim[j]:
                m_sim[j] = sim
                idx[j] = i
    pair_writer = open('pair.csv', 'w', encoding='utf-8')
    for i, j in enumerate(idx):
        pair_writer.write(examples[i]['sid']+','+examples[i]['sentence']+','+
                          examples[j]['sid']+','+examples[j]['sentence']+','+'%.5f' % m_sim[i] + '\n')
    pair_writer.close()


def deal_flat():
    flat_dir = 'data/flat-relation/semeval2010'
    mt = ['train', 'valid', 'test']
    for m in mt:
        file = os.path.join(flat_dir, '%s.txt' % m)
        print(file)
        lines = open(file, 'r', encoding='utf-8').readlines()
        for line in lines:
            ex = eval(line)
            if ex['h']['pos'][1] - ex['h']['pos'][0] > 1 or ex['t']['pos'][1] - ex['t']['pos'][0] > 1:
                print(ex)


def analysis_train():
    from matplotlib import pyplot as plt
    # examples =  [eval(x) for x in open('data/train.json', 'r', encoding='utf-8').readlines()] + \
    examples = [eval(x) for x in open('data/test.json', 'r', encoding='utf-8').readlines()]
    sent_len, rel_cnt, rel_layer = [], [], []
    for e in examples:
        entity = e['entity']
        element = e['element']
        layer = [0] * len(element)
        for i in range(len(entity), len(element)):
            layer[i] = 1 + max(layer[element[i][1][0]], layer[element[i][1][1]])
        sent_len.append(len(e['sentence']))
        rel_cnt.append(len(element)-len(entity))
        rel_layer.append(max(layer))

    a_file = 'data/analysis_test.csv'
    df = pd.DataFrame(list(zip(sent_len, rel_cnt, rel_layer)),
                      columns=['sentence-length', 'relation-count', 'relation layer'])
    df.to_csv(a_file)
    return

    print(set(sent_len), set(rel_cnt), set(rel_layer))
    sent_range = [10, 20, 30, 40, 120]
    sent_l_label = ['(%d,%d]' % (x, y) for x, y in zip([0]+sent_range, sent_range)]
    rel_cr = [1, 2, 3, 5, 20]
    rel_c_label = ['(%d,%d]' % (x, y) for x, y in zip([0]+rel_cr, rel_cr)]
    rel_lr = [1, 2, 3, 4, 5]
    rel_l_label = ['(%d,%d]' % (x, y) for x, y in zip([0]+rel_lr, rel_lr)]
    sent_len_c = [0] * len(sent_range)
    for i in range(len(sent_len)):
        for j in range(len(sent_range)):
            if sent_len[i] <= sent_range[j]:
                sent_len_c[j] += 1
                break

    rel_cnt_c = [0] * len(rel_cr)
    for i in range(len(rel_cnt)):
        for j in range(len(rel_cr)):
            if rel_cnt[i] <= rel_cr[j]:
                rel_cnt_c[j] += 1
                break

    rel_layer_c = [0] * len(rel_lr)
    for i in range(len(rel_layer)):
        for j in range(len(rel_lr)):
            if rel_layer[i] <= rel_lr[j]:
                rel_layer_c[j] += 1
                break
    print([(x, y) for x, y in zip(sent_range, sent_len_c)])
    print([(x, y) for x, y in zip(rel_cr, rel_cnt_c)])
    print([(x, y) for x, y in zip(rel_lr, rel_layer_c)])

    plt.bar([x for x in range(1, len(sent_len_c)+1)], sent_len_c)
    plt.xticks(list(range(1, len(sent_len_c)+1)), sent_l_label)
    plt.xlabel('words in a sentence')
    plt.ylabel('sentences')
    plt.show()

    plt.bar([x for x in range(1, len(rel_cnt_c) + 1)], rel_cnt_c)
    plt.xticks(list(range(1, len(rel_cnt_c) + 1)), rel_c_label)
    plt.xlabel('relations in a sentence')
    plt.ylabel('sentences')
    plt.show()

    plt.bar([x for x in range(1, len(rel_layer_c) + 1)], rel_layer_c)
    plt.xticks(list(range(1, len(rel_layer_c) + 1)), rel_l_label)
    plt.xlabel('relation layers in a sentence')
    plt.ylabel('sentences')
    plt.show()


def my_ob(nums):
    nums[0] = 1


def analysis_few_nk_res(file):
    res = []
    lines = open(file, 'r', encoding='utf-8').readlines()
    cnt = 0
    for i in range(len(lines)):
        if 'test' in lines[i]:
            cnt += 1
            if cnt % 2 == 0:
                res.append(lines[i].strip().split(',')[-1].split(' ')[-1])
    print(len(res), res[:5])
    k_list = ['1', '5', '10']
    model_list = ['BT', 'BL', 'LL', 'BTLow', 'BLLow', 'LLLow'][3:]
    lr_list = ['1e-4', '5e-5', '1e-5', '5e-6']
    cw_list = ['1', '2', '3']
    a_file = os.path.join(os.path.dirname(file), file.split('\\')[-1].split('.')[0]+'.csv')
    writer = open(a_file, 'w', encoding='utf-8')
    cnt = 0
    for lr in range(len(lr_list)):
        l1, l2 = len(model_list)*len(lr_list), len(lr_list)
        writer.write(lr_list[lr] + '\n')
        writer.write('k,' + ','.join(k_list) + '\n')
        for m in range(len(model_list)):
            writer.write(','.join([model_list[m], res[0*l1+m*l2+lr], res[1*l1+m*l2+lr], res[2*l1+m*l2+lr]]) + '\n')
        cw_model = ['BT', 'BTLow'][1:]
        wc = len(k_list)*len(model_list)*len(lr_list)
        l1, l2, l3 = len(k_list)*len(cw_model)*len(lr_list), len(cw_model)*len(lr_list), len(lr_list)
        for m in range(len(cw_model)):
            for cw in range(len(cw_list)):
                writer.write(','.join(['%s-%s' % (cw_model[m], cw_list[cw]),
                                       res[wc + cw * l1 + 0 * l2 + m * l3 + lr],
                                       res[wc + cw * l1 + 1 * l2 + m * l3 + lr],
                                       res[wc + cw * l1 + 2 * l2 + m * l3 + lr]])+'\n')
    
    writer.close()


def convert_layer_res(file):
    layer_gold_num = [394, 209, 61, 7]
    layer_g = [392, 198, 52, 5]
    lines = open(file, 'r', encoding='utf-8').readlines()
    model_res = []  # model name, p, r, f
    cnt = 0
    while cnt < len(lines):
        model_name = lines[cnt].split('|')[1].strip()[:-3]
        print(model_name)
        l1 = [[float(x.strip()) for x in lines[cnt+2+i].strip().split('|')[2:-1]] for i in range(5)]
        l1_cor = [x[1]*layer_gold_num[0] for x in l1]
        l1_cor_g = [x[4]*layer_g[0] for x in l1]
        # print(l1_cor, l1_cor_g)

        l3 = [[float(x.strip()) for x in lines[cnt+16+i].strip().split('|')[2:-1]] for i in range(5)]
        # print(l3)
        l3_cor = [x[1] * layer_gold_num[2] for x in l3]
        l3_pre = [c/x[0] if x[0] != 0 else 0 for c, x in zip(l3_cor, l3)]
        l3_cor_g = [x[4] * layer_g[2] for x in l3]
        l3_pre_g = [c/x[3] if x[3] != 0 else 0 for c, x in zip(l3_cor_g, l3)]
        # print(l3_cor, l3_pre, l3_cor_g, l3_pre_g)

        l4 = [[float(x.strip()) for x in lines[cnt+23+i].strip().split('|')[2:-1]] for i in range(5)]
        # print(l4)
        l4_cor = [x[1]*layer_gold_num[3] for x in l4]
        l4_pre = [c/x[0] if x[0] != 0 else 0 for c, x in zip(l4_cor, l4)]
        l4_cor_g = [x[4]*layer_g[3] for x in l4]
        l4_pre_g = [c/x[3] if x[3] != 0 else 0 for c, x in zip(l4_cor_g, l4)]
        # print(l4_cor, l4_pre, l4_cor_g, l4_pre_g)

        l34_cor = [x1+x2 for x1, x2 in zip(l3_cor, l4_cor)]
        l34_pre = [x1+x2 for x1, x2 in zip(l3_pre, l4_pre)]
        l34_cor_g = [x1+x2 for x1, x2 in zip(l3_cor_g, l4_cor_g)]
        l34_pre_g = [x1+x2 for x1, x2 in zip(l3_pre_g, l4_pre_g)]
        l34_p = [x/y if x > 0 else 0 for x, y in zip(l34_cor, l34_pre)]
        l34_p_g = [x/y if x > 0 else 0 for x, y in zip(l34_cor_g, l34_pre_g)]
        l34_r = [x/sum(layer_gold_num[2:]) for x in l34_cor]
        l34_r_g = [x/sum(layer_g[2:]) for x in l34_cor_g]
        l34_f = [2*x*y/(x+y) for x, y in zip(l34_p, l34_r)]
        l34_f_g = [2*x*y/(x+y) for x, y in zip(l34_p_g, l34_r_g)]
        print(l34_p, l34_p_g, l34_r, l34_r_g, l34_f, l34_f_g)
        model_res.append([model_name, [x for x in zip(*[l34_p, l34_r, l34_f, l34_p_g, l34_r_g, l34_f_g])]])
        cnt += 28
    print(len(model_res))
    res_file = os.path.join(os.path.dirname(file), file.split('\\')[-1].split('.')[0]+'.csv')
    writer = open(res_file, 'w', encoding='utf-8')
    m_name = ['p', 'r', 'f', 'p-g', 'r-g', 'f-g']
    for i in range(len(model_res)):
        writer.write(','.join([model_res[i][0]] + m_name)+'\n')
        # print(len(model_res[i][1]))
        # assert len(m_name) == len(model_res[i][1])
        for j in range(5):
            writer.write(','.join([str(j+1)] + ['%.5f' % x for x in model_res[i][1][j]])+'\n')
        writer.write(
            ','.join(['avg'] + ['%.5f' % (sum([x[idx] for x in model_res[i][1]])/5) for idx in range(len(m_name))])+'\n')
    writer.close()


def get_layer_res(file, get_avg=False):
    res_file = os.path.join(os.path.dirname(file),
                            file.split('\\')[-1].split('.')[0] + '%s.txt' % ('-avg' if get_avg else ''))
    lines = open(file, 'r', encoding='utf-8').readlines()
    l_idx = 0
    line_step = [x-68 for x in [86, 112, 138, 174, 200, 226]]
    model_metric = dict()
    while l_idx < len(lines):
        if 'predict_analysis.csv' in lines[l_idx]:
            model_name = lines[l_idx].split('/')[-2]
            cur_metric = []
            for l_step in line_step:
                cur_metric.append([lines[l_idx+l_step+i_].strip().split()[-1] for i_ in range(4)])
            if model_name in model_metric:
                model_metric[model_name].append(cur_metric)
            else:
                model_metric[model_name] = [cur_metric]
        l_idx += 1
    writer = open(res_file, 'w', encoding='utf-8')
    for item in model_metric.items():
        assert len(item[1]) == 5
        for layer in range(4):
            writer.write('|%s-L%d|\n' % (item[0], layer+1))
            writer.write('---\n')
            for k in range(5):
                writer.write('|%d|%s|\n' % (k, '|'.join([item[1][k][x][layer] for x in range(6)])))
            if get_avg:
                writer.write('|a|%s|\n' % ('|'.join(
                    [str(round(sum([float(item[1][k][x][layer]) for k in range(5)])/5, 6)) for x in range(6)])))
    writer.close()


if __name__ == '__main__':
    
    log_dir = 'E:\master\python\RE\\relation\PLM\\tmp-code'
    
    mn = 'k-BT-c5-1211'
    get_layer_res(os.path.join(log_dir, '%s.out' % mn), get_avg=True)
    get_layer_res(os.path.join(log_dir, '%s.out' % mn))
    convert_layer_res(os.path.join(log_dir, '%s.txt' % mn))
