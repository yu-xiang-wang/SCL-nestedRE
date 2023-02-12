# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : config.py
# @Software: PyCharm


import os


class Config:
    def __init__(self):
        # data path
        self.data_dir = 'data'  # 
        self.labeled_dir = os.path.join(self.data_dir, 'labeled_data')  # 
        self.vocab_file = os.path.join(self.data_dir, 'vocab.txt')  # 
        self.type_file = os.path.join(self.data_dir, 'type.json')  # 
        self.model_dir = 'model'  # 

        # data param
        self.extra_words = ['[PAD]', '[UNK]', '[CLS]']  # 
        self.max_sent_len = 128  # 
        self.padding = 0  # 
        self.train_prefix = 'train'  # 
        self.valid_prefix = 'valid'  # 
        self.test_prefix = 'test'  # 
        self.generated_prefix = 'g'  # 
        self.generate_type = 2  # 

        # model param
        self.random_seed = 111  
        self.batch_size = 1  
        self.predict_batch = 1  
        self.lr = [1e-4, 1e-5, 5e-6][1]  
        self.p_step = 100  
        self.train_epoch = 40  
        self.device = ['cuda:0', 'cpu'][0]  
        self.early_stop = 20  

        # lstm param
        self.vocab_size = 6000  
        self.embedding_size = 256  
        self.hidden_size = 256  

        # bert params
        self.bert_dir = 'PLM/hfl-chinese-bert-base-wwm-ext'
        self.bert_hidden = 768  
        self.bert_head = 4  
        self.bert_lr = [1e-5, 5e-6][0] 

        # trans param
        self.transformer_layer = 3 
        self.trans_type = 0  # 0: use all info; 1: -w/o position; 2: -w/o op type; 3:

        self.cont_weight = 0.1  

        # entity and relation
        self.entity_type = ['Main', 'Main-q', 'Labor', 'Labor-q', 'Service', 'Place',
                            'Rate', 'RateV', 'Fund', 'FundV', 'Time', 'TimeV', 'Base', 'BaseV']
        self.relation_type = ['>', '>=', '<', '<=', '=', '+', '-', '/', 'has', 'and', '@']
        self.guide = True
        self.use_schema = True  

        # few shot learning
        self.few_ratio = 0.1  
        self.few_scale = 100 
        self.few_labeled_prefix = 'few-labeled'  
        self.few_unlabeled_prefix = 'few-unlabeled'  
        self.few_tea_prefix = 'few-tea' 
        self.do_weak_label = True  

        self.few_n = 5  
        self.few_k = 1  #

        self.few_rel_idx = 3 

        # prompt
        self.max_pro_len = 512  
        self.rel_start = 100  
        self.rel_elem_num = 70  
        self.rel_mode = 2  
        self.rel_place = 0  
        self.template = 2  

        self.use_fgm = False  

        # ===========================================flat data==========================================================
        self.bert_en = 'PLM/bert-base-uncased'  
        self.flat_dir = 'data/flat-relation'  
        self.flat_id = 1  
        self.flat_name = ['kbp37', 'semeval2010'][self.flat_id]  
        self.max_flat_len = 512  

        # overlap data
        self.overlap_dir = 'data/overlap-relation'  
        self.overlap_id = 0  
        self.overlap_name = ['NYT', 'WebNLG'][self.overlap_id]  
        self.max_overlap_len = 256  

        self.math_dir = 'math'
        self.max_math_pl = 512
        self.math_ent_num = 20
        self.math_rel_start = 10 + self.math_ent_num
        self.math_rel_num = 60
        self.prompt_mode = 'auto'



