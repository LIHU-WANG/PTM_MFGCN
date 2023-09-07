# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import json
import random
import warnings
import numpy as np
import torch
import torch.nn.functional
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from entity_rel_ptm_mfgcn.predicate_entity_textgcn import Bert_Model

warnings.filterwarnings("ignore")


def build_corpus(split, make_vocab=True, data_dir='data'):
    # 读取数据
    assert split in ['entity_data_all', 'entity_data_train', 'entity_data_dev', 'entity_data_test', 'BIO_DATA', 'BIO_DTA', 'NBIO_DATA', 'BIO_DATA_TWO', 'BIO_DATA_THREE']

    step_index = 1
    word_lists, tag_lists, json_data = [], [], []
    ent1_list, rel_list, ent2_list, all_keys = [], [], [], []
    with open(os.path.join('data/neo_face_data_one.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    with open(os.path.join(data_dir, split + '.txt'), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if step_index == 4000:
                break
            if line != '\n':
                word, tag = line.strip('\n').split(' ')
                word_list.append(word)
                tag_list.append(tag)
            else:
                for j_dta in json_data:
                    if ''.join(word_list) == j_dta['text']:
                        ent1_list.append(j_dta['ent1'])
                        rel_list.append(j_dta['rel'])
                        ent2_list.append(j_dta['ent2'])
                        all_keys.append(j_dta['ent1'])
                        all_keys.append(j_dta['ent2'])
                        word_lists.append(word_list)
                        tag_lists.append(tag_list)
                        step_index += 1
                        break
                word_list = []
                tag_list = []

        word_tag = list(zip(ent1_list, rel_list, ent2_list, word_lists, tag_lists))
        random.shuffle(word_tag)
        ent1_list[:], rel_list[:], ent2_list[:], word_lists[:], tag_lists[:] = zip(*word_tag)

        if make_vocab:
            word2id = build_map(word_lists)
            tag2id = build_map(tag_lists)
            key2id = build_map(all_keys, is_key=True)
            rel2id = build_map(rel_list, is_key=True)
            return word_lists, tag_lists, ent1_list, rel_list, ent2_list, word2id, tag2id, key2id, rel2id
        else:
            return word_lists, tag_lists, ent1_list, rel_list, ent2_list


def build_map(lists, is_key=False):
    maps = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    if is_key:
        for list_ in lists:
            if list_ not in maps:
                maps[list_] = len(maps)
        return maps
    else:
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps


class MyDataset(Dataset):
    # dataset 和 data_loader 是配合使用的，一个batch会走一轮data_loader的pro_batch_data函数，否则遍历__getitem__函数
    def __init__(self, datas, tags, ent1s, rels, ent2s, word2id, tag2id, key2id, rel2id):
        self.datas = datas
        self.tags = tags
        self.ent1s = ent1s
        self.rels = rels
        self.ent2s = ent2s
        self.word2id = word2id
        self.tag2id = tag2id
        self.key2id = key2id
        self.rel2id = rel2id

    def __getitem__(self, item):
        data = self.datas[item]
        tag = self.tags[item]
        ent1 = self.ent1s[item]
        rel = self.rels[item]
        ent2 = self.ent2s[item]

        data_index = [self.word2id[i] for i in data]
        tag_index = [self.tag2id[i] for i in tag]
        ent1_index = [self.key2id[ent1]]
        ent2_index = [self.key2id[ent2]]
        rel_index = [self.rel2id[rel]]

        return data_index, tag_index, ent1_index, ent2_index, rel_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self, batch_datas):
        global device
        datas = []
        masks_ner = []
        tags = []
        ent1s = []
        ent2s = []
        rels = []
        batch_lens = []
        segment_ids = []
        batch_lens_rel = []
        for data, tag, ent1, ent2, rel in batch_datas:
            if len(data) > 512:
                data = data[:512]
                tag = tag[:512]
            datas.append(data)
            tags.append(tag)
            ent1s.append(ent1)
            ent2s.append(ent2)
            rels.append(rel)
            batch_lens.append(len(tag))
            masked_tokens = []  # MASK
            segment_ids.append(np.zeros(len(data)).tolist())
            for pos in range(len(data)):
                if random.randrange(0, 100) > 90:  # 10%被替换为其他单词, 不能被替换为 'CLS', 'SEP', 'PAD'
                    mask_dta = random.randrange(4, len(self.word2id) - 1)
                    data[pos] = mask_dta
                    masked_tokens.append(pos)
                elif random.randrange(0, 100) > 85:  # 5%被mask
                    mask_dta = self.word2id['[MASK]']  # make mask
                    data[pos] = mask_dta
                    masked_tokens.append(pos)
            masks_ner.append(masked_tokens)
        batch_max_len = max(batch_lens)
        datas = [i + [self.word2id['[PAD]']] * (batch_max_len - len(i)) for i in datas]
        segment_ids = [i + [self.word2id['[PAD]']] * (batch_max_len - len(i)) for i in segment_ids]
        masks_ner = [i + [self.word2id['[PAD]']] * (batch_max_len - len(i)) for i in masks_ner]
        tags = [i + [self.tag2id['[PAD]']] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas, dtype=torch.int64, device=device), \
               torch.tensor(segment_ids, dtype=torch.int64, device=device), \
               torch.tensor(masks_ner, dtype=torch.int64, device=device), \
               torch.tensor(tags, dtype=torch.long, device=device), \
               torch.tensor(ent1s, dtype=torch.long, device=device), \
               torch.tensor(ent2s, dtype=torch.long, device=device), \
               torch.tensor(rels, dtype=torch.long, device=device)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_file = 'BIO_DATA_TWO'
    word_lists, tag_lists, ent1_list, rel_list, ent2_list, word2id, tag2id, key2id, rel2id = build_corpus(data_file)
    split_len = int(len(word_lists) * 0.65)  # GCN的权重问题，所以要保持每一轮的batch相同，所以要把train和dev都设定成可以整除batch的值
    train_data, train_tag, train_ent1, train_rel, train_ent2 = word_lists[:split_len], tag_lists[:split_len], ent1_list[:split_len], rel_list[:split_len], ent2_list[:split_len]
    dev_data, dev_tag, dev_ent1, dev_rel, dev_ent2 = word_lists[split_len:], tag_lists[split_len:], ent1_list[:split_len], rel_list[:split_len], ent2_list[:split_len]

    epoch = 1000
    batch_size = 2

    train_dataset = MyDataset(train_data, train_tag, train_ent1, train_rel, train_ent2, word2id, tag2id, key2id, rel2id)
    train_data_loader = DataLoader(train_dataset, batch_size, num_workers=0, shuffle=False, collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, dev_ent1, dev_rel, dev_ent2, word2id, tag2id, key2id, rel2id)
    dev_data_loader = DataLoader(dev_dataset, batch_size, num_workers=0, shuffle=False, collate_fn=dev_dataset.pro_batch_data)

    model_bert = Bert_Model(len(word2id), len(key2id), len(rel2id), device=device)
    opt_bert = torch.optim.Adam(model_bert.parameters(), lr=0.0001)
    model_bert = model_bert.to(device)

    text_file = open('logs/' + data_file + '.txt', mode='a', encoding='utf-8')  # 打开文件，文件存在则打开，不存在则创建后再打开
    for e in range(epoch):
        model_bert.train()
        for datas, segment_ids, masks_ner, tags, ent1, ent2, rels in tqdm(train_data_loader):
            train_loss_bert = model_bert.forward(datas, segment_ids, masks_ner, ent1, ent2, bio_tag=tags, rel_tag=rels)
            train_loss_bert.backward()
            opt_bert.step()
            opt_bert.zero_grad()

        model_bert.eval()

        all_ent_tag = []
        all_ent_bert, all_rel_bert = [], []
        for datas, segment_ids, masks_ner, tags, ent1, ent2, rels in tqdm(dev_data_loader):
            all_ent_tag.extend(tags.detach().cpu().numpy().reshape(-1).tolist())

            model_bert.forward(datas, segment_ids, masks_ner, ent1, ent2, bio_tag=tags, rel_tag=rels)
            all_ent_bert.extend(model_bert.ent_pre.detach().cpu().numpy().tolist())

        average_val = 'weighted'  # micro/macro/weighted
        precision_ent_bert = precision_score(all_ent_tag, all_ent_bert, average=average_val)
        recall_ent_bert = recall_score(all_ent_tag, all_ent_bert, average=average_val)
        f1_ent_bert = f1_score(all_ent_tag, all_ent_bert, average=average_val)

        print('-------------------------------', 'epoch：', e, '------------------------------')
        print('precision_ent_bert：', precision_ent_bert, '  recall_ent_bert：', recall_ent_bert, '  f1_ent_bert：', f1_ent_bert)
        print('------------------------------------------------------------------------------')
        dp_json = {'epoch': e, 'precision_ent_bert': precision_ent_bert, 'recall_ent_bert': recall_ent_bert, 'f1_ent_bert': f1_ent_bert}
        text_file.write(str(dp_json) + '\n')
        text_file.flush()
    text_file.close()  # 关闭embedding_log.txt文件

