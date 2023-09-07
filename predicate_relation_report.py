import json
import os
import random
import warnings

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader

from entity_rel_ptm_mfgcn.predicate_relation_textgcn import TextGCN
warnings.filterwarnings("ignore")


# 读取训练集
def read_data(split, data_dir='data'):
    # 读取数据
    assert split in ['nsbd_relations', 'relations_data_all', 'neo_face_data_one', 'neo_face_data', 'relation_all_trp', 'kg_con_relations']
    step_index = 1
    text_list, tags_list, all_keys = [], [], []
    with open(os.path.join(data_dir, split + '.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        random.shuffle(json_data)
        for index, json_dta in enumerate(json_data):
            if step_index == 6000:
                break
            text_list.append(json_dta['ent1']+json_dta['ent2']+json_dta['text'])
            tags_list.append(json_dta['rel'])
            step_index += 1

        word_2_id = build_map(text_list)
        tag_2_id = build_map(tags_list, all_keys=tags_list, is_not_key=False)

    return text_list, tags_list, word_2_id, tag_2_id


def build_map(lists, all_keys=None, is_not_key=True):
    maps = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    if is_not_key:
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps
    else:
        for key in all_keys:
            if key not in maps:
                maps[key] = len(maps)
        return maps


class MyDataset(Dataset):

    def __init__(self, texts, tags, word_2_id, tag_2_id):
        self.texts = texts
        self.tags = tags
        self.word_2_id = word_2_id
        self.tag_2_id = tag_2_id

    def __getitem__(self, item):
        text = self.texts[item]
        tag = self.tags[item]

        data_index = [self.word_2_id[i] for i in list(text)]
        tag_index = [self.tag_2_id[tag]]

        return data_index, tag_index

    def __len__(self):
        assert len(self.texts) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self, batch_datas):
        global device
        datas, tags, mask_rel, batch_lens = [], [], [], []
        for data, tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        batch_max_len = max(batch_lens)

        datas = [i + [self.word_2_id['[PAD]']] * (batch_max_len - len(i)) for i in datas]

        return torch.tensor(datas, dtype=torch.int64, device=device), \
               torch.tensor(tags, dtype=torch.long, device=device)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_file = 'relations_data_all'   # kg_con_relations    relations_data_all
    text_list, tags_list, word_2_id, tag_2_id = read_data(data_file)
    split_len = int(len(tags_list) * 0.65)  # 由于GCN的特殊性，要同时保证train和dev同时被batch_len整除
    train_text, train_tag = text_list[:split_len], tags_list[:split_len]
    dev_text, dev_tag = text_list[split_len:], tags_list[split_len:]

    epoch = 1000
    batch_size = 1
    embedding_num = 768
    hidden_num = 200
    train_dataset = MyDataset(train_text, train_tag, word_2_id, tag_2_id)
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_text, dev_tag, word_2_id, tag_2_id)
    dev_data_loader = DataLoader(dev_dataset, batch_size, shuffle=False, collate_fn=dev_dataset.pro_batch_data)

    model_textgcn = TextGCN(len(word_2_id), embedding_num, hidden_num, len(tag_2_id))
    opt_textgcn = torch.optim.Adam(model_textgcn.parameters(), lr=0.00001)
    model_textgcn = model_textgcn.to(device)

    # text_file = open('logs/relation' + data_file + '.txt', mode='a', encoding='utf-8')  # 打开文件，文件存在则打开，不存在则创建后再打开
    for e in range(epoch):
        model_textgcn.train()
        for batch_datas, batch_tags in tqdm(train_data_loader):
            train_loss_textgcn = model_textgcn.forward(batch_datas, batch_tags, device=device)
            train_loss_textgcn.backward()
            opt_textgcn.step()
            opt_textgcn.zero_grad()

        model_textgcn.eval()

        all_tag = []
        all_pre_textgcn = []
        for dev_batch_datas, dev_batch_tags in tqdm(dev_data_loader):
            all_tag.extend(dev_batch_tags.detach().cpu().numpy().reshape(-1).tolist())

            model_textgcn.forward(dev_batch_datas, dev_batch_tags, device=device)
            all_pre_textgcn.extend(model_textgcn.pre.detach().cpu().numpy().tolist())

        average_val = 'weighted'  # micro/macro/weighted
        precision_text_gcn = precision_score(all_tag, all_pre_textgcn, average=average_val)

        recall_text_gcn = recall_score(all_tag, all_pre_textgcn, average=average_val)

        f1_text_gcn = f1_score(all_tag, all_pre_textgcn, average=average_val)

        print('-------------------------------', 'epoch：', e, '------------------------------')
        print('precision_text_gcn：', precision_text_gcn, '  recall_text_gcn：', recall_text_gcn, '  f1_text_gcn：', f1_text_gcn)
        print('------------------------------------------------------------------------------')

    #
    #     text_file.write(str(dp_json) + '\n')
    #     text_file.flush()
    # text_file.close()  # 关闭embedding_log.txt文件
    #
