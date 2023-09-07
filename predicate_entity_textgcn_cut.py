import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

class Bert_Model(nn.Module):
    def __init__(self, num_corpus, num_entities, num_relations, embedding_dim=1024, embedding_dim_rel=200, hidden_size=794, hidden_size_ent=200,  input_drop=0.2, hidden_drop=0.3, feat_drop=0.2, bi=True, use_bias=True, device='cpu'):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_corpus, embedding_dim_rel, padding_idx=0)

        self.fc_ner = nn.Linear(hidden_size_ent, num_corpus)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, text, segment_ids, attention_mask, head, tail, bio_tag=None, rel_tag=None):
        text_emb = self.encoder(text)

        ent_pred = self.fc_ner(text_emb)
        self.ent_pre = torch.argmax(ent_pred, dim=-1).reshape(-1)
        loss_ent = self.cross_loss(ent_pred.reshape(-1, ent_pred.shape[-1]), bio_tag.reshape(-1))

        return loss_ent

