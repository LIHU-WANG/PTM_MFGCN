import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TextGCN(nn.Module):

    def __init__(self, corpus_num, embedding_num, hidden_num, class_num, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(corpus_num, embedding_num)

        self.start_gcn = GCNConv(embedding_num, embedding_num)
        self.layers = nn.ModuleList([GCNConv(embedding_num, embedding_num) for _ in range(n_layers)])
        self.end_gcn = GCNConv(embedding_num, hidden_num)

        self.fc = nn.Linear(hidden_num, class_num)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, batch_datas, batch_tags=None, device=None):
        batch_datas = batch_datas[-1, :]
        edge_index, edge_wight = self.build_adj(batch_datas, device=device)
        embedding_all = self.embedding(batch_datas)

        out = F.relu(self.start_gcn(embedding_all, edge_index, edge_wight))
        for l_index, layer in enumerate(self.layers):
            out = F.relu(layer(out, edge_index, edge_wight*(l_index+2)+edge_wight**(l_index+2)))
        out = self.end_gcn(out, edge_index, edge_wight*(len(self.layers)+2)+edge_wight**(len(self.layers)+2))

        pre = self.fc(out[-1, :])
        self.pre = torch.argmax(pre, dim=-1).reshape(-1)
        if batch_tags is not None:
            loss = self.cross_loss(pre.reshape(-1, pre.shape[-1]), batch_tags.reshape(-1))
            return loss

    def build_adj(self, batch_data, device='cpu'):
        n_l = len(batch_data) - 1
        edge_index = np.zeros((2, n_l))
        edge_wight = np.ones(n_l)
        step_index = 0
        for i in range(0, n_l):
            edge_index[0][i], edge_index[1][i] = step_index, step_index + 1
            edge_wight[i] = 1
            step_index += 1
        return torch.tensor(edge_index, dtype=torch.long).to(device), torch.tensor(edge_wight, dtype=torch.float).to(device)

