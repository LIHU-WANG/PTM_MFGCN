import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
bert_path = 'pre_train_model/our_pre_model/bert_our.h5'

# BERT模型参数
maxlen = 512  # 表示同一个batch中的所有句子都由30个token组成，不够的补PAD（这里我实现的方式比较粗暴，直接固定所有batch中的所有句子都为30）
batch_size = 6
max_pred = 20  # 表示最多需要预测多少个单词，即BERT中的完形填空任务
n_layers = 6  # 表示Encoder Layer的数量
n_heads = 12
d_model = 1024  # 表示Token Embeddings、Segment Embeddings、Position Embeddings的维度
d_ff = d_model * 4  # 4*d_model, FeedForward dimension 表示Encoder Layer中全连接层的维度
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2  # 表示Decoder input由几句话组成
vocab_size = 4128


# 模型构建
def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class EncoderLayer(nn.Module):
    def __init__(self, device='cpu'):
        super(EncoderLayer, self).__init__()
        self.device = device
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        residual, batch_size = enc_inputs, enc_inputs.size(0)
        q_s = self.W_Q(enc_inputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(enc_inputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(enc_inputs).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = enc_self_attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = nn.Linear(n_heads * d_v, d_model).to(self.device)
        output = output(context)
        lnd = nn.LayerNorm(d_model).to(self.device)
        enc_outputs = lnd(output + residual)

        enc_outputs = self.fc2(gelu(self.fc1(enc_outputs)))
        return enc_outputs


class BERT(nn.Module):
    def __init__(self, device='cpu'):
        super(BERT, self).__init__()
        self.device = device
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([EncoderLayer(device=device) for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )

        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu

        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = self.tok_embed.weight

    def forward(self, input_ids, segment_ids, masked_pos):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(input_ids).to(self.device)
        embedding = self.tok_embed(input_ids) + self.pos_embed(pos) + self.seg_embed(segment_ids)
        output = self.norm(embedding)

        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)

        output_dta = output
        h_pooled = self.fc(output[:, 0])
        logits_clsf = self.classifier(h_pooled)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.activ2(self.linear(h_masked))
        logits_lm = self.fc2(h_masked)
        return output_dta, logits_lm, logits_clsf


class Bert_Model(nn.Module):
    def __init__(self, num_corpus, num_entities, num_relations, embedding_dim=1024, embedding_dim_rel=200, hidden_size=794, hidden_size_ent=200,  input_drop=0.2, hidden_drop=0.3, feat_drop=0.2, bi=True, use_bias=True, device='cpu'):
        super().__init__()

        self.encoder = BERT(device=device)
        self.encoder.load_state_dict(torch.load(bert_path))
        self.lstm = nn.LSTM(embedding_dim, hidden_size_ent, batch_first=True, bidirectional=bi)

        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.embedding_dim_rel = embedding_dim_rel

        self.emb_e1 = torch.nn.Embedding(num_entities, embedding_dim_rel, padding_idx=0)
        self.emb_e2 = torch.nn.Embedding(num_entities, embedding_dim_rel, padding_idx=0)

        self.conv1d = torch.nn.Conv1d(1, 32, (3,), 1, 0, bias=use_bias)
        self.conv1d_1 = torch.nn.Conv1d(32, 32, (3,), 1, 0, bias=use_bias)

        if bi:
            self.fc_ner = nn.Linear(hidden_size_ent * 2, num_corpus)
        else:
            self.fc_ner = nn.Linear(hidden_size_ent, num_corpus)

        self.fc_rel = nn.Linear(hidden_size, num_relations)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, text, segment_ids, attention_mask, head, tail, bio_tag=None, rel_tag=None):
        text_emb = self.encoder(text, segment_ids, attention_mask)[0]
        out, _ = self.lstm(text_emb)

        ent_pred = self.fc_ner(out)
        self.ent_pre = torch.argmax(ent_pred, dim=-1).reshape(-1)
        loss_ent = self.cross_loss(ent_pred.reshape(-1, ent_pred.shape[-1]), bio_tag.reshape(-1))

        return loss_ent

