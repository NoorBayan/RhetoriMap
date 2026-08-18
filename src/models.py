import torch
import torch.nn as nn
from transformers import AutoModel

class TextOnlyModel(nn.Module):
    """Model 0: X -> P"""
    def __init__(self, model_name, num_p):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.p_head = nn.Linear(self.encoder.config.hidden_size, num_p)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return {'p_logits': self.p_head(out.pooler_output)}

class MappingOnlyModel(nn.Module):
    """Model 1: M(S,T) -> P (No text)"""
    def __init__(self, num_s, num_t, num_p, embed_dim=128):
        super().__init__()
        self.s_embed = nn.Embedding(num_s, embed_dim)
        self.t_embed = nn.Embedding(num_t, embed_dim)
        self.p_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_p)
        )

    def forward(self, s_gold, t_gold, **kwargs):
        s_vec = self.s_embed(s_gold)
        t_vec = self.t_embed(t_gold)
        combined = torch.cat((s_vec, t_vec), dim=1)
        return {'p_logits': self.p_head(combined)}

class ParallelMTLModel(nn.Module):
    """Model 3: X -> (S, T, P)"""
    def __init__(self, model_name, num_s, num_t, num_p):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h_size = self.encoder.config.hidden_size
        self.s_head = nn.Linear(h_size, num_s)
        self.t_head = nn.Linear(h_size, num_t)
        self.p_head = nn.Linear(h_size, num_p)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return {'s_logits': self.s_head(out), 't_logits': self.t_head(out), 'p_logits': self.p_head(out)}

class ConditionedMTLModel(nn.Module):
    """Model 4: X -> M, then (X + M_hat) -> P"""
    def __init__(self, model_name, num_s, num_t, num_p):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h_size = self.encoder.config.hidden_size
        self.s_head = nn.Linear(h_size, num_s)
        self.t_head = nn.Linear(h_size, num_t)
        self.p_head = nn.Sequential(
            nn.Linear(h_size + num_s + num_t, h_size // 2),
            nn.ReLU(),
            nn.Linear(h_size // 2, num_p)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        s_log = self.s_head(out)
        t_log = self.t_head(out)
        s_prob, t_prob = torch.softmax(s_log, dim=1), torch.softmax(t_log, dim=1)
        combined = torch.cat((out, s_prob, t_prob), dim=1)
        return {'s_logits': s_log, 't_logits': t_log, 'p_logits': self.p_head(combined)}

class OracleConditionedModel(nn.Module):
    """Model 5: (X + M_gold) -> P"""
    def __init__(self, model_name, num_s, num_t, num_p):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h_size = self.encoder.config.hidden_size
        self.p_head = nn.Sequential(
            nn.Linear(h_size + num_s + num_t, h_size // 2),
            nn.ReLU(),
            nn.Linear(h_size // 2, num_p)
        )
        self.num_s, self.num_t = num_s, num_t

    def forward(self, input_ids, attention_mask, s_gold, t_gold, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        s_onehot = torch.nn.functional.one_hot(s_gold, num_classes=self.num_s).float()
        t_onehot = torch.nn.functional.one_hot(t_gold, num_classes=self.num_t).float()
        combined = torch.cat((out, s_onehot, t_onehot), dim=1)
        return {'p_logits': self.p_head(combined)}
