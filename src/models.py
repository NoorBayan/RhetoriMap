import torch
import torch.nn as nn
from transformers import AutoModel

class TextOnlyModel(nn.Module):
    """Model 0: X -> P (Baseline)"""
    def __init__(self, model_name, num_p):
        super(TextOnlyModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.p_head = nn.Linear(self.encoder.config.hidden_size, num_p)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        return {'p_logits': self.p_head(pooled_output)}


class ParallelMTLModel(nn.Module):
    """Model 3: X -> (S, T, P) simultaneously (Shared Encoder Baseline)"""
    def __init__(self, model_name, num_s, num_t, num_p):
        super(ParallelMTLModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.encoder.config.hidden_size
        
        self.s_head = nn.Linear(hidden_size, num_s)
        self.t_head = nn.Linear(hidden_size, num_t)
        self.p_head = nn.Linear(hidden_size, num_p)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        return {
            's_logits': self.s_head(pooled_output),
            't_logits': self.t_head(pooled_output),
            'p_logits': self.p_head(pooled_output)
        }


class ConditionedMTLModel(nn.Module):
    """Model 4: X -> (S, T), then (X + S_prob + T_prob) -> P (Proposed Framework)"""
    def __init__(self, model_name, num_s, num_t, num_p):
        super(ConditionedMTLModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.encoder.config.hidden_size
        
        self.s_head = nn.Linear(hidden_size, num_s)
        self.t_head = nn.Linear(hidden_size, num_t)
        
        # Conditioned Head: takes hidden state + S probabilities + T probabilities
        cond_input_size = hidden_size + num_s + num_t
        self.p_head_cond = nn.Sequential(
            nn.Linear(cond_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_p)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        s_logits = self.s_head(pooled_output)
        t_logits = self.t_head(pooled_output)
        
        # Convert logits to probabilities for semantic conditioning
        s_probs = torch.softmax(s_logits, dim=-1)
        t_probs = torch.softmax(t_logits, dim=-1)
        
        # Concatenate: H + S_probs + T_probs
        cond_representation = torch.cat((pooled_output, s_probs, t_probs), dim=1)
        p_logits = self.p_head_cond(cond_representation)
        
        return {
            's_logits': s_logits,
            't_logits': t_logits,
            'p_logits': p_logits
        }
