import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def train_epoch(model, data_loader, optimizer, scheduler, device, model_type, config):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_s = batch['source_label'].to(device)
        labels_t = batch['target_label'].to(device)
        labels_p = batch['prag_label'].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        if model_type == "text_only":
            loss = criterion(outputs['p_logits'], labels_p)
        else: # MTL models
            loss_s = criterion(outputs['s_logits'], labels_s)
            loss_t = criterion(outputs['t_logits'], labels_t)
            loss_p = criterion(outputs['p_logits'], labels_p)
            loss = (config.LAMBDA_S * loss_s) + (config.LAMBDA_T * loss_t) + (config.LAMBDA_P * loss_p)
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    preds_p, true_p = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_p = batch['prag_label'].numpy()
            
            outputs = model(input_ids, attention_mask)
            logits_p = outputs['p_logits'].detach().cpu().numpy()
            
            preds_p.extend(np.argmax(logits_p, axis=1))
            true_p.extend(labels_p)
            
    macro_f1 = f1_score(true_p, preds_p, average='macro')
    acc = accuracy_score(true_p, preds_p)
    return macro_f1, acc
