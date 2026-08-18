import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn

from src.data_loader import MetaphorDataset, get_group_kfold_splits
from src.models import TextOnlyModel, MappingOnlyModel, ParallelMTLModel, ConditionedMTLModel, OracleConditionedModel

def run_experiments(model_name, df, num_classes, seeds=[42], n_splits=5, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    architectures = ["text_only", "mapping_only", "parallel_mtl", "conditioned_mtl", "oracle_mtl"]
    
    # Data structures to return
    results_summary = {arch: {'macro_f1_mean': 0, 'macro_f1_std': 0, 'acc_mean': 0, 'acc_std': 0} for arch in architectures}
    raw_folds = {arch: [] for arch in architectures}
    oof_predictions = {arch: [] for arch in architectures}

    for arch in architectures:
        arch_fold_scores_f1 = []
        arch_fold_scores_acc = []
        
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            splits = get_group_kfold_splits(df, n_splits)
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                train_data = df.iloc[train_idx].reset_index(drop=True)
                val_data = df.iloc[val_idx].reset_index(drop=True)
                
                train_dataset = MetaphorDataset(train_data['text'], train_data['source_lbl'], train_data['target_lbl'], train_data['prag_lbl'], tokenizer, 128)
                val_dataset = MetaphorDataset(val_data['text'], val_data['source_lbl'], val_data['target_lbl'], val_data['prag_lbl'], tokenizer, 128)
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
                
                # Init model
                if arch == "text_only": model = TextOnlyModel(model_name, num_classes['pragmatic'])
                elif arch == "mapping_only": model = MappingOnlyModel(num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                elif arch == "parallel_mtl": model = ParallelMTLModel(model_name, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                elif arch == "conditioned_mtl": model = ConditionedMTLModel(model_name, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                elif arch == "oracle_mtl": model = OracleConditionedModel(model_name, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                
                model = model.to(device)
                optimizer = AdamW(model.parameters(), lr=2e-5)
                
                # Training Loop (Simplified for brevity, implement full training here)
                for epoch in range(epochs):
                    model.train()
                    for batch in train_loader:
                        optimizer.zero_grad()
                        b_input = batch['input_ids'].to(device)
                        b_mask = batch['attention_mask'].to(device)
                        b_s, b_t, b_p = batch['source_label'].to(device), batch['target_label'].to(device), batch['prag_label'].to(device)
                        
                        if arch == "mapping_only": out = model(s_gold=b_s, t_gold=b_t)
                        elif arch == "oracle_mtl": out = model(input_ids=b_input, attention_mask=b_mask, s_gold=b_s, t_gold=b_t)
                        else: out = model(input_ids=b_input, attention_mask=b_mask)
                        
                        loss = nn.CrossEntropyLoss()(out['p_logits'], b_p)
                        loss.backward()
                        optimizer.step()
                
                # Evaluation Loop
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        b_input = batch['input_ids'].to(device)
                        b_mask = batch['attention_mask'].to(device)
                        b_s, b_t, b_p = batch['source_label'].to(device), batch['target_label'].to(device), batch['prag_label'].to(device)
                        
                        if arch == "mapping_only": out = model(s_gold=b_s, t_gold=b_t)
                        elif arch == "oracle_mtl": out = model(input_ids=b_input, attention_mask=b_mask, s_gold=b_s, t_gold=b_t)
                        else: out = model(input_ids=b_input, attention_mask=b_mask)
                        
                        p_preds = torch.argmax(out['p_logits'], dim=1).cpu().numpy()
                        preds.extend(p_preds)
                        trues.extend(b_p.cpu().numpy())
                
                mac_f1 = f1_score(trues, preds, average='macro')
                acc = accuracy_score(trues, preds)
                
                arch_fold_scores_f1.append(mac_f1 * 100)
                arch_fold_scores_acc.append(acc * 100)
                
                # Save Raw Folds Data
                raw_folds[arch].append({
                    'Seed': seed, 'Fold': fold, 'Macro_F1': mac_f1 * 100, 'Accuracy': acc * 100
                })
                
                # Save OOF
                for i in range(len(trues)):
                    oof_predictions[arch].append({
                        'Seed': seed, 'Fold': fold, 'True_Label': trues[i], 'Predicted_Label': preds[i]
                    })
                    
        results_summary[arch]['macro_f1_mean'] = np.mean(arch_fold_scores_f1)
        results_summary[arch]['macro_f1_std'] = np.std(arch_fold_scores_f1)
        results_summary[arch]['acc_mean'] = np.mean(arch_fold_scores_acc)
        results_summary[arch]['acc_std'] = np.std(arch_fold_scores_acc)

    return results_summary, raw_folds, oof_predictions
