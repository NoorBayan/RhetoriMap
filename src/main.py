import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from config import Config
from data_loader import prepare_data, get_group_kfold_splits, MetaphorDataset
from models import TextOnlyModel, ParallelMTLModel, ConditionedMTLModel
from trainer import train_epoch, eval_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    print("=== Empirical Investigation of Predictive Dependencies ===")
    df, num_classes = prepare_data(Config.DATA_URL)
    print(f"Data loaded: {len(df)} instances.")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    splits = get_group_kfold_splits(df, Config.N_SPLITS)
    
    model_types = ["text_only", "parallel_mtl", "conditioned_mtl"]
    final_results = {mt: [] for mt in model_types}
    
    for seed in Config.SEEDS:
        print(f"\n--- Running Seed: {seed} ---")
        set_seed(seed)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"  Fold {fold+1}/{Config.N_SPLITS}")
            
            train_data = df.iloc[train_idx].reset_index(drop=True)
            val_data = df.iloc[val_idx].reset_index(drop=True)
            
            train_dataset = MetaphorDataset(train_data['text'], train_data['source_lbl'], train_data['target_lbl'], train_data['prag_lbl'], tokenizer, Config.MAX_LEN)
            val_dataset = MetaphorDataset(val_data['text'], val_data['source_lbl'], val_data['target_lbl'], val_data['prag_lbl'], tokenizer, Config.MAX_LEN)
            
            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
            
            for m_type in model_types:
                # Initialize Model
                if m_type == "text_only":
                    model = TextOnlyModel(Config.MODEL_NAME, num_classes['pragmatic'])
                elif m_type == "parallel_mtl":
                    model = ParallelMTLModel(Config.MODEL_NAME, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                elif m_type == "conditioned_mtl":
                    model = ConditionedMTLModel(Config.MODEL_NAME, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                
                model = model.to(Config.DEVICE)
                
                # Optimizer & Scheduler
                optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
                total_steps = len(train_loader) * Config.EPOCHS
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
                
                best_f1 = 0
                for epoch in range(Config.EPOCHS):
                    train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE, m_type, Config)
                    val_f1, val_acc = eval_model(model, val_loader, Config.DEVICE)
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                
                final_results[m_type].append(best_f1)
                print(f"    [{m_type}] Best Val Macro-F1: {best_f1:.4f}")

    print("\n" + "="*50)
    print("FINAL RESULTS (Averaged across Seeds and Folds)")
    print("="*50)
    for m_type in model_types:
        mean_f1 = np.mean(final_results[m_type]) * 100
        std_f1 = np.std(final_results[m_type]) * 100
        print(f"{m_type.upper().ljust(20)}: {mean_f1:.2f}% ± {std_f1:.2f}%")

if __name__ == "__main__":
    main()
