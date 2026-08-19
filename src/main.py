import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score

# افتراض أن هذه الملفات موجودة في مشروعك بنفس الهيكلية
from src.data_loader import MetaphorDataset, get_group_kfold_splits
from src.models import TextOnlyModel, MappingOnlyModel, ParallelMTLModel, ConditionedMTLModel, OracleConditionedModel

def run_experiments(model_name, df, num_classes, seeds=[42], n_splits=5, epochs=5):
    """
    Main experimental pipeline to evaluate all architectures robustly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # المعماريات الخاضعة للاختبار التجريبي
    architectures = ["text_only", "mapping_only", "parallel_mtl", "conditioned_mtl", "oracle_mtl"]
    
    # قواميس لحفظ النتائج لضمان قابلية إعادة الإنتاج (Reproducibility)
    results_summary = {arch: {'macro_f1_mean': 0, 'macro_f1_std': 0, 'acc_mean': 0, 'acc_std': 0} for arch in architectures}
    raw_folds = {arch: [] for arch in architectures}
    oof_predictions = {arch: [] for arch in architectures}

    # معاملات وزن الخسائر للمهام المتعددة (MTL Loss Weights)
    lambda_s, lambda_t, lambda_p = 1.0, 1.0, 1.0

    for arch in architectures:
        print(f"\n{'='*50}\n🚀 Training Architecture: {arch.upper()}\n{'='*50}")
        arch_fold_scores_f1 = []
        arch_fold_scores_acc = []
        
        for seed in seeds:
            # [SCIENTIFIC RIGOR] تثبيت البذور العشوائية بالكامل لضمان تطابق التجربة
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # تقسيم البيانات باستخدام GroupKFold لمنع تسرب البيانات (Leakage)
            splits = get_group_kfold_splits(df, n_splits)
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                print(f"   🌱 Seed: {seed} | 📂 Fold: {fold+1}/{n_splits}")
                
                train_data = df.iloc[train_idx].reset_index(drop=True)
                val_data = df.iloc[val_idx].reset_index(drop=True)
                
                train_dataset = MetaphorDataset(train_data['text'], train_data['source_lbl'], train_data['target_lbl'], train_data['prag_lbl'], tokenizer, 128)
                val_dataset = MetaphorDataset(val_data['text'], val_data['source_lbl'], val_data['target_lbl'], val_data['prag_lbl'], tokenizer, 128)
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
                
                # تهيئة النموذج المناسب
                if arch == "text_only": 
                    model = TextOnlyModel(model_name, num_classes['pragmatic'])
                elif arch == "mapping_only": 
                    model = MappingOnlyModel(num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                elif arch == "parallel_mtl": 
                    model = ParallelMTLModel(model_name, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                elif arch == "conditioned_mtl": 
                    model = ConditionedMTLModel(model_name, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                elif arch == "oracle_mtl": 
                    model = OracleConditionedModel(model_name, num_classes['source'], num_classes['target'], num_classes['pragmatic'])
                
                model = model.to(device)
                optimizer = AdamW(model.parameters(), lr=2e-5)
                criterion = nn.CrossEntropyLoss()
                
                # ==========================================
                # TRAINING LOOP
                # ==========================================
                for epoch in range(epochs):
                    model.train()
                    for batch in train_loader:
                        optimizer.zero_grad()
                        
                        # نقل البيانات للمعالج (GPU/CPU)
                        b_input = batch['input_ids'].to(device)
                        b_mask = batch['attention_mask'].to(device)
                        b_s = batch['source_label'].to(device)
                        b_t = batch['target_label'].to(device)
                        b_p = batch['prag_label'].to(device)
                        
                        # [CRITICAL FIX]: التمرير الأمامي وحساب الخسارة الصحيح لكل معمارية
                        if arch == "mapping_only":
                            out = model(s_gold=b_s, t_gold=b_t)
                            loss = criterion(out['p_logits'], b_p)
                            
                        elif arch == "oracle_mtl":
                            out = model(input_ids=b_input, attention_mask=b_mask, s_gold=b_s, t_gold=b_t)
                            loss = criterion(out['p_logits'], b_p)
                            
                        elif arch in ["parallel_mtl", "conditioned_mtl"]:
                            out = model(input_ids=b_input, attention_mask=b_mask)
                            loss_p = criterion(out['p_logits'], b_p)
                            loss_s = criterion(out['s_logits'], b_s)
                            loss_t = criterion(out['t_logits'], b_t)
                            # دمج خسائر المهام المتعددة لتحديث أوزان المشفر (Encoder) والرؤوس المساعدة
                            loss = (lambda_p * loss_p) + (lambda_s * loss_s) + (lambda_t * loss_t)
                            
                        else: # text_only
                            out = model(input_ids=b_input, attention_mask=b_mask)
                            loss = criterion(out['p_logits'], b_p)
                        
                        loss.backward()
                        # [SCIENTIFIC RIGOR] تقليم التدرج لمنع انفجار الأوزان في التعليم المتعدد
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # ==========================================
                # EVALUATION LOOP (Out-Of-Fold Predictions)
                # ==========================================
                model.eval()
                preds, trues = [], []
                
                with torch.no_grad():
                    for batch in val_loader:
                        b_input = batch['input_ids'].to(device)
                        b_mask = batch['attention_mask'].to(device)
                        b_s = batch['source_label'].to(device)
                        b_t = batch['target_label'].to(device)
                        b_p = batch['prag_label'].to(device)
                        
                        # استخراج التنبؤات بناءً على المعمارية
                        if arch == "mapping_only": 
                            out = model(s_gold=b_s, t_gold=b_t)
                        elif arch == "oracle_mtl": 
                            out = model(input_ids=b_input, attention_mask=b_mask, s_gold=b_s, t_gold=b_t)
                        else: 
                            out = model(input_ids=b_input, attention_mask=b_mask)
                        
                        # نستهدف الوظيفة التداولية (Pragmatic Function) للتقييم النهائي
                        p_preds = torch.argmax(out['p_logits'], dim=1).cpu().numpy()
                        preds.extend(p_preds)
                        trues.extend(b_p.cpu().numpy())
                
                # حساب المقاييس
                mac_f1 = f1_score(trues, preds, average='macro')
                acc = accuracy_score(trues, preds)
                
                arch_fold_scores_f1.append(mac_f1 * 100)
                arch_fold_scores_acc.append(acc * 100)
                
                # حفظ البيانات الخام للتحليل الإحصائي
                raw_folds[arch].append({
                    'Seed': seed, 'Fold': fold, 'Macro_F1': mac_f1 * 100, 'Accuracy': acc * 100
                })
                
                # حفظ التنبؤات (OOF) لبناء مصفوفة الارتباك لاحقاً
                for i in range(len(trues)):
                    oof_predictions[arch].append({
                        'Seed': seed, 'Fold': fold, 'True_Label': trues[i], 'Predicted_Label': preds[i]
                    })
                    
        # حساب المتوسط والانحراف المعياري للمقاييس عبر كل البذور والطيات
        results_summary[arch]['macro_f1_mean'] = np.mean(arch_fold_scores_f1)
        results_summary[arch]['macro_f1_std'] = np.std(arch_fold_scores_f1)
        results_summary[arch]['acc_mean'] = np.mean(arch_fold_scores_acc)
        results_summary[arch]['acc_std'] = np.std(arch_fold_scores_acc)
        
        print(f"🏁 Finished {arch} | Macro-F1: {results_summary[arch]['macro_f1_mean']:.2f}%")

    return results_summary, raw_folds, oof_predictions
