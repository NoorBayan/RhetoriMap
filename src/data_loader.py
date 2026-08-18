import requests
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

class MetaphorDataset(Dataset):
    def __init__(self, texts, source_labels, target_labels, prag_labels, tokenizer, max_len):
        self.texts = texts
        self.source_labels = source_labels
        self.target_labels = target_labels
        self.prag_labels = prag_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # Using the standard modern __call__ method for transformers tokenizer
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'source_label': torch.tensor(self.source_labels[item], dtype=torch.long),
            'target_label': torch.tensor(self.target_labels[item], dtype=torch.long),
            'prag_label': torch.tensor(self.prag_labels[item], dtype=torch.long)
        }

def prepare_data(data_url):
    print("Fetching data from GitHub...")
    raw_data = requests.get(data_url).json()
    instances = []
    
    for record in raw_data:
        chap_no = record.get('metadata', {}).get('chapter_no')
        verse_no = record.get('metadata', {}).get('verse_no')
        similes = record.get('rhetorical_analysis', {}).get('similes', [])
        
        for sim in similes:
            components = sim.get('components', {})
            functions = sim.get('functions', [])
            
            # Use only PRIMARY function to prevent multi-label statistical issues
            if functions and components.get('source_domain') and components.get('target_domain'):
                instances.append({
                    'verse_uid': f"{chap_no}_{verse_no}",
                    'text': sim.get('simile_identity', {}).get('segment_text', ''),
                    'source': components.get('source_domain'),
                    'target': components.get('target_domain'),
                    'pragmatic': functions[0].get('pragmatic_function_tage')
                })
                
    df = pd.DataFrame(instances)
    
    # Encode Labels
    le_s, le_t, le_p = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['source_lbl'] = le_s.fit_transform(df['source'])
    df['target_lbl'] = le_t.fit_transform(df['target'])
    df['prag_lbl'] = le_p.fit_transform(df['pragmatic'])
    
    num_classes = {
        'source': len(le_s.classes_),
        'target': len(le_t.classes_),
        'pragmatic': len(le_p.classes_)
    }
    
    return df, num_classes

def get_group_kfold_splits(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    # Group by verse_uid to prevent data leakage of multiple metaphors in same verse
    return list(gkf.split(df, groups=df['verse_uid']))
