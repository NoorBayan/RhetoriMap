import torch

class Config:
    # Data
    DATA_URL = "https://raw.githubusercontent.com/NoorBayan/Burhan/main/corpus/metaphors_data.json"
    
    # Model
    MODEL_NAME = "UBC-NLP/ARBERT"
    MAX_LEN = 128
    
    # Training Parameters
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    
    # Cross Validation
    N_SPLITS = 5
    SEEDS = [42, 100, 2024] # 3 Random seeds for robustness
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loss Weights (For MTL)
    LAMBDA_S = 1.0
    LAMBDA_T = 1.0
    LAMBDA_P = 1.0
