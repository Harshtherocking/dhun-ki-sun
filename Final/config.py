import torch

# General settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../data/spectrograms"
OUTPUT_DIR = "../output"
CHECKPOINT_DIR = "../checkpoints"

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
SAMPLE_RATE = 16000
N_MELS = 128
LATENT_DIM = 512
NUM_WORKERS = 4  # Number of data loading workers

# Model configurations
MODEL_CONFIG = {
    "encoder_dim": LATENT_DIM,
    "decoder_dim": LATENT_DIM,
    "diffusion_steps": 1000,
    "attention_heads": 8,
    "hidden_dim": 1024,
}

# Logging & Checkpoints
LOG_INTERVAL = 10
SAVE_INTERVAL = 5