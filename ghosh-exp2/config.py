class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_LENGTH = 55125  # Reduced from 110250
    
    # Model dimensions
    LATENT_DIM = 64  # Reduced from 128
    HIDDEN_DIM = 256
    
    # Training parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    BETA = 0.1  # KL divergence weight
    
    # Optimization
    GRADIENT_CHECKPOINT = True
    MIXED_PRECISION = True
    
    # Architecture
    ATTENTION_HEADS = 4
    TRANSFORMER_LAYERS = 4
    
    # Memory optimization
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True
    ATTENTION_SLICE_SIZE = 4  # For memory-efficient attention computation
