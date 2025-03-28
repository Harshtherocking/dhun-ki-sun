import torch
import os
import collections

def load_from_standard_weights(ckpt_path, device):
    """
    Loads and converts model weights for spectrogram-based diffusion.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    converted = {"diffusion": {}, "encoder": {}, "decoder": {}, "clip": {}}

    # Convert diffusion model weights efficiently
    diffusion_keys = [
        ("time_embedding.linear_1.weight", "model.diffusion_model.time_embed.0.weight"),
        ("time_embedding.linear_1.bias", "model.diffusion_model.time_embed.0.bias"),
        ("time_embedding.linear_2.weight", "model.diffusion_model.time_embed.2.weight"),
        ("time_embedding.linear_2.bias", "model.diffusion_model.time_embed.2.bias"),
    ]
    for new_key, old_key in diffusion_keys:
        if old_key in state_dict:
            converted["diffusion"][new_key] = state_dict[old_key]

    # Convert encoder/decoder layers dynamically
    for i in range(6):  # Adjust based on actual model depth
        enc_key_w = f"first_stage_model.encoder.down.{i}.conv.weight"
        enc_key_b = f"first_stage_model.encoder.down.{i}.conv.bias"
        dec_key_w = f"first_stage_model.decoder.up.{i}.conv.weight"
        dec_key_b = f"first_stage_model.decoder.up.{i}.conv.bias"
        
        if enc_key_w in state_dict:
            converted["encoder"][f"enc_{i}.weight"] = state_dict[enc_key_w]
        if enc_key_b in state_dict:
            converted["encoder"][f"enc_{i}.bias"] = state_dict[enc_key_b]
        if dec_key_w in state_dict:
            converted["decoder"][f"dec_{i}.weight"] = state_dict[dec_key_w]
        if dec_key_b in state_dict:
            converted["decoder"][f"dec_{i}.bias"] = state_dict[dec_key_b]

    # Convert CLIP model weights
    clip_keys = [
        ("embedding.token_embedding.weight", "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"),
        ("embedding.position_embedding", "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"),
    ]
    for new_key, old_key in clip_keys:
        if old_key in state_dict:
            converted["clip"][new_key] = state_dict[old_key]

    return converted

def save_optimized_checkpoint(model_dict, save_path):
    """
    Saves an optimized checkpoint with reduced precision for efficiency.
    """
    optimized_dict = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in model_dict.items()}
    torch.save(optimized_dict, save_path)
    print(f"Optimized checkpoint saved at: {save_path}")

def convert_weights_for_spectrogram_model(ckpt_path, save_path, device="cuda"):
    """
    Converts, optimizes, and saves model weights for spectrogram-based diffusion.
    """
    state_dict = load_from_standard_weights(ckpt_path, device)
    save_optimized_checkpoint(state_dict, save_path)
    return state_dict
