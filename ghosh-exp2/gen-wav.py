import torch
import torchaudio
import logging
from Encoder import VAEEncoder
from decoder import VAEDecoder
from model import Diffusion
from clip_text_encoder import CLIPTextEncoder
from config import Config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting initialization...")
    
    checkpoint_path = "checkpoint_epoch_0_batch_500.pt"
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Calculate required samples for 5 seconds
    DURATION = 5  # seconds
    TOTAL_SAMPLES = Config.SAMPLE_RATE * DURATION
    logger.info(f"Generating {DURATION} seconds of audio ({TOTAL_SAMPLES} samples)")

    logger.info("Initializing models...")
    vae_encoder = VAEEncoder().to(device)
    vae_decoder = VAEDecoder().to(device)
    diffusion = Diffusion().to(device)
    clip_encoder = CLIPTextEncoder().to(device)

    logger.info("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Checkpoint keys: {checkpoint.keys()}")

    logger.info("Loading state dictionaries...")
    vae_encoder.load_state_dict(checkpoint["encoder_state_dict"])
    vae_decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if "diffusion_state_dict" in checkpoint:
        diffusion.load_state_dict(checkpoint["diffusion_state_dict"])
    if "clip_state_dict" in checkpoint:
        clip_encoder.load_state_dict(checkpoint["clip_state_dict"])

    for name, model in [("VAE Encoder", vae_encoder), ("VAE Decoder", vae_decoder), 
                      ("Diffusion", diffusion), ("CLIP Encoder", clip_encoder)]:
        logger.info(f"Setting {name} to eval mode")
        model.eval()

    text_input = "Bhairavi Thaat"
    logger.info(f"Processing text input: {text_input}")

    with torch.no_grad():
        logger.info("Generating tokens...")
        tokens = torch.randint(0, 5000, (1, 77)).to(device)
        logger.debug(f"Token shape: {tokens.shape}")

        logger.info("Generating text embedding...")
        text_embedding = clip_encoder(tokens)
        logger.debug(f"Text embedding shape: {text_embedding.shape}")
        text_embedding = text_embedding.unsqueeze(0)
        logger.debug(f"Text embedding shape after unsqueeze: {text_embedding.shape}")

        logger.info("Encoding to latent space...")
        mu, logvar = vae_encoder(text_embedding)
        logger.debug(f"mu shape: {mu.shape}, logvar shape: {logvar.shape}")
        latent = vae_encoder.reparameterize(mu, logvar)
        logger.debug(f"Latent shape: {latent.shape}")

        logger.info("Applying diffusion...")
        t = torch.randint(0, diffusion.timesteps, (1,)).to(device)
        refined_latent = diffusion(latent, t)
        logger.debug(f"Refined latent shape: {refined_latent.shape}")

        logger.info("Decoding to waveform...")
        generated_waveform = vae_decoder(refined_latent)
        logger.debug(f"Generated waveform shape before processing: {generated_waveform.shape}")
        
        # Reshape and ensure correct length for 5 seconds
        if len(generated_waveform.shape) > 2:
            generated_waveform = generated_waveform.squeeze(0)
        if len(generated_waveform.shape) == 1:
            generated_waveform = generated_waveform.unsqueeze(0)
        
        # Ensure exactly 5 seconds of audio
        if generated_waveform.shape[1] > TOTAL_SAMPLES:
            generated_waveform = generated_waveform[:, :TOTAL_SAMPLES]
        elif generated_waveform.shape[1] < TOTAL_SAMPLES:
            pad_size = TOTAL_SAMPLES - generated_waveform.shape[1]
            generated_waveform = torch.nn.functional.pad(generated_waveform, (0, pad_size))
        
        logger.debug(f"Final waveform shape: {generated_waveform.shape}")

        logger.info("Saving waveform...")
        output_filename = "generated_bhairavi_5sec.wav"
        torchaudio.save(output_filename, generated_waveform.cpu(), sample_rate=Config.SAMPLE_RATE)

    logger.info(f"Music generated successfully! Saved as {output_filename}")

except Exception as e:
    logger.error(f"Error occurred: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
