import torch
import torchaudio
import logging
import os
from Encoder import VAEEncoder
from decoder import VAEDecoder
from model import Diffusion
from clip_text_encoder import CLIPTextEncoder
from config import Config

# ====== Logging Setup ======
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    logger.info("🎵 Starting song generation pipeline...")

    # Path to folder containing .pt checkpoint files
    checkpoint_folder = "/home/amithk/Desktop/Dhun_ki_sun/Gmodel/epoch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Constants for audio generation
    DURATION = 5  # Each clip is 5 seconds
    SAMPLE_RATE = Config.SAMPLE_RATE
    TOTAL_SAMPLES = SAMPLE_RATE * DURATION
    final_audio = []  # Store generated waveforms for full song

    # List all .pt files in the checkpoint folder
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_folder) if f.endswith(".pt")])

    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the folder!")

    logger.info(f"Found {len(checkpoint_files)} checkpoint files: {checkpoint_files}")

    # ====== Model Initialization ======
    logger.info("Initializing models...")
    vae_encoder = VAEEncoder().to(device)
    vae_decoder = VAEDecoder().to(device)
    diffusion = Diffusion().to(device)
    clip_encoder = CLIPTextEncoder().to(device)

    # ====== Loop Over Each Checkpoint ======
    for i, checkpoint_file in enumerate(checkpoint_files):
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        logger.info(f"🔄 Loading checkpoint: {checkpoint_file}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        vae_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        vae_decoder.load_state_dict(checkpoint["decoder_state_dict"])
        if "diffusion_state_dict" in checkpoint:
            diffusion.load_state_dict(checkpoint["diffusion_state_dict"])
        if "clip_state_dict" in checkpoint:
            clip_encoder.load_state_dict(checkpoint["clip_state_dict"])

        # Set models to evaluation mode
        for name, model in [("VAE Encoder", vae_encoder), ("VAE Decoder", vae_decoder), 
                            ("Diffusion", diffusion), ("CLIP Encoder", clip_encoder)]:
            logger.info(f"Setting {name} to eval mode")
            model.eval()

        # ====== Generate Audio Clip ======
        text_input = f"Thaat {i+1}"  # Example input (modify as needed)
        logger.info(f"Generating music for: {text_input}")

        with torch.no_grad():
            tokens = torch.randint(0, 5000, (1, 77)).to(device)  # Simulate tokenized input
            text_embedding = clip_encoder(tokens).unsqueeze(0)

            mu, logvar = vae_encoder(text_embedding)
            latent = vae_encoder.reparameterize(mu, logvar)

            t = torch.randint(0, diffusion.timesteps, (1,)).to(device)
            refined_latent = diffusion(latent, t)

            generated_waveform = vae_decoder(refined_latent)

            # Ensure correct waveform shape
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

            final_audio.append(generated_waveform.cpu())

            # Save individual clip
            clip_filename = f"generated_clip_{i+1}.wav"
            torchaudio.save(clip_filename, generated_waveform, sample_rate=SAMPLE_RATE)
            logger.info(f"✅ Clip saved: {clip_filename}")

    # ====== Combine All Clips Into a Full Song ======
    logger.info("🔄 Combining all clips into a full song...")
    full_song = torch.cat(final_audio, dim=1)

    song_filename = "generated_song.wav"
    torchaudio.save(song_filename, full_song, sample_rate=SAMPLE_RATE)
    logger.info(f"🎶 Full song saved as {song_filename}")

except Exception as e:
    logger.error(f"❌ Error occurred: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())

