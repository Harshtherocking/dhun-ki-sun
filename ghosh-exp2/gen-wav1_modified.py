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

def setup_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import gc
        gc.collect()

def initialize_model(model_class, model_name, device):
    logger.info(f"Initializing {model_name}...")
    try:
        model = model_class()
        model = model.to('cpu')
        model = model.to(device)
        logger.info(f"Successfully initialized {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize {model_name}: {str(e)}")
        return None

try:
    logger.info("🎵 Starting song generation pipeline...")
    setup_cuda_memory()

    checkpoint_folder = "/home/amithk/Desktop/Dhun_ki_sun/Gmodel/epoch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    DURATION = 5
    SAMPLE_RATE = Config.SAMPLE_RATE
    TOTAL_SAMPLES = SAMPLE_RATE * DURATION
    BATCH_SIZE = 1
    final_audio = []

    # List and sort checkpoint files
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_folder) if f.endswith(".pt")])
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the folder!")
    logger.info(f"Found {len(checkpoint_files)} checkpoint files")

    # Initialize models
    logger.info("Starting model initialization...")
    models = {}
    for model_class, name in [
        (VAEEncoder, "VAE Encoder"),
        (VAEDecoder, "VAE Decoder"),
        (Diffusion, "Diffusion"),
        (CLIPTextEncoder, "CLIP Encoder")
    ]:
        models[name] = initialize_model(model_class, name, device)
        if models[name] is None:
            raise RuntimeError(f"Failed to initialize {name}")
        setup_cuda_memory()

    vae_encoder = models["VAE Encoder"]
    vae_decoder = models["VAE Decoder"]
    diffusion = models["Diffusion"]
    clip_encoder = models["CLIP Encoder"]

    logger.info("All models initialized successfully")

    # Continue from the last successful checkpoint (22)
    start_idx = 22
    for i, checkpoint_file in enumerate(checkpoint_files[start_idx:], start=start_idx):
        setup_cuda_memory()
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        logger.info(f"🔄 Loading checkpoint: {checkpoint_file}")

        try:
            # Load checkpoint to CPU first
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load weights on CPU
            vae_encoder.cpu()
            vae_decoder.cpu()
            diffusion.cpu()
            clip_encoder.cpu()

            try:
                vae_encoder.load_state_dict(checkpoint["encoder_state_dict"])
                vae_decoder.load_state_dict(checkpoint["decoder_state_dict"])
                if "diffusion_state_dict" in checkpoint:
                    diffusion.load_state_dict(checkpoint["diffusion_state_dict"])
                if "clip_state_dict" in checkpoint:
                    clip_encoder.load_state_dict(checkpoint["clip_state_dict"])
            except Exception as e:
                logger.error(f"Failed to load weights from checkpoint {checkpoint_file}: {str(e)}")
                continue

            # Move back to GPU
            vae_encoder.to(device)
            vae_decoder.to(device)
            diffusion.to(device)
            clip_encoder.to(device)

            del checkpoint
            setup_cuda_memory()

            for model in [vae_encoder, vae_decoder, diffusion, clip_encoder]:
                model.eval()

            text_input = f"Thaat {i+1}"
            logger.info(f"Generating music for: {text_input}")

            with torch.no_grad():
                tokens = torch.randint(0, 5000, (BATCH_SIZE, 77)).to(device)
                text_embedding = clip_encoder(tokens)

                mu, logvar = vae_encoder(text_embedding)
                latent = vae_encoder.reparameterize(mu, logvar)

                t = torch.randint(0, diffusion.timesteps, (1,)).to(device)
                refined_latent = diffusion(latent, t)

                generated_waveform = vae_decoder(refined_latent)

                del text_embedding, mu, logvar, latent, refined_latent
                setup_cuda_memory()

                if len(generated_waveform.shape) > 2:
                    generated_waveform = generated_waveform.squeeze(0)
                if len(generated_waveform.shape) == 1:
                    generated_waveform = generated_waveform.unsqueeze(0)

                if generated_waveform.shape[1] > TOTAL_SAMPLES:
                    generated_waveform = generated_waveform[:, :TOTAL_SAMPLES]
                elif generated_waveform.shape[1] < TOTAL_SAMPLES:
                    pad_size = TOTAL_SAMPLES - generated_waveform.shape[1]
                    generated_waveform = torch.nn.functional.pad(generated_waveform, (0, pad_size))

                generated_waveform = generated_waveform.cpu()
                final_audio.append(generated_waveform)

                clip_filename = f"generated_clip_{i+1}.wav"
                torchaudio.save(clip_filename, generated_waveform, sample_rate=SAMPLE_RATE)
                logger.info(f"✅ Clip saved: {clip_filename}")

        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint_file}: {str(e)}")
            continue

    if final_audio:
        logger.info("🔄 Combining all clips into a full song...")
        full_song = torch.cat(final_audio, dim=1)
        song_filename = "generated_song.wav"
        torchaudio.save(song_filename, full_song, sample_rate=SAMPLE_RATE)
        logger.info(f"🎶 Full song saved as {song_filename}")
    else:
        logger.error("No audio clips were generated successfully")

except Exception as e:
    logger.error(f"❌ Error occurred: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
