import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from New_Loader import TRFDataset
from VAEmodel import VAE

# Load dataset
root_dir = r"F:\GDG\Thaat and Raga Forest (TRF) Dataset Output"
trf_dataset = TRFDataset(root_dir, target_length=220500)
trf_loader = DataLoader(trf_dataset, batch_size=16, shuffle=True)  # Batch Size 16

# Initialize VAE model
vae = VAE(input_dim=220500, latent_dim=256)  # Updated latent dim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

# Define optimizer and scheduler
optimizer = optim.AdamW(vae.parameters(), lr=2e-4, weight_decay=0.01)  # AdamW Optimizer
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

def loss_function(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + (0.01 * kl_div)  # Increased KL weight
    
    # Convert loss to percentage
    max_possible_loss = torch.mean(x**2)  # Normalizing factor
    loss_percentage = (loss / (max_possible_loss + 1e-9)) * 100  # Avoid division by zero
    
    return loss, loss_percentage.item()  # Return both loss & percentage

# Training loop
num_epochs = 300  # Increased for proper training
for epoch in range(num_epochs):
    total_loss = 0
    total_loss_percentage = 0  # Track percentage loss

    for batch_idx, (song_id, clip_id, waveform) in enumerate(trf_loader):  
        waveform = waveform.to(device)
        waveform = waveform.view(waveform.shape[0], -1)

        optimizer.zero_grad()
        recon_waveform, mu, logvar = vae(waveform)
        loss, loss_percentage = loss_function(recon_waveform, waveform, mu, logvar)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_loss_percentage += loss_percentage  # Accumulate percentage loss

        if batch_idx % 10 == 0:
            print(f"📌 Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(trf_loader)}], Loss: {loss_percentage:.2f}%")

    avg_loss_percentage = total_loss_percentage / len(trf_loader)
    print(f"✅ Epoch {epoch+1} Completed, Avg Loss: {avg_loss_percentage:.2f}%")

    if epoch > 10 and avg_loss_percentage < 0.5:
        print("🎯 Early stopping activated to prevent overfitting.")
        break

# Save trained model
torch.save(vae.state_dict(), "vae_model.pth")
