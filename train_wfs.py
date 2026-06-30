import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from vajra import config
from vajra.simulator import AOPipelineSimulator
from vajra.reconstructor import Layer0AttentionCNN
from vajra.differentiable_optics import DifferentiablePropagator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_pipeline(mode='point', loss_type='supervised', epochs=5, lr=1e-3, batch_size=2):
    print("=" * 60)
    print(f"Starting WFS CNN Training Loop")
    print(f"Target Mode : {mode.upper()}")
    print(f"Loss Type   : {loss_type.upper()}")
    print(f"Device      : {device}")
    print("=" * 60)
    
    # 1. Initialize simulator & model components
    sim = AOPipelineSimulator(mode=mode)
    model = Layer0AttentionCNN().to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup differentiable propagator for self-supervised mode
    propagator = None
    if loss_type == 'self-supervised':
        propagator = DifferentiablePropagator(sim.pupil_mask, sim.zernike_basis[:config.ZERNIKE_MODES_MAX]).to(device)
        
    print(f"Loading/preparing observation scenes...")
    
    t_time = 0.0
    # Run loop
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        
        # We run multiple iterations per epoch generating random atmospheres on-the-fly
        num_iterations = 5
        for step in range(num_iterations):
            optimizer.zero_grad()
            
            # Synthesize batch on-the-fly
            batch_frames = []
            batch_true_z = []
            batch_illum = []
            batch_gran = []
            
            for b in range(batch_size):
                # Evolve atmosphere dynamically (increasing time monotonically)
                t_time += 0.001
                sim.evolve_atmosphere(t_time)
                
                # Get true phase screen Zernikes for supervised target
                scale = (0.08 / config.R0_500) ** (5.0 / 6.0)
                true_phase = sim.get_phase_screen() * scale
                true_zernikes = []
                for j in range(config.ZERNIKE_MODES_MAX):
                    z_mode = sim.zernike_basis[j]
                    coeff = np.sum(true_phase * z_mode) / (np.sum(z_mode * z_mode) + 1e-8)
                    true_zernikes.append(coeff)
                true_zernikes = np.array(true_zernikes)
                
                # Generate detector frame (spots or solar convolved granulation)
                # We add some small random DM commands to simulate active loops
                dm_rand = np.random.normal(0, 1e-8, config.DM_ACTUATORS_TOTAL)
                raw_frame, illum = sim.generate_hartmannogram(dm_commands=dm_rand)
                
                batch_frames.append(raw_frame)
                batch_true_z.append(true_zernikes)
                batch_illum.append(illum)
                
                if mode == 'solar':
                    # Crop solar granulation patches matching simulator's slicing
                    granulation_scene = sim.granulation.get_patch()
                    patches = []
                    for i in range(len(sim.subaps_pos)):
                        row = i // sim.mla_grid_size
                        col = i % sim.mla_grid_size
                        scene_cropped = granulation_scene[
                            (row*8) % (128-16) : (row*8) % (128-16) + 16,
                            (col*8) % (128-16) : (col*8) % (128-16) + 16
                        ]
                        patches.append(scene_cropped)
                    batch_gran.append(np.array(patches))
                    
            # Convert to PyTorch tensors
            x = torch.tensor(np.array(batch_frames), dtype=torch.float32, device=device).unsqueeze(1) # (B, 1, 256, 256)
            y_true = torch.tensor(np.array(batch_true_z), dtype=torch.float32, device=device) # (B, 66)
            
            # Normalize inputs
            x_norm = (x - 10.0) / (x.max() + 1e-5)
            
            # Predict
            pred_z = model(x_norm)
            
            # Compute loss
            if loss_type == 'supervised':
                loss = nn.MSELoss()(pred_z, y_true)
            else:
                # Self-supervised image reconstruction loss
                raw_subs_list = []
                for b in range(batch_size):
                    frame_t = torch.tensor(batch_frames[b], dtype=torch.float32, device=device)
                    # Slice into 256 patches
                    patches_f = frame_t.unfold(0, 16, 16).unfold(1, 16, 16)
                    patches_f = patches_f.permute(0, 1, 2, 3).contiguous().view(-1, 16, 16)
                    raw_subs_list.append(patches_f)
                raw_subs = torch.cat(raw_subs_list, dim=0) # (B * 256, 16, 16)
                
                if mode == 'solar':
                    gran_patches = torch.tensor(np.concatenate(batch_gran), dtype=torch.float32, device=device) # (B * 256, 16, 16)
                else:
                    gran_patches = None
                    
                loss, _ = propagator(pred_z, gran_patches, raw_subs, target_mode=mode)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / num_iterations
        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.5f} - Time: {elapsed:.2f}s")
        
    # Save model checkpoint
    os.makedirs('data', exist_ok=True)
    checkpoint_path = f"data/wfs_model_{mode}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print("=" * 60)
    print(f"Model saved successfully to: {checkpoint_path}")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAJRA Wavefront Sensor CNN Model")
    parser.add_argument('--mode', type=str, default='point', choices=['point', 'solar'],
                        help="Wavefront target type (point source LGS or extended solar granulation)")
    parser.add_argument('--type', type=str, default='supervised', choices=['supervised', 'self-supervised'],
                        help="Training loss style")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch', type=int, default=2, help="Batch size")
    
    args = parser.parse_args()
    train_pipeline(mode=args.mode, loss_type=args.type, epochs=args.epochs, lr=args.lr, batch_size=args.batch)
