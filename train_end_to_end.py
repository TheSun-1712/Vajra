import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from vajra import config
from vajra.simulator import AOPipelineSimulator
from vajra.detector import DetectorProcessor
from vajra.reconstructor import WavefrontReconstructor, Layer0AttentionCNN
from vajra.controller import ActuatorController
from vajra.differentiable_optics import DifferentiablePropagator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GranulationAugmentation:
    """Applies solar-physics compatible data augmentations for self-supervised contrastive learning (Opt. 3)."""
    def __call__(self, x):
        # x is (1, 16, 16) PyTorch tensor
        # 1. Random 90/180/270 rotation
        rot_k = np.random.choice([0, 1, 2, 3])
        x_aug = torch.rot90(x, rot_k, [1, 2])
        
        # 2. Random translation/rolling (1-2 pixels)
        shift_y, shift_x = np.random.randint(-2, 3, size=2)
        x_aug = torch.roll(x_aug, shifts=(shift_y, shift_x), dims=(1, 2))
        
        # 3. Random scaling & contrast adjustment to prevent texture overfitting
        scale = np.random.uniform(0.8, 1.2)
        x_aug = x_aug * scale
        
        # 4. Add small random noise
        noise = torch.randn_like(x_aug) * 0.02
        x_aug = torch.clamp(x_aug + noise, 0.0, 1.0)
        return x_aug


class ContrastiveGranulationDataset(Dataset):
    """Generates pairs of augmented patches for SimCLR-style contrastive pre-training."""
    def __init__(self, size=100):
        self.size = size
        self.sim = AOPipelineSimulator(mode="solar")
        self.scene = torch.tensor(self.sim.granulation.image, dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Crop a random 16x16 patch from the granulation image
        h, w = self.scene.shape
        start_y = np.random.randint(0, h - 16)
        start_x = np.random.randint(0, w - 16)
        patch = self.scene[start_y:start_y+16, start_x:start_x+16].unsqueeze(0) # (1, 16, 16)
        
        # Apply different augmentations to create positive pairs
        augment = GranulationAugmentation()
        view1 = augment(patch)
        view2 = augment(patch)
        return view1, view2


def train_contrastive_pretraining(epochs=3, batch_size=8):
    """SimCLR-style self-supervised pre-training loop for the ResNet WFS backbone (Opt. 3)."""
    print("\n[Self-Supervised] Starting contrastive pre-training on solar granulation...")
    
    dataset = ContrastiveGranulationDataset(size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Ingest the target CNN model
    wfs_cnn = Layer0AttentionCNN().to(device)
    backbone = wfs_cnn.resnet_backbone
    projection_head = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16)
    ).to(device)
    
    optimizer = optim.Adam(list(backbone.parameters()) + list(projection_head.parameters()), lr=1e-3)
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for view1, view2 in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            
            b = view1.size(0)
            combined = torch.cat([view1, view2], dim=0) # (2*B, 1, 16, 16)
            
            features = backbone(combined) # (2*B, 64)
            projections = projection_head(features) # (2*B, 16)
            
            z1, z2 = projections[:b], projections[b:]
            
            # NT-Xent Contrastive Loss
            z1_norm = nn.functional.normalize(z1, dim=-1)
            z2_norm = nn.functional.normalize(z2, dim=-1)
            
            similarity_matrix = torch.matmul(z1_norm, z2_norm.T) # (B, B)
            positives = torch.diag(similarity_matrix)
            loss = torch.mean(1.0 - positives)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch}/{epochs} | Contrastive Loss: {total_loss/len(dataloader):.4f}")
        
    print("[Self-Supervised] Contrastive pre-training complete. Saving backbone checkpoint.")
    torch.save(backbone.state_dict(), "data/resnet_backbone_pretrained.pth")
    return wfs_cnn


def train_end_to_end_loop(wfs_cnn, epochs=3, batch_size=4):
    """End-to-end differentiable loop training (Opt. 5).
    Chains CNN sensor -> DM projection -> Propagator, using actual physical simulation steps.
    """
    print("\n[End-to-End] Initiating differentiable loop backpropagation training...")
    
    sim = AOPipelineSimulator(mode="solar")
    reconstructor = WavefrontReconstructor(
        sim.pupil_grid, sim.pupil_mask, sim.subaps_pos, sim.subap_masks, target_mode="solar"
    )
    dm_controller = ActuatorController(reconstructor.G_full, sim.pupil_grid, reconstructor.zernike_basis)
    propagator = DifferentiablePropagator(sim.pupil_mask, reconstructor.zernike_basis[:config.ZERNIKE_MODES_MAX])
    
    optimizer = optim.Adam(wfs_cnn.parameters(), lr=1e-4)
    scene = torch.tensor(sim.granulation.image, dtype=torch.float32, device=device)
    
    for epoch in range(1, epochs + 1):
        observed_list = []
        for b in range(batch_size):
            # Generate actual wavefront-degraded hartmannograms by evolving atmosphere
            loop_time = (epoch * batch_size + b) * 0.005
            sim.evolve_atmosphere(loop_time)
            raw_frame, illum = sim.generate_hartmannogram(dm_commands=np.zeros(dm_controller.num_actuators))
            observed_list.append(torch.tensor(raw_frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
            
        observed_frame = torch.cat(observed_list, dim=0) # (Batch, 1, 256, 256)
        
        # Slice actual raw frame into 16x16 subapertures
        raw_subs_tensor = observed_frame.squeeze(1).unfold(1, 16, 16).unfold(2, 16, 16) # (Batch, 16, 16, 16, 16)
        raw_subimages = raw_subs_tensor.contiguous().view(-1, 16, 16) # (Batch * 256, 16, 16)
        
        # Slice corresponding real granulation patches from solar scene
        gran_list = []
        for b in range(batch_size):
            for i in range(256):
                sy = np.random.randint(0, scene.shape[0] - 16)
                sx = np.random.randint(0, scene.shape[1] - 16)
                gran_list.append(scene[sy:sy+16, sx:sx+16].unsqueeze(0))
        granulation_patches = torch.cat(gran_list, dim=0) # (Batch * 256, 16, 16)
        
        # 1. Feed observed frame to WFS CNN
        predicted_z = wfs_cnn(observed_frame) # (Batch, 66)
        
        # 2. Backpropagate loss through differentiable propagator
        loss, simulated_subimages = propagator(
            predicted_zernikes=predicted_z,
            granulation_patches=granulation_patches,
            raw_subimages=raw_subimages,
            target_mode="solar"
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Epoch {epoch}/{epochs} | Differentiable Wavefront Loss: {loss.item():.4f}")
        
    print("[End-to-End] Loop training complete. Saving optimized model weights.")
    torch.save(wfs_cnn.state_dict(), "data/wfs_model_solar_optimized.pth")
    print("[End-to-End] Model saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="VAJRA End-to-End & Contrastive Training Suite")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Run the self-supervised contrastive pre-training
    wfs_cnn = train_contrastive_pretraining(epochs=args.epochs, batch_size=args.batch_size * 2)
    
    # Run the end-to-end differentiable loop backprop training
    train_end_to_end_loop(wfs_cnn, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
