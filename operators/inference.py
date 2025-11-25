import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from fno import PointCloudToGrid
from tools import plot_lat
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError
import itertools
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset, Normalise, custom_collate
from tools import set_random_seed, load_uac
from loss import laplacian_loss, total_variation_loss, RelativeH1Loss, H1Loss, H1LossMetric
import time

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=10, embed_dim=128, patch_size=5):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x, (H, W)

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTDensePredictor(nn.Module):
    def __init__(self, in_channels=10, embed_dim=128, patch_size=5, num_heads=8, layers=6):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (50//patch_size)**2, embed_dim))
        self.blocks = nn.Sequential(*[
            ViTBlock(embed_dim, num_heads) for _ in range(layers)
        ])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.blocks(x)
        x = x.transpose(1, 2).reshape(x.size(0), -1, H, W)  # (B, embed_dim, H, W)
        x = self.decoder(x)  # (B, 1, 50, 50)
        return x


class ViT(nn.Module):
    def __init__(self, embed_dim=128, layers=5, num_heads=8):
        super().__init__()
        self.name = "vit"
        # print(in_channels)
        self.pacing_input_net = PointCloudToGrid() 
        self.vit = ViTDensePredictor(in_channels=7,             # input channels
                                     embed_dim=embed_dim,       # embedding dimension
                                     patch_size=5,     # patch size
                                     num_heads=num_heads,       # attention heads
                                     layers=layers               # number of ViT blocks
                                    )

    def forward(self, conductivity, area, pacing, coordinate, fibre):
        pacing_input_out = self.pacing_input_net(pacing)
        # input = torch.concat([pacing_input_out, conductivity, coordinate, fibre, area], dim=1)

        inputs = torch.concat([pacing_input_out, conductivity, coordinate, area], dim=1)
        out = self.vit(inputs)

        out = torch.sigmoid(out)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="root",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-root", type=str, default="/data/Bei/Caroline")
    parser.add_argument("-i", type=int, default=1)
    parser.add_argument("-regime", type=int, default=1)
    parser.add_argument("-device_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    
    param_grid = {
        "embed_dim": [64],
        "layers":    [5],
        "num_heads": [8]
    }
    param_combinations = [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]
    
    param = param_combinations[args.i - 1]
    print(param, flush=True)
    
    best_mean_mae_list = []
    best_mean_ssim_list = []
    best_mean_h1_list = []

    pacing_locations = [0, 1, 2, 3, 4, 5, 6]  
    for fold in range(2, 3):
        set_random_seed(42)

        train_dataset = TrainDataset(root=args.root, pacing_locations=pacing_locations, fold=fold, regime=args.regime)
        train_dataset.load_data()
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=custom_collate)

        dn = Normalise(train_dataset.LAT_min, train_dataset.LAT_max)
        # print(train_dataset.LAT.shape, train_dataset.LAT.min(), train_dataset.LAT.max(), flush=True)

        test_dataset = TestDataset(train_dataset=train_dataset, pacing_locations=pacing_locations)
        test_dataset.load_data()
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=custom_collate)
        # print(test_dataset.LAT.shape, test_dataset.LAT.min(), test_dataset.LAT.max(), flush=True)

        model = ViT(**param).to(device)
        model.load_state_dict(torch.load(f"./models/{model.name}_{fold}.pth"))
    
        model.eval()
        with torch.no_grad():
            for i, (activations, case_id, areas, pacing_id, pacing_coord, pacing_grid, conductivities, coordinates, fibres) in enumerate(test_loader):
                activations, case_id, areas, pacing_id, pacing_coord, pacing_grid, conductivities, coordinates, fibres = activations.to(device), case_id.to(device), areas.to(device), pacing_id.to(device), pacing_coord.to(device), pacing_grid.to(device), conductivities.to(device), coordinates.to(device), fibres.to(device)
                areas = areas.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 50, 50)
                conductivities = conductivities.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 50, 50)

                # ---- Start timing ----
                torch.cuda.synchronize()  # wait for all CUDA ops to finish before starting
                start_time = time.time()
                preds = model(conductivities, areas, pacing_coord, coordinates, fibres)

                torch.cuda.synchronize()  # wait for ops to finish before stopping
                elapsed = time.time() - start_time
                print(f"Inference time for batch {i}: {elapsed * 1000:.2f} ms")

                break
