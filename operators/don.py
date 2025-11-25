import torch
import torch.nn as nn
from operators.ours import ViTDensePredictor
from fno import PointCloudToGrid
from tools import plot_lat
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError
from loss import H1LossMetric
import itertools


class CNN2dNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, layers=3):
        super(CNN2dNet, self).__init__()

        self.conv_layers = nn.ModuleList()
        
        # First layer with correct input channels
        self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv_layers.append(nn.BatchNorm2d(out_channels))
        self.conv_layers.append(nn.ReLU())

        # Subsequent layers
        for _ in range(1, layers-1):
            self.conv_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nn.ReLU())

        self.conv_layers.append(nn.Conv2d(out_channels, 1, kernel_size=1, padding=0))
        self.conv_layers.append(nn.BatchNorm2d(1))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


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


class DeepONet2d(nn.Module):
    def __init__(self, embed_dim=128, layers=3, num_heads=8):
        super().__init__()
        self.name = "don"

        self.pacing_input_net = PointCloudToGrid()
        self.branch_net = ViTDensePredictor(in_channels=3, embed_dim=embed_dim, patch_size=5, num_heads=num_heads, layers=layers)
        self.trunk_net = ViTDensePredictor(in_channels=4, embed_dim=embed_dim, patch_size=5, num_heads=num_heads, layers=layers)
        self.output_net = ViTDensePredictor(in_channels=1, embed_dim=embed_dim, patch_size=5, num_heads=num_heads, layers=1)

        # self.output_net = CNN2dNet(in_channels=1, out_channels=embed_dim, layers=2)

    def forward(self, conductivity, area, pacing, coordinate, fibre):
        pacing_input_out = self.pacing_input_net(pacing)

        branch_input = torch.concat([pacing_input_out, conductivity], dim=1)
        trunk_input = torch.concat([coordinate, area], dim=1)

        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)

        x = branch_out * trunk_out
        x = self.output_net(x)
        x = torch.sigmoid(x)
        
        return x


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    from dataset import TrainDataset, TestDataset, Normalise, custom_collate
    import argparse
    from tools import set_random_seed, load_uac
    from loss import laplacian_loss, total_variation_loss, RelativeH1Loss, H1Loss


    parser = argparse.ArgumentParser(description="root",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-root", type=str, default="/data/Bei/Caroline")
    parser.add_argument("-i", type=int, default=1)
    parser.add_argument("-regime", type=int, default=1)
    parser.add_argument("-device_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    
    param_grid = {
        "embed_dim": [64, 64, 128, 128, 128, 256],
        "layers":    [4, 5, 3, 6, 5, 5],
        "num_heads": [4, 8, 8, 8, 4, 8]
    }
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in zip(*param_grid.values())]
    param = param_combinations[args.i - 1]
    print(param, flush=True)
    
    best_mean_mae_list = []
    best_mean_ssim_list = []
    best_mean_h1_list = []

    pacing_locations = [0, 1, 2, 3, 4, 5, 6]  
    for fold in range(1, 6):
        set_random_seed(42)

        train_dataset = TrainDataset(root=args.root, pacing_locations=pacing_locations, fold=fold, regime=args.regime)
        train_dataset.load_data()
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=custom_collate)

        dn = Normalise(train_dataset.LAT_min, train_dataset.LAT_max)
        # print(train_dataset.LAT.shape, train_dataset.LAT.min(), train_dataset.LAT.max(), flush=True)

        test_dataset = TestDataset(train_dataset=train_dataset, pacing_locations=pacing_locations)
        test_dataset.load_data()
        test_loader = DataLoader(test_dataset, batch_size=300 * len(pacing_locations), shuffle=False, collate_fn=custom_collate)
        # print(test_dataset.LAT.shape, test_dataset.LAT.min(), test_dataset.LAT.max(), flush=True)

        model = DeepONet2d(**param).to(device)
        criterion = RelativeH1Loss() 
        # criterion = H1Loss() 
        optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        best_mean_mae = torch.inf
        best_mean_ssim = -torch.inf
        best_mean_h1 = torch.inf
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_error = 0
            total_samples = 0
            # print("total_variation_loss")
            for activations, case_id, areas, pacing_id, pacing_coord, pacing_grid, conductivities, coordinates, fibres in train_loader:
                activations, case_id, areas, pacing_id, pacing_coord, pacing_grid, conductivities, coordinates, fibres = activations.to(device), case_id.to(device), areas.to(device), pacing_id.to(device), pacing_coord.to(device), pacing_grid.to(device), conductivities.to(device), coordinates.to(device), fibres.to(device)
                areas = areas.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 50, 50)
                conductivities = conductivities.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 50, 50)

                preds = model(conductivities, areas, pacing_coord, coordinates, fibres)
                loss = criterion(preds, activations) + 0.01 * total_variation_loss(preds)
                

                batch_size = conductivities.size(0)
                train_error += torch.abs(dn.denormalise(preds) - dn.denormalise(activations)).mean().item() * batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_size
                total_samples += batch_size
            scheduler.step()

            avg_train_loss = train_loss / total_samples
            avg_train_acc = train_error / total_samples

            mae_metric = MeanAbsoluteError().to(device)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            h1_metric = H1LossMetric().to(device)
            model.eval()
            with torch.no_grad():
                for i, (activations, case_id, areas, pacing_id, pacing_coord, pacing_grid, conductivities, coordinates, fibres) in enumerate(test_loader):
                    activations, case_id, areas, pacing_id, pacing_coord, pacing_grid, conductivities, coordinates, fibres = activations.to(device), case_id.to(device), areas.to(device), pacing_id.to(device), pacing_coord.to(device), pacing_grid.to(device), conductivities.to(device), coordinates.to(device), fibres.to(device)
                    areas = areas.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 50, 50)
                    conductivities = conductivities.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 50, 50)

                    preds = model(conductivities, areas, pacing_coord, coordinates, fibres)
                    loss = criterion(preds, activations)

                    mae_metric.update(dn.denormalise(preds), dn.denormalise(activations))
                    ssim_metric.update(preds, activations)
                    h1_metric.update(preds, activations)

            mean_mae = mae_metric.compute()
            mean_ssim = ssim_metric.compute()
            mean_h1 = h1_metric.compute()

            if mean_mae < best_mean_mae:  
                best_mean_mae = mean_mae
            if mean_ssim > best_mean_ssim:  
                best_mean_ssim = mean_ssim
            if mean_h1 < best_mean_h1:  
                best_mean_h1 = mean_h1

            mae_metric.reset()
            ssim_metric.reset()
            h1_metric.reset()
        
        best_mean_mae_list.append(best_mean_mae.item())
        best_mean_ssim_list.append(best_mean_ssim.item())
        best_mean_h1_list.append(best_mean_h1.item())

    print("MAE", f"{np.array(best_mean_mae_list).mean():.4f}", f"{np.array(best_mean_mae_list).std():.5f}")
    print("SSIM", f"{np.array(best_mean_ssim_list).mean():.4f}", f"{np.array(best_mean_ssim_list).std():.5f}")
