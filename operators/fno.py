import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError
from loss import H1LossMetric
import itertools


class PointCloudToGrid(nn.Module):
    def __init__(self, point_dim=2, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        self.grid_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 50 * 50)
        )
    
    def forward(self, x):
        """
        x: (B, N, 2)
        """
        point_feats = self.point_mlp(x)  # (B, N, embed_dim)
        pooled = point_feats.mean(dim=1)  # (B, embed_dim)
        out = self.grid_mlp(pooled).view(-1, 1, 50, 50)
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep along height; must <= H
        self.modes2 = modes2  # Number of Fourier modes to keep along width;  must <= (W//2+1)

        self.scale = 1 / (in_channels * out_channels)
        # Low positive frequencies 
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)  # (in_channels, out_channels, modes1, modes2)
        )
        # Low negative frequencies 
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)  # (in_channels, out_channels, modes1, modes2)
        )

    def compl_mul2d(self, input, weights):
        """
        Args:
            input: (batch, in_channel, height, width)
            weights: (in_channel, out_channel, modes1, modes2)
        Returns:
            (batch, out_channel, height, width)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (batch, in_channels, H, W//2 + 1)

        # Initialize output in Fourier space
        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1,    # (batch, out_channels, H, W//2 + 1)
                             device=x.device, dtype=torch.cfloat)

        # Apply weights on the selected modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],   # (batch, in_channels, modes1, modes2) @ (in_channels, out_channels, modes1, modes2) 
                                                                    self.weights1)                            # = (batch, out_channels, modes1, modes2)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], 
                                                                     self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')  # (batch, out_channels, H, W)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, in_channels=4, layers=4):
        super(FNO2d, self).__init__()
        self.name = "fno"

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.layers = layers

        # Input lifting: project input to high-dimensional space
        self.fc0 = nn.Linear(in_channels, self.width)

        # Fourier layers + pointwise convolutions
        self.spectral_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()

        for _ in range(self.layers):
            self.spectral_convs.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.pointwise_convs.append(nn.Conv2d(self.width, self.width, 1)) 

        # self.activation = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C_in, H, W) -> (N, H, W, C_in)
        x = self.fc0(x)            # (N, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, width) -> (N, width, H, W)

        for spectral_conv, pointwise_conv in zip(self.spectral_convs, self.pointwise_convs):
            x1 = spectral_conv(x)
            x2 = pointwise_conv(x)
            # x = self.activation(x1 + x2)
            x = x1 + x2
        return x


class FNO(nn.Module):
    def __init__(self, n_mode=20, width=32, layers=1):
        super().__init__()
        self.name = "fno"

        self.pacing_input_net = PointCloudToGrid()
        self.fno1 = FNO2d(modes1=n_mode, modes2=n_mode, width=width, in_channels=7, layers=layers)
        # self.fno2 = FNO2d(modes1=20, modes2=20, width=32, in_channels=32, depth=1)
        # self.fno3 = FNO2d(modes1=20, modes2=20, width=16, in_channels=32, depth=1)
        self.fno4 = FNO2d(modes1=n_mode, modes2=n_mode, width=1, in_channels=width, layers=layers)

        self.gelu = nn.GELU()

    def forward(self, conductivity, area, pacing, coordinate, fibre):
        pacing_input_out = self.pacing_input_net(pacing)

        x = torch.concat([pacing_input_out, conductivity, coordinate, area], dim=1)
        x = self.gelu(self.fno1(x))
        # x = self.gelu(self.fno2(x))
        # x = self.gelu(self.fno3(x))
        x = self.fno4(x)

        out = torch.sigmoid(x)
        return out


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
    parser.add_argument("-root", type=str, default="/data/Bei/")
    parser.add_argument("-i", type=int, default=1)
    parser.add_argument("-regime", type=int, default=1)
    parser.add_argument("-device_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)

    param_grid = {
        "n_mode": [8, 8, 16, 16, 24, 24],
        "width":  [16, 32, 16, 32, 64, 64],
        "layers": [2, 2, 4, 4, 2, 4],
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

        model = FNO(**param).to(device)
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
