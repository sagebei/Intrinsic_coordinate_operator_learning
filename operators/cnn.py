import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from fno import PointCloudToGrid
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError
from loss import H1LossMetric
import itertools
    

class ResidualBlock2d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock2d, self).__init__()
        padding = 1
        if kernel_size == 5:
            padding = 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet2dNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=64, kernel_size=3, layers=3):
        super(ResNet2dNet, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock2d(channels=out_channels, kernel_size=kernel_size) for _ in range(layers)]
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return x



class CNN(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, layers=2):
        super().__init__()
        self.name = "cnn"

        self.pacing_input_net = PointCloudToGrid() # CNN1dNet(input_dim=2, layers=3, latent_dim=2500)
        self.cnn = ResNet2dNet(in_channels=7, out_channels=out_channels, kernel_size=kernel_size, layers=layers)

    def forward(self, conductivity, area, pacing, coordinate, fibre):
        pacing_input_out = self.pacing_input_net(pacing)

        input = torch.concat([pacing_input_out, conductivity, coordinate, area], dim=1)
        out = self.cnn(input)

        out = torch.sigmoid(out)
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
    parser.add_argument("-i", type=int, default=6)
    parser.add_argument("-regime", type=int, default=1)
    parser.add_argument("-device_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    
    param_grid = {
        "out_channels": [64, 64, 128, 128, 256, 256],
        "layers": [2, 4, 2, 4, 2, 4],
        "kernel_size": [3, 5, 3, 5, 3, 5]
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

        model = CNN(**param).to(device)
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
