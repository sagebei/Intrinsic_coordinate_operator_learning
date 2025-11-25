import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_convolution import WaveConv2d
from fno import PointCloudToGrid
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError
from loss import H1LossMetric
import itertools

class WNO2d(nn.Module):
    def __init__(self, width=64, level=3, layers=2, size=[100, 100], wavelet="haar", in_channel=3):
        super(WNO2d, self).__init__()
        self.name = "wno"

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range(self.layers):
            self.conv.append(WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet))
            self.w.append(nn.Conv2d(self.width, self.width, 1))


    def forward(self, x):
        x = x.permute(0, 2, 3, 1)                # (B, C, H, W) -> (B, H, W, C)

        x = self.fc0(x)                      # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)            # (B, C, H, W)

        for (convl, wl) in zip(self.conv, self.w):
            x = convl(x) + wl(x)  
            # x = F.mish(x)                    # (B, C, H, W)
        
        return x
    

class WNO(nn.Module):
    def __init__(self, width=128, layers=1, wavelet="haar"):
        super().__init__()
        self.name = "wno"

        self.pacing_input_net = PointCloudToGrid()
        self.wno1 = WNO2d(width=width, level=3, layers=layers, size=[50, 50], wavelet=wavelet, in_channel=7)
        # self.wno2 = WNO2d(width=128, level=3, layers=1, size=[50, 50], wavelet="haar", in_channel=128)
        # self.wno3 = WNO2d(width=128, level=3, layers=1, size=[50, 50], wavelet="haar", in_channel=128)
        self.wno4 = WNO2d(width=1, level=3, layers=1, size=[50, 50], wavelet=wavelet, in_channel=width)

        self.mish = nn.Mish()

    def forward(self, conductivity, area, pacing, coordinate, fibre):
        pacing_input_out = self.pacing_input_net(pacing)

        x = torch.concat([pacing_input_out, conductivity, coordinate, area], dim=1)
        x = self.mish(self.wno1(x))
        # x = self.mish(self.wno2(x))
        # x = self.mish(self.wno3(x))
        x = self.wno4(x)

        out = torch.sigmoid(x)
        return out


# wno = WNO()
# conductivity = torch.randn(8, 2, 50, 50)
# area = torch.randn(8, 1, 50, 50)
# pacing = torch.randn(8, 100, 2)
# coordinate = torch.randn(8, 3, 50, 50)
# fibre = torch.randn(8, 3, 50, 50)

# output = wno(conductivity, area, pacing, coordinate, fibre)
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
    device_id = torch.cuda.current_device()
    
    param_grid = {
        "width": [16, 16, 32, 32, 64, 64],
        "layers": [2, 4, 2, 4, 2, 4],
        "wavelet": ["haar", "db1", "db1", "haar", "db1", "db1"]
    }
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in zip(*param_grid.values())]
    param = param_combinations[args.i - 1]
    print(param)
    
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

        model = WNO(**param).to(device)
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
