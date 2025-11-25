import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Manager
from tools import read_nodes, read_elems, load_stimulus_region, compute_surface_area, fibers_elements_to_nodes, normalize_coordinates, pacing_coord_to_grid
from torch.nn.utils.rnn import pad_sequence


pacing_to_id = {"LAA": 0, "LIPV": 1, "LSPV": 2, "RIPV": 3, "ROOF": 4, "RSPV": 5}

class Normalise:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    
    def denormalise(self, y_normalized):
        y_normalized = y_normalized.detach().clone()
        y = y_normalized *  (self.max - self.min) + self.min
        return y
    
    def normalise(self, y):
        y = y.detach().clone()
        y_noralise = (y - self.min) / (self.max - self.min)
        return y_noralise


def custom_collate(batch):
    LAT, uac, area, pacing_id, pacing_coord, pacing_grid, y, coordinate, fibre = zip(*batch)

    LAT = torch.stack(LAT)       
    uac = torch.stack(uac)     
    area = torch.stack(area)     
    pacing_id = torch.stack(pacing_id)
    pacing_coord = pad_sequence(pacing_coord, batch_first=True)     
    pacing_grid = torch.stack(pacing_grid)
    y = torch.stack(y) 
    coordinate = torch.stack(coordinate)
    fibre = torch.stack(fibre)

    return LAT, uac, area, pacing_id, pacing_coord, pacing_grid, y, coordinate, fibre


def get_fold_split(all_case_ids, fold, k=5):
    if fold < 1 or fold > k:
        raise ValueError(f"fold must be between 1 and {k}")
    
    all_case_ids = np.array(all_case_ids)
    np.random.shuffle(all_case_ids)
    folds = np.array_split(all_case_ids, k)
    
    test_case_ids = folds[fold - 1].tolist()
    train_case_ids = np.concatenate([f for i, f in enumerate(folds) if i != fold - 1]).tolist()
    
    return train_case_ids, test_case_ids


class TrainDataset(Dataset):
    def __init__(self, root="/data/Bei/Angela/", pacing_locations=[0], fold=1, regime=1):
        A = [i for i in range(1, 101) if i != 76]
        B = [i for i in range(144, 191) if i != 153 and i != 181]
        
        if regime == 1:
            # if fold == 1:
                # print("A", flush=True)
            self.train_case_ids, self.test_case_ids = get_fold_split(A, fold=fold)
        elif regime == 2:
            if fold == 1:
                print("B", flush=True)
            self.train_case_ids, self.test_case_ids = get_fold_split(B, fold=fold)
        elif regime == 3:
            if fold == 1:
                print("A -> B", flush=True)
            self.train_case_ids, self.test_case_ids = A, B
        elif regime == 4:
            if fold == 1:
                print("B -> A", flush=True)
            self.train_case_ids, self.test_case_ids = B, A
        elif regime == 5:
            if fold == 1:
                print("A + B -> A", flush=True)
            self.train_case_ids, self.test_case_ids = get_fold_split(A, fold=fold)
            self.train_case_ids += B
        elif regime == 6:
            if fold == 1:
                print("A + B -> B", flush=True)
            self.train_case_ids, self.test_case_ids = get_fold_split(B, fold=fold)
            self.train_case_ids += A
        elif regime == 7:
            if fold == 1:
                print("A + B -> A + B", flush=True)
            self.train_case_ids, self.test_case_ids = get_fold_split(A, fold=fold)
            B_train, B_test = get_fold_split(B, fold=fold)
            self.train_case_ids += B_train
            self.test_case_ids += B_test

        
        # self.train_case_ids, self.test_case_ids = [i for i in range(144, 180) if i != 153 and i != 181] + caroline, \
        #                                           [i for i in range(181, 190) if i != 153 and i != 181]

        self.root = Path(root)
        self.mesh_dir = self.root / "meshes"
        self.dataset_path = self.root / "dataset"
        
        self.n_uac_points = 50
        self.LAT = []
        self.case_id = []
        self.area = []
        self.pacing_id = []
        self.pacing_coord = []
        self.pacing_grid = []
        self.y = []
        self.pacing_locations = pacing_locations
        self.coordinates = []
        self.fibres = []

    def load_data(self):
        for i in self.train_case_ids:
            data = np.load(self.dataset_path / f"Case_{i}.npz", allow_pickle=True)
            if data['LAT'].shape[0] != 2100:  # 560  2100
                continue

            mask = np.isin(data["pacing_id"], self.pacing_locations)

            self.LAT.append(data['LAT'][mask])
            self.case_id.append(data['case_id'][mask])
            self.area.append(data["area"][mask])
            self.pacing_id.append(data["pacing_id"][mask])
            self.pacing_coord.append(data["pacing_coord"][mask])
            self.pacing_grid.append(data["pacing_grid"][mask])
            self.y.append(data['y'][mask])
            self.coordinates.append(data["coordinates"][mask])
            self.fibres.append(data["fibres"][mask])

        LAT = np.concatenate(self.LAT, axis=0)
        self.LAT_min, self.LAT_max = LAT.min(), LAT.max()
        # print("Train LAT", LAT.shape, self.LAT_min, self.LAT_max)
        self.LAT = (LAT - self.LAT_min) / (self.LAT_max - self.LAT_min)


        self.case_id = np.concatenate(self.case_id, axis=0).reshape(-1, 1)
        self.pacing_id = np.concatenate(self.pacing_id, axis=0).reshape(-1, 1)
        self.pacing_coord = np.concatenate(self.pacing_coord, axis=0).reshape(-1, 1)
        self.pacing_grid = np.concatenate(self.pacing_grid, axis=0)

        area = np.concatenate(self.area, axis=0).reshape(-1, 1)
        self.area_min, self.area_max = area.min(), area.max()
        self.area = (area - self.area_min) / (self.area_max - self.area_min)
        
        y = np.concatenate(self.y, axis=0)
        self.y = y

        self.coordinates = np.concatenate(self.coordinates, axis=0)
        self.fibres = np.concatenate(self.fibres, axis=0)

        
    def __len__(self):
        return len(self.LAT)

    def __getitem__(self, idx):
        return torch.from_numpy(self.LAT[idx]), \
               torch.from_numpy(self.case_id[idx]), \
               torch.from_numpy(self.area[idx]), \
               torch.from_numpy(self.pacing_id[idx]), \
               torch.from_numpy(self.pacing_coord[idx][0].astype(np.float32)), \
               torch.from_numpy(self.pacing_grid[idx]), \
               torch.from_numpy(self.y[idx]), \
               torch.from_numpy(self.coordinates[idx]), \
               torch.from_numpy(self.fibres[idx])
    


class TestDataset:
    def __init__(self, train_dataset=None, pacing_locations=[0]):
        # super().__init__(root=train_dataset.root, 
        #                  pacing_locations=pacing_locations)
        self.train_dataset = train_dataset

        self.LAT = []
        self.case_id = []
        self.area = []
        self.pacing_id = []
        self.pacing_coord = []
        self.pacing_grid = []
        self.y = []
        self.pacing_locations = pacing_locations
        self.coordinates = []
        self.fibres = []

    def load_data(self):
        for i in self.train_dataset.test_case_ids:
            data = np.load(self.train_dataset.dataset_path / f"Case_{i}.npz", allow_pickle=True)
            if data['LAT'].shape[0] != 2100:
                continue

            mask = np.isin(data["pacing_id"], self.pacing_locations)

            self.LAT.append(data['LAT'][mask])

            self.case_id.append(data['case_id'][mask])
            self.area.append(data["area"][mask])
            self.pacing_id.append(data["pacing_id"][mask])
            self.pacing_coord.append(data["pacing_coord"][mask])
            self.pacing_grid.append(data["pacing_grid"][mask])
            self.y.append(data['y'][mask])
            self.coordinates.append(data["coordinates"][mask])
            self.fibres.append(data["fibres"][mask])

        LAT = np.concatenate(self.LAT, axis=0)
        # print("Test LAT", LAT.shape, LAT.min(), LAT.max())
        self.LAT = (LAT - self.train_dataset.LAT_min) / (self.train_dataset.LAT_max - self.train_dataset.LAT_min)
        
        self.case_id = np.concatenate(self.case_id, axis=0).reshape(-1, 1)
        self.pacing_id = np.concatenate(self.pacing_id, axis=0).reshape(-1, 1)
        self.pacing_coord = np.concatenate(self.pacing_coord, axis=0).reshape(-1, 1)
        self.pacing_grid = np.concatenate(self.pacing_grid, axis=0)
        
        area = np.concatenate(self.area, axis=0).reshape(-1, 1)
        self.area_min, self.area_max = self.train_dataset.area_min, self.train_dataset.area_max
        self.area = (area - self.area_min) / (self.area_max - self.area_min)
        
        y = np.concatenate(self.y, axis=0)
        self.y = y  

        self.coordinates = np.concatenate(self.coordinates, axis=0)
        self.fibres = np.concatenate(self.fibres, axis=0)

    def __len__(self):
        return len(self.LAT)

    def __getitem__(self, idx):
        return torch.from_numpy(self.LAT[idx]), \
               torch.from_numpy(self.case_id[idx]), \
               torch.from_numpy(self.area[idx]), \
               torch.from_numpy(self.pacing_id[idx]), \
               torch.from_numpy(self.pacing_coord[idx][0].astype(np.float32)), \
               torch.from_numpy(self.pacing_grid[idx]), \
               torch.from_numpy(self.y[idx]), \
               torch.from_numpy(self.coordinates[idx]), \
               torch.from_numpy(self.fibres[idx])

    

class BuildDataset():
    def __init__(self, root="/data/Bei/Caroline/"):
        self.root = Path(root)
        self.mesh_dir = self.root / "meshes"
        self.at_rt_dir = self.root / "results"
        self.dataset_path = self.root / "dataset"
        
        self.n_uac_points = 50

    def build_data(self):
        self.dataset_path.mkdir(exist_ok=True, parents=True)
        with Manager() as manager:
            LAT_list = manager.list()
            y_list = manager.list()
            case_id_list = manager.list()
            pacing_id_list = manager.list()
            pacing_coord_list = manager.list()
            pacing_grid_list = manager.list()
            surface_areas = manager.list()
            coordinates_list = manager.list()
            fibres_list = manager.list()

            with ProcessPoolExecutor() as executor:
                for i in range(144, 191):    # 1 to 101
                    case = f"Case_{i}"
                    print(case)
                    UAC_IIR = np.load(self.mesh_dir / case / "UAC_IIR.npz")
                    UAC = np.stack([UAC_IIR["UAC1"], 
                                    UAC_IIR["UAC2"]], axis=1)
                    
                    area = compute_surface_area(self.mesh_dir / f"{case}")       
                    coordinates = read_nodes(self.mesh_dir / f"{case}/{case}.pts")
                    triangles = read_elems(self.mesh_dir / f"{case}/{case}.elem")
                    fibres = np.loadtxt(self.mesh_dir / f"{case}/{case}.lon", dtype=np.float64, skiprows=1)
                    fibres = fibers_elements_to_nodes(coordinates, triangles, fibres, area_weighted=True)

                    coordinates = normalize_coordinates(coordinates)
                    coordinates = self.coordinate_interpolate(UAC, coordinates, self.n_uac_points)
                    coordinates = np.transpose(coordinates, (2, 0, 1))

                    fibres = self.coordinate_interpolate(UAC, fibres, self.n_uac_points)
                    fibres = np.transpose(fibres, (2, 0, 1))            
                    
                    futures = []
                    for at_rt_path in (self.at_rt_dir / case).iterdir():
                        pacing_names = at_rt_path.stem.split("_") 

                        if len(pacing_names) > 1:
                            vtx_file = self.mesh_dir / case / f"{case}_{pacing_names[1]}.vtx"
                            if not vtx_file.exists():
                                continue
                            region = load_stimulus_region(vtx_file)
                            pacing_coord = UAC[region]
                            sorted_indices = np.lexsort((pacing_coord[:, 1], pacing_coord[:, 0]))
                            pacing_coord = pacing_coord[sorted_indices]
                            pacing_grid = pacing_coord_to_grid(pacing_coord, self.n_uac_points)

                            future = executor.submit(self.process_at, UAC, at_rt_path, self.n_uac_points, pacing_to_id[pacing_names[1]], pacing_coord, pacing_grid)
                            futures.append(future)
                        else:
                            region = load_stimulus_region(self.mesh_dir / case / f"{case}.vtx")
                            pacing_coord = UAC[region]
                            sorted_indices = np.lexsort((pacing_coord[:, 1], pacing_coord[:, 0]))
                            pacing_coord = pacing_coord[sorted_indices]
                            pacing_grid = pacing_coord_to_grid(pacing_coord, self.n_uac_points)

                            future = executor.submit(self.process_at, UAC, at_rt_path, self.n_uac_points, 6, pacing_coord, pacing_grid)
                            futures.append(future)

                    for future in as_completed(futures):
                        LAT, y, pacing_id, pacing_coord, pacing_grid = future.result()
                        LAT_list.append(LAT)
                        y_list.append(y)
                        case_id_list.append(i)
                        pacing_id_list.append(pacing_id)
                        pacing_coord_list.append(pacing_coord)
                        pacing_grid_list.append(pacing_grid.reshape(1, self.n_uac_points, self.n_uac_points))
                        surface_areas.append(area)
                        coordinates_list.append(coordinates)
                        fibres_list.append(fibres)

                    np.savez(self.dataset_path / f"{case}.npz", 
                             LAT=np.stack(LAT_list, axis=0).astype(np.float32), # (500, 1, 100, 100) 
                             y=np.stack(y_list, axis=0).astype(np.float32), # (500, 2) 
                             case_id=np.array(case_id_list, dtype=np.long),
                             pacing_id=np.array(pacing_id_list, dtype=np.long),
                             pacing_coord=np.array(pacing_coord_list, dtype=object),
                             pacing_grid=np.stack(pacing_grid_list, axis=0).astype(np.float32),
                             area=np.array(surface_areas).astype(np.float32),
                             coordinates=np.array(coordinates_list).astype(np.float32),
                             fibres=np.array(fibres_list).astype(np.float32))
                    LAT_list[:] = []
                    y_list[:] = []
                    case_id_list[:] = []
                    pacing_id_list[:] = []
                    pacing_coord_list[:] = []
                    pacing_grid_list[:] = []
                    surface_areas[:] = []
                    coordinates_list[:] = []
                    fibres_list[:] = []


    def process_at(self, UAC, at_path, n_uac_points, pacing_id, pacing_coord, pacing_grid):
        AT = torch.load(at_path / "ATs.pt", weights_only=False).numpy().astype(np.float32)
        AT = self.uac_interpolate(UAC, AT, n_uac_points) # (100, 100)
        AT = np.expand_dims(AT, axis=0)                  # (1, 100, 100) 

        conductivity = np.load(at_path / "conductivity.npz")
        g_il, g_it = conductivity["g_il"], conductivity["g_it"]
        y = np.array([g_il, g_it])  # [2]
        
        return AT, torch.from_numpy(y), pacing_id, pacing_coord, pacing_grid

    def uac_interpolate(self, uac, LAT, n_uac_points):
        grid = np.linspace(0, 1, n_uac_points)  
        grid_points = np.meshgrid(grid, grid)

        grid_linear = griddata(uac, LAT, (grid_points[0], grid_points[1]), method='linear', fill_value=np.nan)
        grid_nearest = griddata(uac, LAT, (grid_points[0], grid_points[1]), method='nearest')
        results = np.where(np.isnan(grid_linear),
                           grid_nearest,
                           grid_linear)

        return results
    

    def coordinate_interpolate(self, uac, coordinates, n_uac_points):
        grid = np.linspace(0, 1, n_uac_points)  
        grid_points = np.meshgrid(grid, grid)

        grid_linear = griddata(uac, coordinates, (grid_points[0], grid_points[1]), method='linear', fill_value=np.nan)
        grid_nearest = griddata(uac, coordinates, (grid_points[0], grid_points[1]), method='nearest')
        results = np.where(np.isnan(grid_linear),
                            grid_nearest,
                            grid_linear)

        return results
    

if __name__ == "__main__":
    d = BuildDataset()
    d.build_data()