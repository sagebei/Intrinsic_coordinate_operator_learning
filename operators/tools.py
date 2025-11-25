import random
import numpy as np
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import label, center_of_mass
import os


def plot_lat(
    tensor,
    name="./figures/lat_map",
    i=0,
    highlight_top=0.01,
    max_regions=4,
):
    """
    High-quality Nature-style LAT map with region highlights and edge-aware value labels.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        LAT map or error map to visualize.
    name : str
        Output path (without extension).
    i : int
        Panel index for labeling, e.g. (a), (b), ...
    highlight_top : float
        Fraction (0–1) of top values to consider (default: top 1%).
    max_regions : int
        Maximum number of high-value regions to highlight.
    """
    os.makedirs(os.path.dirname(name), exist_ok=True)

    # Convert tensor to NumPy
    img = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.array(tensor)
    vmin, vmax = 0, 20
    height, width = img.shape

    # --- Base figure ---
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(
        img,
        cmap="viridis",
        interpolation="bilinear",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    # --- Identify top regions ---
    threshold = np.quantile(img, 1 - highlight_top)
    mask = img >= threshold
    labeled, num_features = label(mask)

    if num_features > 0:
        centers = center_of_mass(mask, labeled, range(1, num_features + 1))
        sizes = np.bincount(labeled.ravel())[1:]
        region_means = [img[labeled == (idx + 1)].mean() for idx in range(num_features)]

        # Select top-N regions by intensity
        top_regions = np.argsort(region_means)[-max_regions:][::-1]

        # --- Draw circles + edge-aware labels ---
        for idx in top_regions:
            y, x = centers[idx]
            mean_val = region_means[idx]
            radius = np.sqrt(sizes[idx]) / 1.2  # adjust for larger circles

            # Draw circle
            circ = Circle(
                (x, y),
                radius=radius,
                edgecolor="red",
                facecolor="none",
                linewidth=1.8,
                alpha=0.9,
            )
            ax.add_patch(circ)

            # --- Edge-aware label placement ---
            # Horizontal alignment
            if x < width * 0.3:        # near left → label on right
                x_label = x + radius + 1
                ha = "left"
            elif x > width * 0.7:      # near right → label on left
                x_label = x - radius - 1
                ha = "right"
            else:
                x_label = x + radius + 1
                ha = "left"

            # Vertical alignment
            if y < height * 0.2:       # near bottom → move up
                y_label = y + radius / 2
                va = "bottom"
            elif y > height * 0.8:     # near top → move down
                y_label = y - radius / 2
                va = "top"
            else:
                y_label = y + radius / 2
                va = "center"

            if i == 4 and round(mean_val, 1) == 16.9: 
                y_label = y_label - 1.8
            if i == 4 and round(mean_val, 1) == 16.5:
                y_label = y_label 

            if i == 5 and round(mean_val, 1) == 14.4:
                y_label = y_label - 1
            
            if i == 9 and round(mean_val, 1) == 11.7:
                y_label = y_label - 1

            if i == 11 and round(mean_val, 1) == 14.5: 
                y_label = y_label - 1.5
            if i == 11 and round(mean_val, 1) == 13.6:
                y_label = y_label - 2.5

            if i == 15 and round(mean_val, 1) == 11.5:
                y_label = y_label - 2

            ax.text(
                x_label,
                y_label,
                f"{mean_val:.1f}",
                color="red",
                fontsize=12,
                fontweight="bold",
                ha=ha,
                va=va,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.6,
                    boxstyle="round,pad=0.2"
                ),
            )

    # --- Colorbar ---
    
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        ticks=np.linspace(vmin, vmax, 5),
    )
    cbar.ax.tick_params(labelsize=11)
    if i in [11, 15]:
        cbar.set_label("LAT prediction error (ms)", fontsize=12)
    cbar.ax.set_yticklabels([f"{t:.0f}" for t in np.linspace(vmin, vmax, 5)])

    # --- Axes formatting ---
    l = {9: "A", 1: "B", 15: "C", 5: "D", 4: "E", 11: "F"}
    ax.set_title(f"({l[i]})", fontsize=14, fontweight="bold")
    ax.set_xticks(np.linspace(0, 49, 6))
    ax.set_yticks(np.linspace(0, 49, 6))
    ax.set_xticklabels([str(int(x)) for x in np.linspace(0, 50, 6)], fontsize=12)
    ax.set_yticklabels([str(int(y)) for y in np.linspace(0, 50, 6)], fontsize=12)
    ax.tick_params(axis="both", which="both", direction="out", length=6, width=1)
    ax.set_aspect("equal")

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def boxplot(x, name):
    """
    Generates a clean, publication-quality boxplot suitable for Nature-style figures.
    Assumes x is a tensor of shape (N, 1, H, W) or (N, H, W).
    Saves figure as a vector graphic for high-quality export.
    """
    os.makedirs(os.path.dirname(name), exist_ok=True)
    
    # Flatten spatial dimensions: e.g., (N, 1, 50, 50) → (N, 2500)
    x_flat = x.view(x.size(0), -1)  # Shape: (N, H*W)
    
    # Convert to NumPy and transpose: (N, H*W) → (H*W, N)
    x_np = x_flat.cpu().numpy().T

    # Set up the figure
    fig, ax = plt.subplots(figsize=(12.3, 3.1))  # Wide and short is typical for boxplots
    bp = ax.boxplot(
        x_np,
        showfliers=True,       # Clean presentation
        patch_artist=True,      # Filled boxes
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(facecolor='#1f77b4', linewidth=1),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1)
    )

    # Labeling
    ax.set_xlabel('Test case ID', fontsize=12)
    ax.set_ylabel('Mean LAT prediction error (ms)', fontsize=12)
    
    # Style adjustments
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Tight layout and save as vector graphic
    # plt.tight_layout()
    plt.savefig(f"{name}.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)



def set_random_seed(seed):
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # Numpy random
    torch.manual_seed(seed)                 # Torch CPU random
    torch.cuda.manual_seed(seed)            # Torch current GPU random
    torch.cuda.manual_seed_all(seed)        # Torch all GPUs (if you use DataParallel or DDP)
    torch.backends.cudnn.deterministic = True    # Force CuDNN to be deterministic
    torch.backends.cudnn.benchmark = False       # Disable CuDNN auto-tuner (it can introduce randomness)


def round_to_decimal(x, decimals=1):
    factor = 10 ** decimals
    return torch.round(x * factor) / factor


def mask_random_patch(images, min_patch=10, max_patch=30, mask_value=-1):
    B, C, H, W = images.shape
    masked_images = images.clone()

    for i in range(B):
        # Random patch size
        ph = random.randint(min_patch, max_patch)
        pw = random.randint(min_patch, max_patch)

        # Random top-left corner
        top = random.randint(0, H - ph)
        left = random.randint(0, W - pw)

        # Mask the patch in all channels
        masked_images[i, :, top:top+ph, left:left+pw] = mask_value

    return masked_images


def read_nodes(node_file, unit_conversion=1000):
    node_file = Path(node_file)
    with node_file.open("r") as f:
        num_pts_expected = int(f.readline().strip()) 
    
    nodes = np.loadtxt(node_file, dtype=float, skiprows=1).astype(np.float32)
    num_pts_actual = nodes.shape[0]
    if num_pts_actual != num_pts_expected:
        raise ValueError(f"Mismatch in number of nodes: expected {num_pts_expected}, but found {num_pts_actual}")

    coordinates = nodes / unit_conversion
    return coordinates


def read_elems(elem_file):
    with elem_file.open("r") as f:
        num_elems_expected = int(f.readline().strip())
        first_data_line = f.readline().strip().split()
        usecols = list(range(1, len(first_data_line)))

    data = np.loadtxt(elem_file, dtype=int, skiprows=1, usecols=usecols)
    
    num_elems_actual = data.shape[0]
    if num_elems_actual != num_elems_expected:
        raise ValueError(f"Mismatch: expected {num_elems_expected}, but found {num_elems_actual}")
    elems = data[:, :-1]

    return elems


def compute_surface_area(case_path="/data/Bei/meshes_refined/Case_1"):
    case_path = Path(case_path)
    coordinates = read_nodes(case_path / (case_path.stem + ".pts"))
    elems = read_elems(case_path / (case_path.stem + ".elem"))
    p1 = coordinates[elems[:, 0]]
    p2 = coordinates[elems[:, 1]]
    p3 = coordinates[elems[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1)

    return areas.sum()


def load_stimulus_region(vtx_filepath):
    with Path(vtx_filepath).open("r") as f:
        region_size = int(f.readline().strip())

    region = np.loadtxt(vtx_filepath, dtype=int, skiprows=2)
    
    if len(region) != region_size:
        raise Exception(f"Error loading {vtx_filepath}")
    
    return region


def load_uac(mesh_dir="/data/Bei/meshes_refined"):
    mesh_dir = Path(mesh_dir)
    uac_list = []
    for i in range(1, 101):
        case = f"Case_{i}"
        UAC_IIR = np.load(mesh_dir / case / "UAC_IIR.npz")
        UAC = np.stack([UAC_IIR["UAC1"], 
                        UAC_IIR["UAC2"]], axis=1)
        sorted_indices = np.lexsort((UAC[:, 1], UAC[:, 0]))
        UAC = UAC[sorted_indices]
        UAC = torch.from_numpy(UAC)  # [204322, 2]
        uac_list.append(UAC)
    uac_list = pad_sequence(uac_list, batch_first=True) 

    return uac_list


def load_pacing(mesh_dir="/data/Bei/meshes_refined"):
    mesh_dir = Path(mesh_dir)
    pacing_list = []
    for i in range(1, 101):
        case = f"Case_{i}"

        UAC_IIR = np.load(mesh_dir / case / "UAC_IIR.npz")
        UAC = torch.from_numpy(np.stack([UAC_IIR["UAC1"], 
                                        UAC_IIR["UAC2"]], axis=1))  # [204322, 2]
        # uac_list.append(UAC)

        pacing_sub_list = []
        for pacing in ["LAA", "LIPV", "LSPV", "RIPV", "roof", "RSPV", ""]:
            if (pacing == "RSPV" and i == 40) or (pacing == "LSPV" and i == 66):
                continue
            if pacing == "":
                region = load_stimulus_region(mesh_dir / case / f"{case}.vtx")
            else:
                region = load_stimulus_region(mesh_dir / case / f"{case}_{pacing}.vtx")
            pacing = UAC[region]
            pacing_sub_list.append(pacing)
        
        pacing_sub_list = pad_sequence(pacing_sub_list, batch_first=True)   
        pacing_list.append(pacing_sub_list)
    
    return pacing_list
        

def fibers_elements_to_nodes(vertices, triangles, fibers, area_weighted=False):
    num_nodes = vertices.shape[0]
    node_fibers_sum = np.zeros((num_nodes, 3))
    node_counts = np.zeros((num_nodes, 1))

    if area_weighted:
        # Compute per-element areas
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
    else:
        areas = np.ones(triangles.shape[0])

    for i in range(triangles.shape[0]):
        tri = triangles[i]
        fiber = fibers[i]
        area = areas[i]
        for v in tri:
            node_fibers_sum[v] += area * fiber
            node_counts[v] += area

    # Avoid division by zero
    node_counts[node_counts == 0] = 1.0
    node_fibers = node_fibers_sum / node_counts

    # Normalize to unit vectors
    norms = np.linalg.norm(node_fibers, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    node_fibers /= norms

    return node_fibers


def normalize_coordinates(vertices):
    # Center at origin
    centroid = np.mean(vertices, axis=0)
    vertices_centered = vertices - centroid

    # Scale to unit sphere
    max_dist = np.linalg.norm(vertices_centered, axis=1).max()
    vertices_normalized = vertices_centered / max_dist

    return vertices_normalized

if __name__ == "__main__":
    case = "Case_1"
    coordinates = read_nodes(Path("/data/Bei/meshes_refined/") / f"{case}/{case}.pts")
    triangles = read_elems(Path("/data/Bei/meshes_refined/") / f"{case}/{case}.elem")
    fibres = np.loadtxt(Path("/data/Bei/meshes_refined/") / f"{case}/{case}.lon", dtype=np.float64, skiprows=1)
    print(fibres.shape)
    fibres = fibers_elements_to_nodes(coordinates, triangles, fibres, area_weighted=True)
    print(fibres.shape)
    print(fibres)



    

def pacing_coord_to_grid(pacing_coord, n_uac_points):
    r = 1  # radius in pixels

    H = W = n_uac_points
    grid_binary = np.zeros((H, W), dtype=np.float32)
    for p in pacing_coord:
        u, v = p

        j = int(np.round(u * (W - 1)))
        i = int(np.round(v * (H - 1)))

        for di in range(-r, r+1):
            for dj in range(-r, r+1):
                if di**2 + dj**2 <= r**2:  # within circle
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < H and 0 <= jj < W:
                        grid_binary[ii, jj] = 1
    return grid_binary