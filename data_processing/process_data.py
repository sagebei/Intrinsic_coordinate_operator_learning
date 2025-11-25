import pyvista as pv
import numpy as np
from pathlib import Path
import subprocess

folder_path = Path("/home/bzhou6/Data/") / "UMC_AF/case001"
mesh = pv.read(folder_path / "Labelled_Coords_2D_Rescaling_v3_C.vtk")
filename = folder_path / "clean-Labelled-refined-fibres.vtk"

def write_stimulus(name, stimregion):
    fname = f'{name}.vtx'
    with open(fname,'w') as f:
        f.write('{0:d}\nintra\n'.format(stimregion.shape[0]) )
        for jl in stimregion:
            f.write('{0:d}\n'.format(jl))

def read_nodes(node_file, unit_conversion=1/1000):
    node_file = Path(node_file)
    with node_file.open("r") as f:
        num_pts_expected = int(f.readline().strip()) 
    
    nodes = np.loadtxt(node_file, dtype=float, skiprows=1).astype(np.float32)
    num_pts_actual = nodes.shape[0]
    if num_pts_actual != num_pts_expected:
        raise ValueError(f"Mismatch in number of nodes: expected {num_pts_expected}, but found {num_pts_actual}")

    coordinates = nodes / unit_conversion
    return coordinates

cmd = [
    "python", "convertvtk42.py",
    f"{folder_path}/clean-Labelled-refined-fibres.vtk",
    f"{folder_path}/clean-Labelled-refined-fibres.vtk",
]
subprocess.run(cmd, check=True)

cmd = [
    "meshtool", "convert",
    f"-imsh={folder_path}/clean-Labelled-refined-fibres.vtk",
    "-ifmt=vtk",
    "-ofmt=carp_txt",
    f"-omsh={folder_path}/LA"
]
subprocess.run(cmd, check=True)

cmd = [
    "python", "extract.py",
    f"{folder_path}/clean-Labelled-refined-fibres.vtk",
    f"{folder_path}/LA.lon"
]
subprocess.run(cmd, check=True)


Pts = read_nodes(folder_path / "LA.pts")
UAC = np.array(mesh.points)[:, :2]
np.save(folder_path / "UAC.npy", UAC)

X = UAC[:, 0]
Y = UAC[:, 1]

pacing = []

PV_pacings = np.zeros(UAC.shape[0], dtype=np.float32)
p_radius   = 2.0*1000 #(2 mm in microns)    
# LSPV:  .86-.88    .65-.75
I  = np.logical_and( np.logical_and((X>=0.86),(X<=0.88)  ), np.logical_and((Y>=0.65),(Y<=0.75) ) )

c0 = Pts[I,:].mean(axis=0)
IUAC    = np.linalg.norm(Pts-c0, axis=1)<=p_radius
stim    = np.where(IUAC)[0].astype(int)
write_stimulus(folder_path / "LSPV", stim)
pacing.append(stim)

# LIPV:  .85-.86    .25-.26
I  = np.logical_and( np.logical_and((X>=0.85),(X<=0.86)  ), np.logical_and((Y>=0.25),(Y<=0.26) ) )
c0 = Pts[I,:].mean(axis=0)
IUAC    = np.linalg.norm(Pts-c0, axis=1)<=p_radius
stim    = np.where(IUAC)[0].astype(int)
write_stimulus(folder_path / "LIPV", stim)
pacing.append(stim)

# RSPV:  .05-.08    .60-.63
I  = np.logical_and( np.logical_and((X>=0.05),(X<=0.08)  ), np.logical_and((Y>=0.60),(Y<=0.63) ) )
c0 = Pts[I,:].mean(axis=0)
IUAC    = np.linalg.norm(Pts-c0, axis=1)<=p_radius
stim    = np.where(IUAC)[0].astype(int)
write_stimulus(folder_path / "RSPV", stim)
pacing.append(stim)

# RIPV:  .12-.13    .25-.26
I  = np.logical_and( np.logical_and((X>=0.12),(X<=0.13)  ), np.logical_and((Y>=0.25),(Y<=0.26) ) )
c0 = Pts[I,:].mean(axis=0)
IUAC    = np.linalg.norm(Pts-c0, axis=1)<=p_radius
stim    = np.where(IUAC)[0].astype(int)
write_stimulus(folder_path / "RIPV", stim)
pacing.append(stim)

# LAA:  .9-.91    .9 - .91
I = np.logical_and( np.logical_and((X>=0.9),(X<=0.91)  ), np.logical_and((Y>=0.9),(Y<=0.91) ) )
c0 = Pts[I,:].mean(axis=0)
IUAC    = np.linalg.norm(Pts-c0, axis=1)<=p_radius
stim    = np.where(IUAC)[0].astype(int)
write_stimulus(folder_path / "LAA", stim)
pacing.append(stim)

# ROOF:  .48-.49    .48 - .49
I = np.logical_and( np.logical_and((X>=0.48),(X<=0.49)  ), np.logical_and((Y>=0.48),(Y<=0.49) ) )
c0 = Pts[I,:].mean(axis=0)
IUAC    = np.linalg.norm(Pts-c0, axis=1)<=p_radius
stim    = np.where(IUAC)[0].astype(int)
write_stimulus(folder_path / "ROOF", stim)
pacing.append(stim)

# CS
IUAC    = np.logical_and(np.logical_and(X>=0.5, X<=0.7 ),np.logical_and(Y>=0.8, Y<=0.9 ))
c0      = Pts[IUAC,:].mean(axis=0)
IUAC    = np.linalg.norm(Pts-c0, axis=1)<=p_radius
stim    = np.where(IUAC)[0].astype(int)
write_stimulus(folder_path / "CS", stim)
pacing.append(stim)


