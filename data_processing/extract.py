import numpy as np
import pyvista as pv
import argparse

def vtk2lon(mesh, lon_file,array_name='endo_avg'):
    """
    Convert a VTK mesh to a .lon file format.
    Parameters:
    - vtk_file (str): Path to the input VTK file.
    - lon_file (str): Path to the output .lon file.
    """
    # Check if 'endo_avg' is in cell data
    if array_name not in mesh.cell_data:
        raise KeyError(array_name + " not found in mesh.cell_data")
    # Extract 'epi_avg' data
    endo_avg = mesh.cell_data[array_name]
    # Write to .lon file
    with open(lon_file, 'w') as f:
        f.write("1\n")  # Header line
        np.savetxt(f, endo_avg, fmt="%.6f")  # Values, one per line
def vtk2pts(mesh,ptsfile):
    """
    Write the points of the mesh to a .pts file.
    :param ptsfile: str, output .pts file name
    :param msh: pyvista.PolyData, input mesh
    """
    num_pts = mesh.n_points
    if ptsfile is None:
        print('No ptsfile given')
        print(num_pts)
        for x,y,z in mesh.points:
            print(f"{x} {y} {z}")
    else:
        print('Writing to '+ptsfile)
        with open(ptsfile, 'w') as f:
            f.write(f"{num_pts}\n")
            for x,y,z in mesh.points:
                f.write(f"{x} {y} {z}\n")
            f.close()
    return
# def extract_triangles(mesh):
#     """Extract only triangle cells from the mesh."""
#     return mesh.extract_cells(mesh.cells_dict[5])  # VTK_TRIANGLE = 5
def get_triangle_cells_with_tags(mesh):
    """Extract triangle cells and their point indices with tags."""
    if 'elemTag' not in mesh.cell_data:
        raise ValueError("Cell data must contain 'elemTag'.")
    tags = mesh.cell_data['elemTag']
    cells = mesh.cells
    offset = 0
    elements = []
    for i, n_points in enumerate(mesh.cell_sizes):
        # Only process triangles (3-point cells)
        if n_points == 3:
            pt1 = cells[offset + 1]
            pt2 = cells[offset + 2]
            pt3 = cells[offset + 3]
            tag = tags[i]
            elements.append((pt1, pt2, pt3, tag))
        offset += n_points + 1  # +1 for the size prefix in VTK format
    return elements
def write_elem_file(filename, elements):
    """Write a .elem file with triangle data."""
    with open(filename, 'w') as f:
        f.write(f"{len(elements)}\n")
        #f.write("Tr cell_point1 cell_point2 cell_point3 elemTag_of_cell\n")
        for pt1, pt2, pt3, tag in elements:
            f.write(f"Tr {pt1} {pt2} {pt3} {tag}\n")
def vtk2elem(mesh, elemfile):
    """
    Convert a VTK mesh to an .elem file format.
    Parameters:
    - mesh (pyvista.PolyData): Input mesh.
    - elemfile (str): Path to the output .elem file.
    """
    # Extract triangle cells with tags
    elements = get_triangle_cells_with_tags(mesh)
    # Write to .elem file
    write_elem_file(elemfile, elements)
    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VTK file to .lon format')
    parser.add_argument('vtk_file', help='Path to the input VTK file')
    parser.add_argument('lon_file', help='Path to the output .lon file')
    parser.add_argument('--array_name', default='endo_avg', help='Name of the array to extract from VTK file for .lon file')
    parser.add_argument('--ptsfile', default=None, help='Path to the output .pts file')
    parser.add_argument('--elemfile', default=None, help='Path to the output .elem file')
    args = parser.parse_args()
    vtkfile = args.vtk_file
    lonfile = args.lon_file
    array_name = args.array_name
    ptsfile = args.ptsfile
    elemfile= args.elemfile
    # Load your mesh (e.g., VTK, VTU, or OBJ format)
    #vtkfile="/data5/alee9/Dropbox/Work/Atria/GSTT/data/case001//9_PV/Monolayer//LA_noLAA_dilPVs_train_fibrosis.vtk"  # Replace with your actual file
    #lonfile="/data5/alee9/Dropbox/Work/Atria/GSTT/data/case001//9_PV/Monolayer//LA_noLAA_dilPVs_train_fibrosis.lon"
    #array_name='epi_avg'  # Replace with your actual array name
    # Read the VTK file
    mesh = pv.read(args.vtk_file)
    vtk2lon(mesh, args.lon_file, args.array_name)
    if elemfile is not None:
        elements = get_triangle_cells_with_tags(mesh)
        write_elem_file(elemfile, elements)
    if ptsfile is not None:
        vtk2pts(mesh, ptsfile)