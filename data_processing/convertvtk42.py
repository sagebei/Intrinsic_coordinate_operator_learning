from vtk import *
import argparse

# Needs vtk 9.1.0

def convert42(invtk, outvtk, grid=False, binary=True): 
    
    if grid: 
        reader=vtkUnstructuredGridReader(); # type: ignore
        reader.SetFileName(invtk)
        reader.Update()
        
        writer = vtkUnstructuredGridWriter()
        if binary: 
            writer.SetFileTypeToBinary()
        writer.SetInputData(reader.GetOutput())
        writer.SetFileName(outvtk)
        writer.SetFileVersion(42)
        writer.Write()
    else: 
        #reader=vtkPolyDataReader();
        reader = vtkDataSetReader()
        reader.SetFileName(invtk)
        reader.Update()
        geomFilter=vtkGeometryFilter()
        geomFilter.SetInputConnection(reader.GetOutputPort())
        geomFilter.Update()

        writer = vtkPolyDataWriter()
        #writer.SetFileTypeToBinary()
        writer.SetInputData(geomFilter.GetOutput())
        writer.SetFileName(outvtk)
        writer.SetFileVersion(42)
        writer.Write()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert vtk file format to 4.2 version (works for MITK2018)')

    parser.add_argument('invtk', help='input vtk file')
    
    parser.add_argument('outvtk', help='output vtk file')
    parser.add_argument('-grid', action='store_true', help='Save as a unstructured grid instead of a polydata')
    
    args = parser.parse_args()

    convert42(args.invtk, args.outvtk, args.grid)
