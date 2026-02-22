"""Inspect ERA5 NetCDF file structure and save to file"""
import xarray as xr
import sys

output_file = "netcdf_structure.txt"

with open(output_file, 'w') as f:
    sys.stdout = f
    
    print("Inspecting ERA5 NetCDF files...\n")
    
    files = [
        "Dataset/data_stream-oper_stepType-instant.nc",
        "Dataset/data_stream-oper_stepType-accum.nc"
    ]
    
    for file_path in files:
        print(f"{'='*70}")
        print(f"File: {file_path}")
        print(f"{'='*70}")
        
        ds = xr.open_dataset(file_path)
        
        print(f"\nVariables: {list(ds.data_vars)}")
        print(f"Coordinates: {list(ds.coords)}")
        print(f"Dimensions: {dict(ds.dims)}")
        
        print(f"\nDataset info:")
        print(ds)
        
        # Print first few values of each variable
        for var in ds.data_vars:
            print(f"\n{var} shape: {ds[var].shape}")
            print(f"{var} first values: {ds[var].values.flat[:5]}")
        
        print("\n")

sys.stdout = sys.__stdout__
print(f"NetCDF structure saved to {output_file}")
