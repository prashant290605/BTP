"""Debug ERA5 data loading"""
import xarray as xr
import numpy as np

file_path = "Dataset/data_stream-oper_stepType-instant.nc"

# Load dataset
ds = xr.open_dataset(file_path)

# Extract temperature
temp = ds['t2m']

print(f"Temperature shape: {temp.shape}")
print(f"Temperature values (first 10): {temp.values.flat[:10]}")
print(f"Temperature min: {temp.min().values}")
print(f"Temperature max: {temp.max().values}")
print(f"Temperature mean: {temp.mean().values}")

# Select location
lat, lon = 28.5, 76.5
temp_location = temp.sel(latitude=lat, longitude=lon, method='nearest')

print(f"\nLocation temperature shape: {temp_location.shape}")
print(f"Location temperature values (first 10): {temp_location.values[:10]}")
print(f"Location temperature min: {temp_location.min().values}")
print(f"Location temperature max: {temp_location.max().values}")

# Convert to Celsius
temp_celsius = temp_location - 273.15

print(f"\nCelsius values (first 10): {temp_celsius.values[:10]}")
print(f"Celsius min: {temp_celsius.min().values}")
print(f"Celsius max: {temp_celsius.max().values}")
