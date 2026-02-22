"""Test hourly to daily conversion"""
import xarray as xr
import pandas as pd

file_path = "Dataset/data_stream-oper_stepType-instant.nc"

# Load dataset
ds = xr.open_dataset(file_path)
temp = ds['t2m']
temp_location = temp.sel(latitude=28.5, longitude=76.5, method='nearest')

# Convert to Celsius
temp_celsius = temp_location - 273.15

print(f"Hourly data shape: {temp_celsius.shape}")
print(f"Hourly data type: {type(temp_celsius)}")
print(f"First 5 values: {temp_celsius.values[:5]}")

# Convert to DataFrame
df = temp_celsius.to_dataframe()
print(f"\nDataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"DataFrame index type: {type(df.index)}")
print(f"\nDataFrame head:")
print(df.head())

# Resample to daily
daily = df['t2m'].resample('D').mean()
print(f"\nDaily data shape: {daily.shape}")
print(f"Daily data type: {type(daily)}")
print(f"First 5 daily values: {daily.values[:5]}")
print(f"Daily min: {daily.min():.2f}°C")
print(f"Daily max: {daily.max():.2f}°C")
print(f"Daily mean: {daily.mean():.2f}°C")
