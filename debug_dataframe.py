"""Debug DataFrame structure"""
import xarray as xr
import pandas as pd

file_path = "Dataset/data_stream-oper_stepType-instant.nc"

# Load and process
ds = xr.open_dataset(file_path)
temp = ds['t2m']
temp_location = temp.sel(latitude=28.5, longitude=76.5, method='nearest')
temp_celsius = temp_location - 273.15

# Convert to DataFrame
df = temp_celsius.to_dataframe()

print("DataFrame info:")
print(df.info())
print("\nDataFrame head:")
print(df.head(10))
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nColumn types: {df.dtypes}")

# Check if there are any non-numeric columns
print(f"\nFirst column name: '{df.columns[0]}'")
print(f"First column values (first 5): {df[df.columns[0]].values[:5]}")

# Try resampling
try:
    daily = df[df.columns[0]].resample('D').mean()
    print(f"\nDaily resampling successful!")
    print(f"Daily shape: {daily.shape}")
    print(f"Daily values (first 5): {daily.values[:5]}")
except Exception as e:
    print(f"\nError during resampling: {e}")
    import traceback
    traceback.print_exc()
