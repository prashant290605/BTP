"""
ERA5 Climate Data Loader

Loads and preprocesses ERA5 reanalysis data for early warning system analysis.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


def load_era5_temperature(file_path, lat=28.5, lon=76.5):
    """
    Load 2m temperature from ERA5 NetCDF file.
    
    Args:
        file_path: Path to NetCDF file
        lat: Latitude (default: Delhi ~28.5°N)
        lon: Longitude (default: Delhi ~76.5°E)
        
    Returns:
        xarray DataArray with temperature time series
    """
    # Load NetCDF
    ds = xr.open_dataset(file_path)
    
    # Extract t2m (2m temperature)
    temp = ds['t2m']
    
    # Select nearest location (Delhi region)
    temp_location = temp.sel(latitude=lat, longitude=lon, method='nearest')
    
    return temp_location


def hourly_to_daily(hourly_data):
    """
    Convert hourly data to daily averages.
    
    Args:
        hourly_data: xarray DataArray with hourly time series
        
    Returns:
        pandas Series with daily averages
    """
    # Convert to pandas Series directly
    series = hourly_data.to_series()
    
    # Resample to daily mean
    daily = series.resample('D').mean()
    
    return daily


def kelvin_to_celsius(temp_kelvin):
    """Convert temperature from Kelvin to Celsius."""
    return temp_kelvin - 273.15


def normalize_series(data, return_params=False):
    """
    Z-score normalization.
    
    Args:
        data: pandas Series or numpy array
        return_params: If True, return (normalized, mean, std)
        
    Returns:
        Normalized data (and optionally mean, std)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    normalized = (data - mean) / std
    
    if return_params:
        return normalized, mean, std
    return normalized


def create_forecast_windows(data, window_size=30):
    """
    Create rolling windows for forecasting.
    
    Args:
        data: 1D array or Series
        window_size: Number of past days to use
        
    Returns:
        X: (n_samples, window_size) - input windows
        y: (n_samples,) - target next day
    """
    data = np.array(data)
    
    X = []
    y = []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    
    return np.array(X), np.array(y)


def load_and_preprocess_era5(
    file_path,
    lat=28.5,
    lon=76.5,
    convert_celsius=True
):
    """
    Complete pipeline: load ERA5, extract temperature, convert to daily.
    
    Args:
        file_path: Path to NetCDF file
        lat: Latitude
        lon: Longitude
        convert_celsius: Convert from Kelvin to Celsius
        
    Returns:
        pandas Series with daily temperature
    """
    # Load hourly temperature
    temp_hourly = load_era5_temperature(file_path, lat, lon)
    
    # Convert to Celsius BEFORE converting to daily
    if convert_celsius:
        temp_hourly = kelvin_to_celsius(temp_hourly)
    
    # Convert to daily
    temp_daily = hourly_to_daily(temp_hourly)
    
    # Handle missing values
    if temp_daily.isna().any():
        print(f"Warning: {temp_daily.isna().sum()} missing values found. Interpolating...")
        temp_daily = temp_daily.interpolate(method='linear')
    
    return temp_daily


if __name__ == "__main__":
    # Test data loading
    print("Testing ERA5 data loader...")
    
    file_path = "Dataset/data_stream-oper_stepType-instant.nc"
    
    try:
        temp_daily = load_and_preprocess_era5(file_path)
        
        print(f"\n✓ Loaded {len(temp_daily)} daily samples")
        print(f"  Date range: {temp_daily.index[0]} to {temp_daily.index[-1]}")
        print(f"  Temperature range: {temp_daily.min():.2f}°C to {temp_daily.max():.2f}°C")
        print(f"  Mean: {temp_daily.mean():.2f}°C")
        print(f"  Std: {temp_daily.std():.2f}°C")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
