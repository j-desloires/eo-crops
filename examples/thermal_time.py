import os
import copy
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from eocrops.inputs.meteoblue import WeatherDownload
from eocrops.climatools.format_data import WeatherPostprocess
from eocrops.climatools import resampling
from eocrops.tasks import curve_fitting
from eocrops.utils import base_functions as utils
from shapely.geometry import Point

###############################################################################################################
# Define the path to get store the data
PATH_EXPERIMENT = os.path.join("./data", "experiments")
TIME_INTERVAL = ("2017-08-15", "2018-07-15")

YEAR = TIME_INTERVAL[-1].split("-")[0]

if not os.path.exists(PATH_EXPERIMENT):
    os.mkdir(PATH_EXPERIMENT)

# Read the dataset
## Read the excel file
sentinel_data = pd.read_excel("./data/rapeseed_data_s1.xlsx")
sentinel_data = sentinel_data.sort_values(by=["ID", "band"])
## Define the features
features_data = [("DATA", f, f, "float32", 1) for f in sentinel_data["band"].unique()]

## Define the meta data
### Get the informations regarding the location
meta_data = sentinel_data[["ID", "lon", "lat"]].drop_duplicates()
### Add in the metadata file the dates of interest in the time series
meta_data["timestamp"] = [TIME_INTERVAL]
### Convert the coordinates into Point
meta_data["geometry"] = [
    Point(x, y) for x, y in zip(meta_data["lon"], meta_data["lat"])
]
feature_vector = [(col_name, "string") for col_name in meta_data.columns]
np.save(file=f"{PATH_EXPERIMENT}/meta_data.npy", arr=meta_data.values)

## Define the 3D array from Sentinel-2 data
### Keep only band values
sentinel_bands = sentinel_data.drop(["ID", "lon", "lat", "band"], axis=1)
dates = sentinel_bands.columns
sentinel_values = sentinel_bands.values
### Reshape in 3D array
array_data = sentinel_values.reshape(
    sentinel_values.shape[0] // len(features_data),
    len(features_data),
    sentinel_values.shape[1],
)
array_data = np.moveaxis(array_data, 1, 2)

### Check if the reshape worked well
np.alltrue(sentinel_data.iloc[0, :].values == array_data[0, :, 0])
### Plot the first observation
plt.plot(array_data[0, :, 0])
plt.show()

## Get the day of the year of the samples
dates = [datetime.datetime.strptime(str(k), "%Y%m%d") for k in dates]
## Subset for the season of interest
### Usually for rapeseed, season start 15 August previous year
start_season = datetime.datetime.strptime(TIME_INTERVAL[0], "%Y-%m-%d")
### And harvest is up to mid-July
end_season = datetime.datetime.strptime(TIME_INTERVAL[1], "%Y-%m-%d")
subset_dates = [k for k in dates if k >= start_season and k <= end_season]
len(subset_dates)


########################################################################################
# Check the dates and resample the data over fixed periods
def days_since_first(dates):
    """Compute number of days since the start of the season"""
    first_date = dates[0]
    days_since = [(d - first_date).days for d in dates]
    return np.array(days_since)


def is_evenly_spaced(lst, step):
    """Check if day of the year are evenly spaced"""
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] != step:
            print(lst[i])
            return False
    return True


# Day of the year since start of season (15th August)
subset_dates_doy = days_since_first(subset_dates)
array_data = array_data[:, : len(subset_dates_doy), :]
# Check if doy are evenly spaced, otherwise we need to resample
is_evenly_spaced(subset_dates_doy, subset_dates_doy[1] - subset_dates_doy[0])
# Resample the data
curve_fit = curve_fitting.CurveFitting(
    range_doy=(subset_dates_doy[0], subset_dates_doy[-1])
)

new_dates_doy, array_resample = curve_fit.resample_ts(
    doy=subset_dates_doy,
    ts_mean_=array_data,
    resampling=6,  # Resample every 6 days
)
# Check the results
plt.plot(subset_dates_doy, array_data[0, :, 1])
plt.plot(new_dates_doy, array_resample[0, :, 1])
plt.show()

# If you want to smooth the data
# array_resample = curve_fit.fit_whitakker(array_resample, degree_smoothing=1)


########################################
# Save the data
np.save(file=f"{PATH_EXPERIMENT}/satellite_data.npy", arr=array_resample)

###############################################################################################################
# Extract the weather data
## Download the data using meteoblue API
pipeline_cehub = WeatherDownload(
    api_key="",  # Please put your meteoblue API here
    shapefile=meta_data,
    id_column="ID",  # column from the meta_data with the ID of the field
    timestamp_column="timestamp",  # timestamp of interest
)

query_base = [
    {"domain": "ERA5T", "gapFillDomain": "ERA5", "timeResolution": "daily", "codes": []}
]

query_sum = copy.deepcopy(query_base)
query_sum[0]["codes"].extend(
    [
        {
            "code": 731,
            "level": "2 m above gnd",
            "aggregation": "sum",
            "gddBase": 0,  # Tmin for rapeseed
            "gddLimit": 30,  # Tmax for rapeseed
        }
    ]
)
weather_data = pipeline_cehub.execute(query=query_sum)
weather_data.to_csv(f"{PATH_EXPERIMENT}/GDD_data.csv", index=False)

## Format the data to compute accumulated GDUs
weather_data = pd.read_csv(f"{PATH_EXPERIMENT}/GDD_data.csv")

pipeline_refactor = WeatherPostprocess(
    shapefile=meta_data,
    id_column="ID",
    resample_range=(
        TIME_INTERVAL[0],
        TIME_INTERVAL[1],
        1,
    ),  # You can even resample it using fixed periods of day (e.g. every 8 day)
)

daily_gdus = pipeline_refactor.execute(
    df_weather=weather_data, stat="sum", return_pivot=True
)
daily_gdus.to_csv(f"{PATH_EXPERIMENT}/GDD_data.csv", index=False)

###############################################################################################################
# Thermal time resampling

gdu = resampling.TempResampling(
    range_dates=TIME_INTERVAL,  #  extracted from weather data
    stop=TIME_INTERVAL[
        -1
    ],  # Stopping date for yield prediction model (useful for in season)
    id_column="ID",  # Ids of the observations, which is the saving path of EOPatch
    smooth=True,  # Apply smoothing using sg filter
    period_nas_rate=0.2,
)

# You must load those 3 files that you did in the step data_loader to do the resampling
## Meta data
meta_data = gdu.load_meta_data(
    feature_vector=feature_vector, filepath=f"{PATH_EXPERIMENT}/meta_data.npy"
)
## Weather data
weather_data = gdu.load_weather_data(filepath=f"{PATH_EXPERIMENT}/GDD_data.csv")
## Satellite data
x_s1 = gdu.load_sat_data(filepath=f"{PATH_EXPERIMENT}/satellite_data.npy")

# Define the parameters for temporal resampling
kwargs_input = dict(
    stat="mean",  # aggregation over GDD periods (mean)
    thermal_time=True,  # resample over thermal time? otherwise, calendar time
    features_data=features_data,
    increment=120,  # Number of GDDs per period (if thermal_time=False, you can specify 10 for 10-day period)
    period_length=6,
)

# Prepare the output file with VH and VV resampled over thermal time
ds = pd.DataFrame()
for _, _, fname_, _, num in features_data:
    # Resample satellite data for a given fname (VV or VH)
    output, _ = gdu.resample_s2(**kwargs_input, fname=fname_)
    # Contenate all the bands
    ds = utils.concatenate_outputs(ds, output, fname_, id_column="ID")

cols_vv = [k for k in ds.columns if 'VV' in k]
# Save the output file
ds.to_csv(os.path.join(PATH_EXPERIMENT, "rapeseed_data_thermal_time.csv"), index=False)
