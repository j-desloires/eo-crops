import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from eolearn.core import EOPatch, FeatureType, EOTask
from eocrops.utils import data_loader

#################################################################################

# Aggregate all the EOPatch saved into a root directory into a single 3D np.array, where each observation is an aggregated multiviariate time series
# The objective is to get a dataset for machine learning project (e.g. yield prediction) where we work at the object level (e.g. averaged time series)

root_dir = "/home/johann/Documents/DATA/EOPatch samples"
os.listdir(root_dir)

# Features of EOPatch to load in the final dataset
features_data = [
    (FeatureType.DATA, "LAI", "LAI", "float32", 1),
    (FeatureType.DATA, "fapar", "fapar", "float32", 1),
    (FeatureType.DATA, "Cab", "Cab", "float32", 1),
]


# Read EOPatch into a 3D np.array, where each observation is the median time series of the field

f = os.listdir("/home/johann/Documents/DATA/EOPatch samples")[1]
path = os.path.join("/home/johann/Documents/DATA/EOPatch samples", f)
patch = EOPatch.load(path)

file = pd.read_csv(
    os.path.join("/home/johann/Documents/DATA/prod_shp", "polygon_process_output.csv")
)
file["path"] = [
    os.path.join("/home/johann/Documents/DATA/EOPatch samples", f + "_S2_L2A")
    for f in file["key"]
]


pipeline_eopatch_tfds = data_loader.EOPatchDataset(
    root_dir=root_dir,  # root directory where EOPatch are saved
    features_data=features_data,  # features to read in EOPatch
    suffix="_S2_L2A",  # suffix for eopatch file names to filter out only one data source (e.g. S2_L2A). Put '' if you want to read all the files in root_dir
    resampling=dict(
        start="-01-01", end="-12-31", day_periods=8
    ),  # resample eopatch into 8 days period
    range_doy=(
        100,
        300,
    ),  # subset in term of doy to apply doubly logistic only on a subset period of the year
    function=np.nanmedian,  # get average time series from the field
)

path = (
    "/home/johann/Documents/DATA/EOPatch samples/1225-IIJ6395DGGNFA6657ZL-2021_S2_L2A"
)

# Get the 3D array dataset with 'cubic' interpolation over the resampling period
##If we do not have planting and/or harvest date
npy_eopatch_ = pipeline_eopatch_tfds.get_eopatch_tfds(
    algorithm="cubic",
    meta_file=file,
    path_column="path",
    planting_date_column="days_planting",
    harvest_date_column="days_harvest",
)

## If we have planting and/or harvest date
npy_eopatch = pipeline_eopatch_tfds.get_eopatch_tfds(
    algorithm="cubic",
    doubly_logistic=True,
    return_params=False,
    meta_file=file,
    path_column="path",
    planting_date_column="days_planting",
    harvest_date_column="days_harvest",
)

# Plot
import time

obs = 1
for i in range(len(features_data)):
    print(i)
    time.sleep(1)
    plt.plot(npy_eopatch[obs, :, i])
    plt.plot(npy_eopatch_[obs, :, i])
    plt.show()

# Create an example of vector dataset with auxilliary data. It must have a column 'path' to match with the corresponding EOPatch
dict_df = pd.DataFrame(
    dict(
        biomass=[np.random.normal(1) for k in range(len(os.listdir(root_dir)))],
        path=[os.path.join(root_dir, k) for k in os.listdir(root_dir)],
    )
)


# Features from vector file which contains auxilliary data (e.g. labels for crop classification)
feature_vector = [("biomass", "float32"), ("path", "string")]

# Get corresponding labels from the vector file dict_df
npy_labels = pipeline_eopatch_tfds.get_vector_tfds(
    vector_data=dict_df, features_list=feature_vector, column_path="path"
)
