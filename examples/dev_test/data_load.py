import pandas as pd
from eolearn.core import EOPatch, FeatureType, EOTask
import numpy as np
import os
from eocrops.utils import data_loader
import matplotlib.pyplot as plt
import pandas as pd

#################################################################################

# Aggregate all the EOPatch saved into a root directory into a single 3D np.array, where each observation is an aggregated multiviariate time series
# The objective is to get a dataset for machine learning project (e.g. yield prediction) where we work at the object level (e.g. averaged time series)

#################################################################################
# Features of EOPatch to load in the final dataset
features_data = [
    (FeatureType.DATA, "LAI", "LAI", "float32", 1),
    (FeatureType.DATA, "fapar", "fapar", "float32", 1),
    (FeatureType.DATA, "Cab", "Cab", "float32", 1),
]


root_dir = "/home/johann/Documents/DATA/EOPatch samples/"
files = os.listdir(root_dir)

# Create an example of vector dataset with auxilliary data. It must have a column 'path' to match with the corresponding EOPatch
meta_file = pd.DataFrame(
    dict(
        biomass=[np.random.normal(1) for i in files],
        path=[os.path.join(root_dir, f) for f in files],
        year=[2017 for i in files],
        days_planting=[100 for i in files],
        days_harvest=[300 for i in files],
    )
)

feature_vector = [
    ("biomass", "float32"),
    ("days_planting", "int16"),
    ("days_harvest", "int16"),
    ("path", "string"),
]

#################################################################################
# Data loader pipeline
pipeline_eopatch_tfds = data_loader.EOPatchDataset(
    root_dir=root_dir,  # root directory where EOPatch are saved
    features_data=features_data,  # features to read in EOPatch
    suffix="_S2_L2A",  # suffix for eopatch file names to filter out only one data source (e.g. S2_L2A). Put '' if you want to read all the files in root_dir
    resampling=dict(
        start="01-01", end="12-31", day_periods=8
    ),  # resample eopatch into 8 days period
    range_doy=(
        100,
        300,
    ),  # subset in term of doy to apply doubly logistic only on a subset period of the year
    function=np.nanmedian,  # get average time series from the field
)


# Get the 3D array dataset with interpolation over the resampling period
##Load data and resample time sereis using cubic intepolation
npy_eopatch = pipeline_eopatch_tfds.get_eopatch_tfds(
    algorithm="cubic", meta_file=meta_file, path_column="path"
)


with open(
    os.path.join("./predfin/yield_in_season/tests/dataset", "X_S2.npy"), "wb"
) as f:
    np.save(f, npy_eopatch)


##Load data and resample time series using doubly logistic intepolation
npy_eopatch_doubly = pipeline_eopatch_tfds.get_eopatch_tfds(
    algorithm="cubic",
    doubly_logistic=True,
    meta_file=meta_file,
    path_column="path",
    planting_date_column="days_planting",
    harvest_date_column="days_harvest",
)

##Load data and resample time series using asymmetric gaussian intepolation
npy_eopatch_asym = pipeline_eopatch_tfds.get_eopatch_tfds(
    algorithm="cubic",
    asym_gaussian=True,
    meta_file=meta_file,
    path_column="path",
    planting_date_column="days_planting",
    harvest_date_column="days_harvest",
)


plt.plot(npy_eopatch[0, :, 0])
plt.plot(npy_eopatch_doubly[0, :, 0])
plt.plot(npy_eopatch_asym[0, :, 0])
plt.legend(["original", "doubly", "asym"], loc="lower right")
plt.show()
