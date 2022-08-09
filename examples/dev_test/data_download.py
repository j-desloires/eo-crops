
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
from scipy.signal import savgol_filter

import os
import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import FeatureType

from eocrops.input import utils_sh as utils_sh
from eocrops.input import sentinel1 as sentinel1
from eocrops.input import sentinel2 as sentinel2
from eocrops.tasks import cmd_otb as cmd_otb
from eocrops.tasks import preprocessing as preprocessing

dir_path = os.path.dirname(os.getcwd())
print(dir_path)
# read microplot data
shapefile_input = gpd.read_file(os.path.join(dir_path, 'eo-crops/examples/layers/POLYGON.shp'))

api = ''
client_id = ''
client_secret = ''


config = utils_sh.config_sentinelhub_cred(api, client_id, client_secret)
# Provide here your planet API key
config.planet_key = ''

# %%

time_period = ('2020-02-15', '2020-08-15')
kwargs = dict(polygon=shapefile_input,
              time_stamp=time_period,
              config=config)



os.getcwd()
warnings.filterwarnings("ignore")
patch = sentinel2.workflow_instructions_S2L2A(**kwargs,
                                              path_out='/home/johann/Documents/patch',  # you can specify here a path to save the EOPatch object
                                              coverage_predicate=0.5,
                                              interpolation={'interpolate': True, 'period_length' : 8})

curve_fit = preprocessing.CurveFitting(range_doy=(100, 365))
ts_mean = curve_fit.get_time_series_profile(patch,feature='LAI', feature_mask='MASK').flatten()
ts_mean[:2] = np.nan

valid_values = np.where(~np.isnan(ts_mean))[0]
ts_mean[:valid_values[0]] = ts_mean[valid_values[0]]
ts_mean[valid_values[-1]:] = ts_mean[valid_values[-1]]

fitted = curve_fit.execute(eopatch, feature='LAI')