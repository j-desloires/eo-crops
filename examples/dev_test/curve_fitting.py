from eolearn.core import EOPatch
import numpy as np
from eocrops.tasks import preprocessing
import matplotlib.pyplot as plt
import os
from scipy import interpolate

##########################################################################################
list_files = os.listdir('/home/johann/Documents/DATA/EOPatch samples')
f = list_files[0]
eopatch = EOPatch.load('/home/johann/Documents/DATA/EOPatch samples/' + f)

curve_fit_task = preprocessing.DoublyLogistic(range_doy=(100, 300))
doy_, _ = curve_fit_task.get_doy_period(eopatch)
curve_fit_task.params



ts_mean = curve_fit_task.get_time_series_profile(eopatch, feature='fapar', feature_mask='polygon_mask').flatten()
doy, fitted = curve_fit_task.execute(eopatch, feature='fapar', feature_mask='polygon_mask')

curve_fit_asym = preprocessing.AsymmetricGaussian(range_doy=(100, 300))
doy, fitted_asym = curve_fit_asym.execute(eopatch, feature='fapar', feature_mask='polygon_mask')
curve_fit_asym.params

ts_mean[ts_mean<0] = 0
plt.plot(doy, fitted_asym)
plt.plot(doy, fitted)
plt.plot(doy_, ts_mean)
plt.show()


##########################################################################
##########################################################################################
