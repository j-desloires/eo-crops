from eolearn.core import EOPatch
import numpy as np
from eocrops.tasks import preprocessing
import matplotlib.pyplot as plt
import os
from scipy import interpolate

##########################################################################################
list_files = os.listdir('/home/johann/Documents/DATA/EOPatch samples')
f = list_files[1]
eopatch = EOPatch.load('/home/johann/Documents/DATA/EOPatch samples/' + f)

curve_fit_task = preprocessing.DoublyLogistic(range_doy=(120, 250))
doy_, _ = curve_fit_task.get_doy_period(eopatch)
self = curve_fit_task

ts_mean = curve_fit_task.get_time_series_profile(eopatch, feature='fapar', feature_mask='polygon_mask').flatten()
doy, fitted = curve_fit_task.execute(eopatch, feature='fapar', feature_mask='polygon_mask')

curve_fit_asym = preprocessing.AsymmetricGaussian(range_doy=(120, 250))
doy, fitted_asym = curve_fit_asym.execute(eopatch, feature='fapar', feature_mask='polygon_mask')

plt.plot(doy, fitted_asym)
plt.plot(doy, fitted)
plt.plot(doy_, ts_mean)
plt.show()


##########################################################################
##########################################################################################


flinear = interpolate.interp1d(doy, ts_mean, kind='cubic')
flinear_fitted = interpolate.interp1d(doy, fitted, kind='cubic')
flinear_cspline = interpolate.Akima1DInterpolator(doy, fitted)

xnew = np.arange(0, 365, 8)
len(xnew)
s = np.where(xnew>=doy[0])[0][0]
e = np.where(xnew<=doy[-1])[0][-1]

ylinear_ts = flinear(xnew[s:e])
ylinear_fitted = flinear_fitted(xnew[s:e])
ylinear_cspline = flinear_cspline(xnew[s:e])

plt.plot(ylinear_ts)
plt.plot(ylinear_fitted)
plt.plot(ylinear_cspline)
plt.show()

before_season = np.repeat(ylinear_cspline[0], xnew[:s].shape[0])
after_season = np.repeat(ylinear_cspline[-1], xnew[e:].shape[0])
output = np.concatenate([before_season, ylinear_cspline, after_season],axis = 0)

plt.plot(xnew, output)
plt.show()