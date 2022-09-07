from eolearn.geometry import VectorToRasterTask
from eolearn.core import FeatureType, EOTask
import sentinelhub
from eolearn.features.interpolation import LinearInterpolationTask, CubicInterpolationTask
import copy

import eocrops.utils.utils as utils
from eolearn.geometry.morphology import ErosionTask

from scipy.optimize import curve_fit
import numpy as np
from eolearn.core import RemoveFeatureTask


class PolygonMask(EOTask):
    """
    EOTask that performs rasterization from an input shapefile into :
    - data_timeless feature 'FIELD_ID' (0 nodata; 1,...,N for each observation of the shapefile ~ object IDs)
    - mask_timeless feature 'polygon_mask' (0 if pixels outside the polygon(s) from the shapefile, 1 otherwise

    Parameters
    ----------
    geodataframe : TYPE GeoDataFrame
        Input geodataframe read as GeoDataFrame, each observation represents a polygon (e.g. fields)
    new_feature_name : TYPE string
        Name of the new features which contains clustering_task predictions

    Returns
    -------
    EOPatch
    """

    def __init__(self, geodataframe):
        self.geodataframe = geodataframe

    def execute(self, eopatch):

        # Check CRS and transform into UTM
        self.geodataframe = utils.check_crs(self.geodataframe)

        # Get an ID for each polygon from the input shapefile
        self.geodataframe['FIELD_ID'] = list(range(1, self.geodataframe.shape[0] + 1))

        if self.geodataframe.shape[0] > 1:
            bbox = self.geodataframe.geometry.total_bounds
            polygon_mask = sentinelhub.BBox(bbox=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], crs=self.geodataframe.crs)
            self.geodataframe['MASK'] = polygon_mask.geometry
        else:
            self.geodataframe['MASK'] = self.geodataframe['geometry']

        self.geodataframe['polygon_bool'] = True

        rasterization_task = VectorToRasterTask(self.geodataframe, (FeatureType.DATA_TIMELESS, "FIELD_ID"),
                                                values_column="FIELD_ID", raster_shape=(FeatureType.MASK, 'IS_DATA'),
                                                raster_dtype=np.uint16)
        eopatch = rasterization_task.execute(eopatch)

        rasterization_task = VectorToRasterTask(self.geodataframe,
                                                (FeatureType.MASK_TIMELESS, "MASK"),
                                                values_column="polygon_bool",
                                                raster_shape=(FeatureType.MASK, 'IS_DATA'),
                                                raster_dtype=np.uint16)
        eopatch = rasterization_task.execute(eopatch)

        eopatch.mask_timeless['MASK'] = eopatch.mask_timeless['MASK'].astype(bool)

        return eopatch


class MaskPixels(EOTask):
    def __init__(self, features, fname='MASK'):
        '''
        Parameters
        ----------
        feature (list): of features in data and/or data_timeless
        fname (str): name of the mask
        '''
        self.features = features
        self.fname = fname

    @staticmethod
    def _filter_array(patch, ftype, fname, mask):
        ivs = patch[ftype][fname]

        arr0 = np.ma.array(ivs,
                           dtype=np.float32,
                           mask=(1 - mask).astype(bool),
                           fill_value=np.nan)

        arr0 = arr0.filled()
        patch[ftype][fname] = arr0

        return patch

    def execute(self, patch, erosion=0):
        copy_patch = copy.deepcopy(patch)
        times = len(patch.timestamp)
        if erosion:
            erode = ErosionTask(mask_feature=(FeatureType.MASK_TIMELESS, self.fname),
                                disk_radius=erosion)
            erode.execute(copy_patch)

        crop_mask = copy_patch["mask_timeless"][self.fname]
        # Filter the pixels of each features
        for index in self.features:
            if index in list(patch.data.keys()):
                ftype = 'data'
                shape = patch[ftype][index].shape[-1]
                mask = crop_mask.reshape(1, crop_mask.shape[0], crop_mask.shape[1], 1)
                mask = [mask for k in range(times)]
                mask = np.concatenate(mask, axis=0)
                mask = [mask for k in range(shape)]
                mask = np.concatenate(mask, axis=-1)
            else:
                ftype = 'data_timeless'
                mask = crop_mask
            patch = self._filter_array(patch, ftype, index, mask)

        return patch


class InterpolateFeatures(EOTask):
    def __init__(self, resampled_range,
                 features=None,
                 algorithm='linear',
                 copy_features=None):
        self.resampled_range = resampled_range
        self.features = features
        self.algorithm = algorithm
        self.copy_features = copy_features

    def _interpolate_feature(self, eopatch, features, mask_feature):

        kwargs = dict(mask_feature=mask_feature,
                      resample_range=self.resampled_range,
                      feature=features,
                      bounds_error=False)

        if self.resampled_range is not None:
            kwargs['copy_features'] = self.copy_features

        if self.algorithm == 'linear':
            interp = LinearInterpolationTask(
                parallel=True,
                **kwargs
            )
        elif self.algorithm == 'cubic':
            interp = CubicInterpolationTask(
                **kwargs
            )
        eopatch = interp.execute(eopatch)
        return eopatch

    def execute(self, eopatch):

        '''Gap filling after data extraction, very useful if did not include it in the data extraction workflow'''

        mask_feature = None
        if 'VALID_DATA' in list(eopatch.mask.keys()):
            mask_feature = (FeatureType.MASK, 'VALID_DATA')

        if self.features is None:
            self.features = [(FeatureType.DATA, fname) for fname in eopatch.get_features()[FeatureType.DATA]]

        dico = {}
        for ftype, fname in self.features:
            new_eopatch = copy.deepcopy(eopatch)
            new_eopatch = self._interpolate_feature(new_eopatch, (ftype, fname), mask_feature)
            dico[fname] = new_eopatch[ftype][fname]

        eopatch['data'] = dico
        t, h, w, _ = dico[fname].shape
        eopatch.timestamp = new_eopatch.timestamp
        eopatch['mask']['IS_DATA'] = (np.zeros((t, h, w, 1)) + 1).astype(int)
        eopatch['mask']['VALID_DATA'] = (np.zeros((t, h, w, 1)) + 1).astype(bool)
        if "CLM" in eopatch.mask.keys():
            remove_feature = RemoveFeatureTask([(FeatureType.MASK, "CLM")])
            remove_feature.execute(eopatch)

        return eopatch


class CurveFitting(EOTask):
    def __init__(self, range_doy=None):
        self.range_doy = range_doy
        self.params = None

    def get_time_series_profile(self,
                                eopatch,
                                feature,
                                feature_mask='polygon_mask',
                                function=np.nanmedian):

        feature_array = eopatch[FeatureType.DATA][feature]
        if feature_mask not in eopatch[FeatureType.MASK_TIMELESS].keys():
            raise ValueError('The feature ' + feature_mask + " is missing in MASK_TIMELESS")
        crop_mask = eopatch[FeatureType.MASK_TIMELESS][feature_mask]
        # Transform mask from 3D to 4D
        times, h, w, shape = feature_array.shape
        mask = crop_mask.reshape(1, h, w, 1)
        mask = [mask for k in range(times)]
        mask = np.concatenate(mask, axis=0)
        #######################
        mask = [mask for k in range(shape)]
        mask = np.concatenate(mask, axis=-1)
        ########################
        a = np.ma.array(feature_array, mask=np.invert(mask))
        ts_mean = np.ma.apply_over_axes(function, a, [1, 2])
        ts_mean = ts_mean.reshape(ts_mean.shape[0], ts_mean.shape[-1])
        if self.range_doy is not None:
            _, ids_filter = self.get_doy_period(eopatch)

            ts_mean[:ids_filter[0]] = 0.2
            ts_mean[ids_filter[-1]:] = 0.2

        ########################
        # Replace nas values with nearest valid value
        valid_values = np.where(~np.isnan(ts_mean))[0]
        ts_mean[:valid_values[0]] = ts_mean[valid_values[0]]
        ts_mean[valid_values[-1]:] = ts_mean[valid_values[-1]]

        return ts_mean

    def get_doy_period(self, eopatch):

        first_of_year = eopatch.timestamp[0].timetuple().tm_yday
        last_of_year = eopatch.timestamp[-1].timetuple().tm_yday

        times = np.asarray([time.toordinal() for time in eopatch.timestamp])
        times_ = (times - times[0]) / (times[-1] - times[0])
        times_doy = times_ * (last_of_year - first_of_year) + first_of_year

        if self.range_doy is not None:
            ids_filter = np.where((times_doy > int(self.range_doy[0])) &
                                  (times_doy < int(self.range_doy[1])))[0]
            #print(ids_filter)
            return times_doy, ids_filter
        else:
            return times_doy

    def _check_range_doy(self, doy, ts_mean, length_period=8):
        if doy[0] > self.range_doy[0]:
            nb_periods_add = int((doy[0] - self.range_doy[0]) // length_period)
            if nb_periods_add == 0:
                nb_periods_add += 1
            doy_add = [doy[0] - nb_periods_add * i for i in range(1, nb_periods_add + 1)]
            doy_add.sort()

            doy = np.concatenate([doy_add, doy], axis=0)
            ts_add = [ts_mean[0] * 0.5 / i for i in range(1, nb_periods_add + 1)] #
            ts_add.sort()
            ts_mean = np.concatenate([ts_add, ts_mean], axis=0)


        if doy[-1] < self.range_doy[1]:
            nb_periods_add = int((self.range_doy[-1] - doy[-1]) // length_period)
            if nb_periods_add == 0:
                nb_periods_add += 1
            doy_add = [doy[-1] + nb_periods_add * i for i in range(1, nb_periods_add + 1)]
            doy_add.sort()

            doy = np.concatenate([doy, doy_add], axis=0)
            ts_add = [ts_mean[-1] * 0.5 / i for i in range(1, nb_periods_add + 1)]
            ts_add.sort()
            ts_mean = np.concatenate([ts_add, ts_mean], axis=0)

        ts_mean[0] = 0.2
        ts_mean[-1] = 0.2

        return doy, ts_mean.flatten()

    def _instance_inputs(self, eopatch, feature,
                         feature_mask='polygon_mask',
                         function=np.nanmedian):

        y = self.get_time_series_profile(eopatch, feature, feature_mask, function)

        if self.range_doy is not None:
            times_doy, _ = self.get_doy_period(eopatch)
        else:
            times_doy = self.get_doy_period(eopatch)

        x = (times_doy - self.range_doy[0]) / (self.range_doy[-1] - self.range_doy[0])
        ids = np.where(((x>0) & (x<1)))[0]

        return times_doy[ids], x[ids], y.flatten()[ids]



class AsymmetricGaussian(CurveFitting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _asym_gaussian(middle, initial_value, scale, a1, a2, a3, a4, a5):
        return initial_value + scale * np.piecewise(
            middle,
            [middle < a1, middle >= a1],
            [lambda y: np.exp(-(((a1 - y) / a4) ** a5)), lambda y: np.exp(-(((y - a1) / a2) ** a3))],
        )

    def _fit_optimize_asym(self, x_axis, y_axis, initial_parameters=None):
        bounds_lower = [
            np.quantile(y_axis, 0.1),
            -np.inf,
            x_axis[0],
            0,
            1,
            0,
            1,
        ]
        bounds_upper = [
            np.max(y_axis),
            np.inf,
            x_axis[-1],
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ]
        if initial_parameters is None:
            initial_parameters = [np.mean(y_axis), 0.2,
                                  x_axis[np.argmax(y_axis)],
                                  0.15, 10, 0.2, 5]

        popt, pcov = curve_fit(
            self._asym_gaussian,
            x_axis,
            y_axis,
            initial_parameters,
            bounds=(bounds_lower, bounds_upper),
            maxfev=10e10,
            absolute_sigma=True,
            #method =  'trf'
        )
        self.params = popt

        return popt

    def execute(self, eopatch, feature, feature_mask='polygon_mask',
                function=np.nanmedian, seeding_doy = 0,
                harvest_doy = 0, resampling = 0):

        times_doy, x, y = self._instance_inputs(eopatch, feature,
                                                feature_mask,
                                                function)

        initial_value, scale, a1, a2, a3, a4, a5 = self._fit_optimize_asym(x, y)

        if resampling>0:
            times_doy = np.array(range(0, 365, resampling))
            x = (times_doy - self.range_doy[0]) / (self.range_doy[-1] - self.range_doy[0])
            x[x < 0] = 0
            x[x > 1] = 1
            fitted = self._asym_gaussian(x, initial_value, scale, a1, a2, a3, a4, a5)
        else:
            fitted = self._asym_gaussian(x, initial_value, scale, a1, a2, a3, a4, a5)

        return times_doy, fitted




class DoublyLogistic(CurveFitting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _doubly_logistic(x, vb, ve, k, c, p, d, q):
        return vb + (k / (1 + np.exp(-c * (x - p)))) - ((k + vb + ve) / (1 + np.exp(d * (x - q))))


    def _fit_optimize_doubly(self, x_axis, y_axis, initial_parameters=None):

        popt, _ = curve_fit(self._doubly_logistic,
                            x_axis, y_axis,
                            method='lm',
                            p0=initial_parameters,  # np.array([0.5, 6, 0.2, 150, 0.23, 240]),
                            maxfev=int(10e6),
                            bounds=[-np.Inf, np.Inf])

        self.params = popt

        return popt

    def execute(self, eopatch, feature, feature_mask='polygon_mask',
                function=np.nanmedian, seeding_doy = 0,
                harvest_doy = 0, resampling = 0):

        times_doy, x, y = self._instance_inputs(eopatch, feature,
                                                feature_mask,
                                                function)

        vb, ve, k, c, p, d, q = self._fit_optimize_doubly(x, y)

        if resampling>0:
            times_doy = np.array(range(0, 365, resampling))
            x = (times_doy - self.range_doy[0]) / (self.range_doy[-1] - self.range_doy[0])
            x[x < 0] = 0
            x[x > 1] = 1
            fitted = self._doubly_logistic(x, vb, ve, k, c, p, d, q)
        else:
            fitted = self._doubly_logistic(x, vb, ve, k, c, p, d, q)

        return times_doy, fitted