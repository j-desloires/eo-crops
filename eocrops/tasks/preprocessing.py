import copy
import sentinelhub

import numpy as np
import pandas as pd

from eolearn.geometry import VectorToRasterTask
from eolearn.core import FeatureType, EOTask


from eolearn.features.interpolation import (
    LinearInterpolationTask,
    CubicInterpolationTask,
)


import eocrops.utils.utils as utils
from eolearn.geometry.morphology import ErosionTask

from scipy.optimize import curve_fit

from eolearn.core import RemoveFeatureTask
from eolearn.core import FeatureType, EOTask
from scipy.optimize import curve_fit


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
        self.geodataframe["FIELD_ID"] = list(range(1, self.geodataframe.shape[0] + 1))

        if self.geodataframe.shape[0] > 1:
            bbox = self.geodataframe.geometry.total_bounds
            polygon_mask = sentinelhub.BBox(
                bbox=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], crs=self.geodataframe.crs
            )
            self.geodataframe["MASK"] = polygon_mask.geometry
        else:
            self.geodataframe["MASK"] = self.geodataframe["geometry"]

        self.geodataframe["polygon_bool"] = True

        rasterization_task = VectorToRasterTask(
            self.geodataframe,
            (FeatureType.DATA_TIMELESS, "FIELD_ID"),
            values_column="FIELD_ID",
            raster_shape=(FeatureType.MASK, "IS_DATA"),
            raster_dtype=np.uint16,
        )
        eopatch = rasterization_task.execute(eopatch)

        rasterization_task = VectorToRasterTask(
            self.geodataframe,
            (FeatureType.MASK_TIMELESS, "MASK"),
            values_column="polygon_bool",
            raster_shape=(FeatureType.MASK, "IS_DATA"),
            raster_dtype=np.uint16,
        )
        eopatch = rasterization_task.execute(eopatch)

        eopatch.mask_timeless["MASK"] = eopatch.mask_timeless["MASK"].astype(bool)

        return eopatch


class MaskPixels(EOTask):
    def __init__(self, features, fname="MASK"):
        """
        Parameters
        ----------
        feature (list): of features in data and/or data_timeless
        fname (str): name of the mask
        """
        self.features = features
        self.fname = fname

    @staticmethod
    def _filter_array(patch, ftype, fname, mask):
        ivs = patch[ftype][fname]

        arr0 = np.ma.array(
            ivs, dtype=np.float32, mask=(1 - mask).astype(bool), fill_value=np.nan
        )

        arr0 = arr0.filled()
        patch[ftype][fname] = arr0

        return patch

    def execute(self, patch, erosion=0):
        copy_patch = copy.deepcopy(patch)
        times = len(patch.timestamp)
        if erosion:
            erode = ErosionTask(
                mask_feature=(FeatureType.MASK_TIMELESS, self.fname),
                disk_radius=erosion,
            )
            erode.execute(copy_patch)

        crop_mask = copy_patch["mask_timeless"][self.fname]
        # Filter the pixels of each features
        for index in self.features:
            if index in list(patch.data.keys()):
                ftype = "data"
                shape = patch[ftype][index].shape[-1]
                mask = crop_mask.reshape(1, crop_mask.shape[0], crop_mask.shape[1], 1)
                mask = [mask for k in range(times)]
                mask = np.concatenate(mask, axis=0)
                mask = [mask for k in range(shape)]
                mask = np.concatenate(mask, axis=-1)
            else:
                ftype = "data_timeless"
                mask = crop_mask
            patch = self._filter_array(patch, ftype, index, mask)

        return patch


class InterpolateFeatures(EOTask):
    def __init__(
        self, resampled_range, features=None, algorithm="linear", copy_features=None
    ):
        self.resampled_range = resampled_range
        self.features = features
        self.algorithm = algorithm
        self.copy_features = copy_features

    def _interpolate_feature(self, eopatch, features, mask_feature):

        kwargs = dict(
            mask_feature=mask_feature,
            resample_range=self.resampled_range,
            feature=features,
            bounds_error=False,
        )

        if self.resampled_range is not None:
            kwargs["copy_features"] = self.copy_features

        if self.algorithm == "linear":
            interp = LinearInterpolationTask(parallel=True, **kwargs)
        elif self.algorithm == "cubic":
            interp = CubicInterpolationTask(**kwargs)
        eopatch = interp.execute(eopatch)
        return eopatch

    def execute(self, eopatch):

        """Gap filling after data extraction, very useful if did not include it in the data extraction workflow"""

        mask_feature = None
        if "VALID_DATA" in list(eopatch.mask.keys()):
            mask_feature = (FeatureType.MASK, "VALID_DATA")

        if self.features is None:
            self.features = [
                (FeatureType.DATA, fname)
                for fname in eopatch.get_features()[FeatureType.DATA]
            ]

        dico = {}
        for ftype, fname in self.features:
            new_eopatch = copy.deepcopy(eopatch)
            new_eopatch = self._interpolate_feature(
                new_eopatch, (ftype, fname), mask_feature
            )
            dico[fname] = new_eopatch[ftype][fname]

        eopatch["data"] = dico
        t, h, w, _ = dico[fname].shape
        eopatch.timestamp = new_eopatch.timestamp
        eopatch["mask"]["IS_DATA"] = (np.zeros((t, h, w, 1)) + 1).astype(int)
        eopatch["mask"]["VALID_DATA"] = (np.zeros((t, h, w, 1)) + 1).astype(bool)
        if "CLM" in eopatch.mask.keys():
            remove_feature = RemoveFeatureTask([(FeatureType.MASK, "CLM")])
            remove_feature.execute(eopatch)

        return eopatch


class CurveFitting:
    def __init__(self, range_doy=(1, 365)):
        self.range_doy = range_doy
        self.params = None

    @staticmethod
    def _reformat_arrays(x, y):
        x = np.array(x)
        if len(x.shape) < 2:
            x = x.reshape(x.shape[0], 1)
        y = np.array(y)
        if len(y.shape) < 2:
            y = y.reshape(y.shape[0], 1)
        return x, y

    def get_time_series_profile(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        threshold=None,
        resampling=None,
    ):
        """
        Get aggregated time series at the object level, i.e. aggregate all pixels not masked
        Parameters
        ----------
        eopatch (EOPatch) : EOPatch saved
        feature (str) : Name of the feature
        feature_mask (str) : Name of the boolean mask for pixels outside field boundaries
        function (np.function) : Function to aggregate object pixels

        Returns (np.array) : Aggregated 1-d time series
        -------

        """

        if feature not in eopatch[FeatureType.DATA].keys():
            raise ValueError("The feature name " + feature + " is not in the EOPatch")

        feature_array = eopatch[FeatureType.DATA][feature]

        if feature_mask not in eopatch[FeatureType.MASK_TIMELESS].keys():
            raise ValueError(
                "The feature {0} is missing in MASK_TIMELESS".format(feature_mask)
            )
        crop_mask = eopatch[FeatureType.MASK_TIMELESS][feature_mask]

        # Transform mask from 3D to 4D
        times, h, w, shape = feature_array.shape
        mask = crop_mask.reshape(1, h, w, 1)
        mask = [mask for _ in range(times)]
        mask = np.concatenate(mask, axis=0)
        #######################
        mask = [mask for _ in range(shape)]
        mask = np.concatenate(mask, axis=-1)
        ########################
        a = np.ma.array(feature_array, mask=np.invert(mask))
        ts_mean = np.ma.apply_over_axes(function, a, [1, 2])
        ts_mean = ts_mean.reshape(ts_mean.shape[0], ts_mean.shape[-1])

        # Thresholding time series values before and after the season
        doy, ids_filter = self.get_doy_period(eopatch)
        if self.range_doy is not None and threshold:
            ts_mean[: ids_filter[0]] = threshold
            ts_mean[ids_filter[-1] :] = threshold

        # Replace nas values with nearest valid value
        nans, x = self._nan_helper(ts_mean)
        ts_mean[nans] = np.interp(x(nans), x(~nans), ts_mean[~nans])

        if resampling:
            cs = Akima1DInterpolator(doy, ts_mean)
            doy = np.arange(1, 365, resampling)
            ts_mean = cs(doy)

        return doy, ts_mean

    def get_doy_period(self, eopatch):
        """
        Get day of the year acquisition dates from eopatch.timestamp
        Parameters
        ----------
        eopatch (EOPatch) : eopatch to read

        Returns (np.array : day of the years, np.array : index of image within the season (range_doy))
        -------

        """

        first_of_year = eopatch.timestamp[0].timetuple().tm_yday
        last_of_year = eopatch.timestamp[-1].timetuple().tm_yday

        times = np.asarray([time.toordinal() for time in eopatch.timestamp])
        times_ = (times - times[0]) / (times[-1] - times[0])
        times_doy = times_ * (last_of_year - first_of_year) + first_of_year

        if self.range_doy is None:
            return times_doy
        ids_filter = np.where(
            (times_doy > int(self.range_doy[0])) & (times_doy < int(self.range_doy[1]))
        )[0]

        return times_doy, ids_filter

    def _instance_inputs(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        threshold=0.2,
    ):
        """
        Initialize inputs to perform curve fitting
        Parameters
        ----------
        eopatch (EOPatch) : EOPatch which will be processed
        feature (str) : name of the feature to process
        feature_mask (str) : mask name in EOPatch to subset field pixels
        function (np.function) : function to aggregate pixels at object level

        Returns (np.array : day of the year within growing season ,
                np.array : day of the year min/max w.r.t growing season,
                np.array : aggregated time series witihin growing season)
        -------

        """

        times_doy, y = self.get_time_series_profile(
            eopatch, feature, feature_mask, function
        )

        x = (times_doy - self.range_doy[0]) / (self.range_doy[-1] - self.range_doy[0])
        ids = np.where(((x > 0) & (x < 1)))[0]

        times_doy_idx = list(times_doy[ids])
        x_idx = list(x[ids])
        y_idx = list(y.flatten()[ids])

        if times_doy_idx[0] > self.range_doy[0]:
            times_doy_idx.insert(0, self.range_doy[0])
            x_idx.insert(0, 0.0)
            y_idx.insert(0, threshold)

        if times_doy_idx[-1] < self.range_doy[-1]:
            times_doy_idx.append(self.range_doy[-1])
            x_idx.append(1.0)
            y_idx.append(threshold)

        x_idx = np.array(x_idx)
        times_doy_idx, y_idx = self._reformat_arrays(times_doy_idx, y_idx)

        return times_doy_idx, x_idx, y_idx

    def _init_execute(
        self, eopatch, feature, feature_mask, function, resampling, threshold=0.2
    ):
        times_doy, x, y = self._instance_inputs(
            eopatch, feature, feature_mask, function, threshold
        )

        if resampling > 0:
            times_doy = np.array(range(1, 365, resampling))
            new_x = (times_doy - self.range_doy[0]) / (
                self.range_doy[-1] - self.range_doy[0]
            )
            new_x[new_x < 0] = 0
            new_x[new_x > 1] = 1
        else:
            new_x = x

        return x, y.flatten(), times_doy, new_x

    def _replace_values(
        self, eopatch, ts_mean, doy_o, ts_fitted, doy_r, resampling, cloud_coverage=0.05
    ):
        # Replace interpolated values by original ones cloud free
        corrected_doy_d = doy_r.copy()
        corrected_doubly_res = ts_fitted.copy()

        for idx, i in enumerate(doy_o):
            if eopatch.scalar["COVERAGE"][idx][0] < cloud_coverage:
                previous_idx = np.where(doy_r <= i)[0][-1]
                next_idx = (
                    np.where(doy_r > i)[0][0] if i < np.max(doy_r) else previous_idx
                )
                argmin_ = np.argmin(
                    np.array(
                        [
                            i - corrected_doy_d[previous_idx],
                            corrected_doy_d[next_idx] - i,
                        ]
                    )
                )
                last_idx = previous_idx if argmin_ == 0 else next_idx
                corrected_doy_d[last_idx] = i
                corrected_doubly_res[last_idx] = ts_mean[idx]

        if resampling:
            cs = Akima1DInterpolator(corrected_doy_d, corrected_doubly_res)
            corrected_doy_d = np.arange(1, 365, resampling)
            corrected_doubly_res = cs(corrected_doy_d)

        nans, x = self._nan_helper(corrected_doubly_res)
        corrected_doubly_res[nans] = np.interp(
            x(nans), x(~nans), corrected_doubly_res[~nans]
        )

        return corrected_doy_d, corrected_doubly_res

    def _crop_values(self, ts_mean, min_threshold, max_threshold):
        ts_mean[ts_mean < float(min_threshold)] = min_threshold
        if max_threshold:
            ts_mean[ts_mean > float(max_threshold)] = max_threshold
        return ts_mean

    @staticmethod
    def _nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def _fit_whittaker(
        self, doy, ts_mean, degree_smoothing, threshold=0.2, weighted=True
    ):
        import modape
        from modape.whittaker import ws2d, ws2doptv, ws2doptvp

        # Apply whittaker smoothing
        w = np.array((ts_mean >= threshold) * 1, dtype="double")
        if weighted:
            w = w * ts_mean / np.max(ts_mean)
        smoothed_y = ws2d(ts_mean, degree_smoothing, w.flatten())
        doy, smoothed_y = self._reformat_arrays(doy, smoothed_y)
        return doy, smoothed_y

    def whittaker_smoothing(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        resampling=0,
        min_threshold=0.2,
        max_threshold=None,
        degree_smoothing=1,
        weighted=True,
    ):
        import modape
        from modape.whittaker import ws2d

        # Get time series profile
        doy, ts_mean_ = self.get_time_series_profile(
            eopatch,
            feature,
            feature_mask=feature_mask,
            function=function,
        )

        indices = np.setdiff1d(
            np.arange(len(doy)), np.unique(doy, return_index=True)[1]
        )
        doy = np.array([int(k) for i, k in enumerate(doy) if i not in indices])
        ts_mean_ = np.array([k for i, k in enumerate(ts_mean_) if i not in indices])

        nans, x = self._nan_helper(ts_mean_)
        ts_mean_[nans] = np.interp(x(nans), x(~nans), ts_mean_[~nans])

        ts_mean = self._crop_values(ts_mean_, min_threshold, max_threshold)

        # Interpolate the data to a regular time grid
        if resampling:
            try:
                cs = Akima1DInterpolator(doy, ts_mean)
                doy = np.arange(1, 365, resampling)
                ts_mean = cs(doy)

            except Exception as e:
                warnings.WarningMessage(
                    f"Cannot interpolate over new periods due to : {e}"
                )

        nans, x = self._nan_helper(ts_mean)
        ts_mean[nans] = np.interp(x(nans), x(~nans), ts_mean[~nans])
        # Apply Whittaker smoothing
        w = np.array((ts_mean > min_threshold) * 1, dtype="float")

        if weighted:
            w = w * ts_mean / np.max(ts_mean)

        if ts_mean.shape[1] > 1:
            ts_mean = np.array(
                [
                    ws2d(
                        ts_mean[..., i].flatten().astype("float"),
                        degree_smoothing,
                        w[..., i].flatten(),
                    )
                    for i in range(ts_mean.shape[1])
                ]
            )
            ts_mean = np.swapaxes(ts_mean, 0, 1)

        else:
            ts_mean = np.array(
                ws2d(ts_mean.flatten().astype("float"), degree_smoothing, w.flatten())
            )

        ts_mean = self._crop_values(ts_mean, min_threshold, max_threshold)
        doy, ts_mean = self._reformat_arrays(doy, ts_mean)

        return doy, ts_mean


class AsymmetricGaussian(CurveFitting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _asym_gaussian(x, initial_value, scale, a1, a2, a3, a4, a5):
        return initial_value + scale * np.piecewise(
            x,
            [x < a1, x >= a1],
            [
                lambda y: np.exp(-(((a1 - y) / a4) ** a5)),
                lambda y: np.exp(-(((y - a1) / a2) ** a3)),
            ],
        )

    def _fit_optimize_asym(self, x_axis, y_axis, initial_parameters=None):
        bounds_lower = [
            np.quantile(y_axis, 0.1),
            -np.inf,
            x_axis[0],
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
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
            initial_parameters = [
                np.mean(y_axis),
                0.2,
                x_axis[np.argmax(y_axis)],
                0.15,
                10,
                0.2,
                5,
            ]

        popt, pcov = curve_fit(
            self._asym_gaussian,
            x_axis,
            y_axis,
            initial_parameters,
            bounds=(bounds_lower, bounds_upper),
            maxfev=10e10,
            absolute_sigma=True,
            # method =  'lm'
        )
        self.params = popt

        return popt

    def fit_asym_gaussian(self, x, y):
        a = self._fit_optimize_asym(x, y)
        return self._asym_gaussian(x, *a)

    def get_fitted_values(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        resampling=0,
        threshold=0.2,
    ):
        """
        Apply Asymmetric function to do curve fitting from aggregated time series.
        It aims to reconstruct, smooth and extract phenological parameters from time series
        Parameters
        ----------
        eopatch (EOPatch)
        feature (str) : name of the feature to process
        feature_mask (str) : name of the polygon mask
        function (np.function) : function to aggregate pixels at object level

        resampling (np.array : dates from reconstructed time series, np.array : aggregated time series)

        Returns
        -------

        """

        x, y, times_doy, new_x = self._init_execute(
            eopatch, feature, feature_mask, function, resampling, threshold
        )
        y[y < threshold] = threshold

        initial_value, scale, a1, a2, a3, a4, a5 = self._fit_optimize_asym(x, y)
        fitted = self._asym_gaussian(new_x, initial_value, scale, a1, a2, a3, a4, a5)
        fitted[fitted < threshold] = threshold

        return times_doy, fitted

    def execute(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        resampling=0,
        threshold=0.2,
        degree_smoothing=0.5,
    ):
        # Get time series profile
        doy_o_s2, ts_mean_s2 = self.get_time_series_profile(
            eopatch, feature, feature_mask=feature_mask, function=function
        )
        # Apply curve fitting
        doy_d, doubly_res = self.get_fitted_values(
            eopatch,
            feature=feature,
            function=np.nanmean,
            resampling=resampling,
            threshold=threshold,
        )

        # Replace interpolated values by original ones
        corrected_doy_d, corrected_doubly_res = self._replace_values(
            eopatch, ts_mean_s2, doy_o_s2, doubly_res, doy_d, resampling
        )

        # Smooth interpolated with observed ones
        corrected_doy_d, smoothed_y = self._fit_whittaker(
            doy=corrected_doy_d,
            ts_mean=corrected_doubly_res,
            degree_smoothing=degree_smoothing,
            threshold=threshold,
            weighted=True,
        )

        return corrected_doy_d, smoothed_y


class DoublyLogistic(CurveFitting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _doubly_logistic(x, vb, ve, k, c, p, d, q):
        return (
            vb
            + (k / (1 + np.exp(-c * (x - p))))
            - ((k + vb + ve) / (1 + np.exp(d * (x - q))))
        )

    def _fit_optimize_doubly(self, x_axis, y_axis, initial_parameters=None):
        popt, _ = curve_fit(
            self._doubly_logistic,
            x_axis,
            y_axis,
            # method="lm",
            p0=initial_parameters,  # np.array([0.5, 6, 0.2, 150, 0.23, 240]),
            maxfev=int(10e6),
            bounds=[-np.Inf, np.Inf],
        )

        self.params = popt

        return popt

    def fit_logistic(self, x, y):
        a = self._fit_optimize_doubly(x, y)
        return self._doubly_logistic(x, *a)

    def get_fitted_values(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        resampling=0,
        threshold=0.2,
    ):
        """
        Apply Doubly Logistic function to do curve fitting from aggregated time series.
        It aims to reconstruct, smooth and extract phenological parameters from time series
        Parameters
        ----------
        eopatch (EOPatch)
        feature (str) : name of the feature to process
        feature_mask (str) : name of the polygon mask
        function (np.function) : function to aggregate pixels at object level

        resampling (np.array : dates from reconstructed time series, np.array : aggregated time series)

        Returns
        -------

        """

        x, y, times_doy, new_x = self._init_execute(
            eopatch, feature, feature_mask, function, resampling, threshold
        )
        y[y < threshold] = threshold

        vb, ve, k, c, p, d, q = self._fit_optimize_doubly(x, y)
        fitted = self._doubly_logistic(new_x, vb, ve, k, c, p, d, q)
        fitted[fitted < threshold] = threshold

        return times_doy, fitted

    def execute(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        resampling=0,
        threshold=0.2,
        degree_smoothing=0.5,
    ):
        # Get time series profile
        doy_o_s2, ts_mean_s2 = self.get_time_series_profile(
            eopatch, feature, feature_mask=feature_mask, function=function
        )

        doy_d, doubly_res = self.get_fitted_values(
            eopatch,
            feature=feature,
            function=np.nanmean,
            resampling=resampling,
            threshold=threshold,
        )

        corrected_doy_d, corrected_doubly_res = self._replace_values(
            eopatch, ts_mean_s2, doy_o_s2, doubly_res, doy_d, resampling
        )

        # Apply whittaker smoothing
        corrected_doy_d, smoothed_y = self._fit_whittaker(
            doy=corrected_doy_d,
            ts_mean=corrected_doubly_res,
            degree_smoothing=degree_smoothing,
            threshold=threshold,
            weighted=True,
        )

        return corrected_doy_d, smoothed_y


class FourrierDiscrete(CurveFitting):
    def __init__(self, omega=1.5, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega

    @staticmethod
    def _clean_data(x_axis, y_axis):
        good_vals = np.argwhere(~np.isnan(y_axis.flatten()))
        y_axis = np.take(y_axis, good_vals, 0)
        x_axis = np.take(x_axis, good_vals, 0)
        if x_axis[-1] > 1:
            x_axis = (x_axis - x_axis[0]) / (x_axis[-1] - x_axis[0])
        return x_axis, y_axis

    def _hants(self, x, c, omega, a1, a2, b1, b2):
        """
        Formula for a 2nd order Fourier series

        Parameters
        ----------
        x : Timepoint t on the x-axis
        c : Intercept coefficient
        omega : Omega coefficient (Wavelength)
        a1 : 1st order cosine coefficient
        a2 : 2nd order cosine coefficient
        b1 : 1st oder sine coefficient
        b2 : 2nd order sine coefficient

        Returns
        -------
        Parameters of the 2nd order Fourier series formula to be used in the "fouriersmoother" function
        """
        return (
            c
            + (a1 * np.cos(2 * np.pi * omega * 1 * x))
            + (b1 * np.sin(2 * np.pi * omega * 1 * x))
            + (a2 * np.cos(2 * np.pi * omega * 2 * x))
            + (b2 * np.sin(2 * np.pi * omega * 2 * x))
        )

    def _fourier(self, x, *a):
        ret = a[0] * np.cos(np.pi / self.omega * x)
        for deg in range(1, len(a)):
            ret += a[deg] * np.cos((deg + 1) * np.pi / self.omega * x)
        return ret

    def fit_fourier(self, x_axis, y_axis, new_x):
        '''
        Other version of fourier, but we should focus on hants instead
        '''
        x_axis, y_axis = self._clean_data(x_axis, y_axis)
        popt, _ = curve_fit(self._fourier, x_axis[:, 0], y_axis[:, 0], [1.0] * 5)
        self.params = popt
        return self._fourier(new_x, *popt)

    def fit_hants(self, x_axis, y_axis, new_x):
        '''
        Harmonic analysis of Time Series
        '''
        x_axis, y_axis = self._clean_data(x_axis, y_axis)

        popt, _ = curve_fit(
            self._hants,
            x_axis[:, 0],
            y_axis[:, 0],
            maxfev=5000,
            method="trf",
            p0=[np.mean(y_axis), 1, 1, 1, 1, 1],
        )
        self.params = popt
        return self._hants(new_x, *popt)

    def get_fitted_values(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        resampling=0,
        fft=True,
        threshold=0.2,
    ):
        """
        Apply Fourrier transform function to do curve fitting from aggregated time series.
        It aims to reconstruct, smooth and extract phenological parameters from time series
        Parameters
        ----------
        eopatch (EOPatch)
        feature (str) : name of the feature to process
        feature_mask (str) : name of the polygon mask
        function (np.function) : function to aggregate pixels at object level

        resampling (np.array : dates from reconstructed time series, np.array : aggregated time series)

        Returns
        -------
        """
        x, y, times_doy, new_x = self._init_execute(
            eopatch, feature, feature_mask, function, resampling, threshold
        )

        y[y < threshold] = threshold

        fitted = self.fit_hants(x, y, new_x) if fft else self.fit_fourier(x, y, new_x)
        fitted[fitted < threshold] = threshold

        return times_doy, fitted

    def execute(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        function=np.nanmedian,
        resampling=0,
        fft=True,
        threshold=0.2,
        degree_smoothing=0.5,
    ):
        # Get time series profile
        doy_o_s2, ts_mean_s2 = self.get_time_series_profile(
            eopatch, feature, feature_mask=feature_mask, function=function
        )

        # Apply curve fitting
        doy_d, doubly_res = self.get_fitted_values(
            eopatch,
            feature=feature,
            function=np.nanmean,
            resampling=resampling,
            fft=fft,
            threshold=threshold,
        )

        # Replace interpolated values by original ones
        corrected_doy_d, corrected_doubly_res = self._replace_values(
            eopatch=eopatch,
            ts_mean=ts_mean_s2,
            doy_o=doy_o_s2,
            ts_fitted=doubly_res,
            doy_r=doy_d,
            resampling=resampling,
        )

        # Apply whittaker smoothing
        corrected_doy_d, smoothed_y = self._fit_whittaker(
            doy=corrected_doy_d,
            ts_mean=corrected_doubly_res,
            degree_smoothing=degree_smoothing,
            threshold=threshold,
            weighted=True,
        )

        return corrected_doy_d, smoothed_y

