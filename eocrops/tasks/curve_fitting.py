import warnings

import numpy as np

from eolearn.core import FeatureType, EOTask
from scipy.optimize import curve_fit
from scipy.interpolate import *


class CurveFitting:
    def __init__(self, range_doy=(1, 365), q=0.5):
        self.range_doy = range_doy
        self.params = None
        self.q = q

    @staticmethod
    def _reformat_arrays(x, y):
        x = np.array(x)
        if len(x.shape) < 2:
            x = x.reshape(x.shape[0], 1)
        y = np.array(y)
        if len(y.shape) < 2:
            y = y.reshape(y.shape[0], 1)
        return x, y

    def _apply_quantile(self, x, axis):
        if self.q != 0.5:
            return np.nanquantile(np.copy(x), q=self.q, axis=axis)
        else:
            return np.nanmedian(x, axis=axis)

    def get_time_series_profile(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        threshold=None,
        resampling=None,
    ):
        """
        Get aggregated time series at the object level, i.e. aggregate all pixels not masked

        Parameters
        ----------
        eopatch : EOPatch
            EOPatch saved
        feature : str
            Name of the feature
        feature_mask : str
            Name of the boolean mask for pixels outside field boundaries

        Returns
        -------
        Aggregated 1-d time series (np.array)
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
        ts_mean = np.ma.apply_over_axes(self._apply_quantile, a, [1, 2])
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
        eopatch : EOPatch
            eopatch to read

        Returns
        -------
        np.array : day of the years, np.array : index of image within the season (range_doy)
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
        threshold=0.2,
    ):
        """
        Initialize inputs to perform curve fitting

        Parameters
        ----------
        eopatch : EOPatch
            EOPatch which will be processed
        feature : str
            name of the feature to process
        feature_mask : str
            mask name in EOPatch to subset field pixels
        function : np.function)
            function to aggregate pixels at object level

        Returns
        -------
        np.array : day of the year within growing season , np.array : day of the year min/max w.r.t growing season, np.array : aggregated time series witihin growing season)
        """

        times_doy, y = self.get_time_series_profile(eopatch, feature, feature_mask)

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

    def _init_execute(self, eopatch, feature, feature_mask, resampling, threshold=0):
        """
        Initialize the inputs prior applying any method of curve_fitting.
        It returns the day of th year from acquisition dates, the time series profile and the resampled day of the year given a resampling period

        Parameters
        ----------
        eopatch : EOPatch
            patch from a given fied
        feature : str
            name of the feature in FeatureType.DATA
        feature_mask : str
            name of the mask in FeatureType.MASK_TIMELESS
        function : np.function
            Function to aggregate pixels at the field level
        resampling : int
            length of the period for resampling (e.g. 8 day period)
        threshold : float
            threshold for values outside the season (below or above range_doy) which can affect the interpolation)

        Returns
        -------
        np.array : day of the year within growing season, np.array : aggregated time series np.array : resampled day of the year, np.array : resampled time series
        """
        # Get time series profile
        times_doy, x, y = self._instance_inputs(
            eopatch, feature, feature_mask, threshold
        )
        y[times_doy < self.range_doy[0]] = threshold
        y[times_doy > self.range_doy[1]] = threshold
        y[y < threshold] = threshold

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

    def replace_values(
        self, eopatch, ts_mean, doy_o, ts_fitted, doy_r, resampling, cloud_coverage=0.05
    ):
        """
        Given an interpolated time series using a curve fitting method,
        replace interpolated values with the nearest observed one iif the image had less than a given cloud coverage

        Parameters
        ----------
        eopatch : EOPatch

        ts_mean :np.array) : the original observed time series at the field-level
        doy_o : np.array)
            days of the year from the observed scene
        ts_fitted : np.array
            fitted time series using a cure fitting method
        doy_r : np.array
            resampled day of the year
        resampling : int
            period length to resample time series (e.g. 8 for 8-day period starting on the 1st January)
        cloud_coverage : float
            maximum percentage of cloud given an image (e.g. we keep only values if <5% cloudy)

        Returns
        -------
        Fitted time series mixing interpolated and observed values
        """

        # Replace interpolated values by original ones cloud free
        corrected_doy_d = doy_r.copy().flatten()
        corrected_doubly_res = ts_fitted.copy().flatten()

        for idx, i in enumerate(doy_o):
            if (
                self.range_doy[0] < i < self.range_doy[-1]
                and eopatch.scalar["COVERAGE"][idx][0] < cloud_coverage
            ):
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

    def _replace_values_iterate(self, patch, y_array, dates, new_y, new_doy):
        if len(y_array.shape) == 1:
            y_array = y_array.reshape(y_array.shape[0], 1)

        replace_y = np.empty((len(new_y), y_array.shape[1]))

        for i in range(y_array.shape[1]):
            _, replace_y[:, i] = self.replace_values(
                patch,
                y_array[:, i].flatten(),
                dates,
                new_y[:, i].flatten(),
                new_doy,
                resampling=8,
                cloud_coverage=0.05,
            )

        return replace_y

    def _crop_values(self, ts_mean, min_threshold, max_threshold):
        if max_threshold:
            ts_mean[ts_mean > float(max_threshold)] = max_threshold
        if min_threshold:
            ts_mean[ts_mean < float(min_threshold)] = min_threshold
        return ts_mean

    @staticmethod
    def _nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def resample_ts(
        self, doy, ts_mean_, resampling=8, min_threshold=None, max_threshold=None
    ):
        """
        Apply resampling over fixed period before applying whitakker smoothing
        """
        nans, x = self._nan_helper(ts_mean_)
        ts_mean_[nans] = np.interp(x(nans), x(~nans), ts_mean_[~nans])
        ts_mean = self._crop_values(ts_mean_, min_threshold, max_threshold)

        # Interpolate the data to a regular time grid
        if resampling:
            try:
                new_doy = np.arange(
                    self.range_doy[0], self.range_doy[-1] + 1, resampling
                )
                ts_mean_resampled = np.empty(
                    (ts_mean.shape[0], len(new_doy), ts_mean.shape[2])
                )
                for i in range(ts_mean.shape[0]):
                    for j in range(ts_mean.shape[2]):
                        cs = Akima1DInterpolator(doy, ts_mean[i, :, j])
                        ts_mean_resampled[i, :, j] = cs(new_doy)
                ts_mean = ts_mean_resampled

            except Exception as e:
                warnings.WarningMessage(
                    f"Cannot interpolate over new periods due to : {e}"
                )
        else:
            new_doy = doy

        return new_doy, ts_mean

    def fit_whitakker(self, ts_mean, degree_smoothing, weighted=False, min_threshold=0):
        """
        Fit whitakker smoothing given a ndarray (1,T,D)
        """
        import modape
        from modape.whittaker import ws2d, ws2doptv, ws2doptvp

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

        return ts_mean

    def whittaker_zonal(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        degree_smoothing=1,
        min_threshold=0,
        max_threshold=None,
        weighted=False,
        resampling=8,
    ):
        """
        Apply whitakker smoothing over time series

        Parameters
        ----------
        eopatch : EOPatch
            eopatch to apply the smoothing method
        feature : str
            Feature to apply the smoothing (e.g. NDVI)
        feature_mask : str
            Feature that refers to the pixel masking outside the field polygon
        degree_smoothing : float
            Degree of smoothing (0.5 works great)
        min_threshold : float
            Minimum threshold of the time series. If it is below, we mask as np.nan to avoid outlier values
        max_threshold : float
            Minimum threshold of the time sereis. If it is above, we mask as np.nan to avoid outlier values
        weighted : bool
            Weight the smoother w.r.t. the maximimum value to better fit the enveloppe over high value data
        resampling : int
            Resample over fixed periods from the 1st of January

        Returns
        -------
        np.array : day of the year, np.array : aggregated time series
        """
        # Check if we do not have duplicates

        doy, ts_data = self.get_time_series_profile(
            eopatch,
            feature,
            feature_mask,
        )

        indices = np.setdiff1d(
            np.arange(len(doy)), np.unique(doy, return_index=True)[1]
        )
        doy = np.array([int(k) for i, k in enumerate(doy) if i not in indices])
        ts_data = np.array([k for i, k in enumerate(ts_data) if i not in indices])
        ts_data = self._crop_values(ts_data, min_threshold, max_threshold)

        # Resampling
        doy_resampled, ts_data_resampled = self.resample_ts(
            doy=doy,
            ts_mean_=ts_data,
            resampling=resampling,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        ts_data_resampled[np.where(np.isnan(ts_data_resampled))[0]] = min_threshold
        # Replace interpolated values by original ones
        doy_resampled, ts_data_resampled = self.replace_values(
            eopatch, ts_data, doy, ts_data_resampled, doy_resampled, resampling
        )
        ts_data_resampled = ts_data_resampled.reshape(ts_data_resampled.shape[0], 1)

        ts_data_resampled = self._fit_whittaker(
            ts_data_resampled, degree_smoothing, weighted
        )

        return self._reformat_arrays(doy_resampled, ts_data_resampled)


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

    def _fit_asym_gaussian(self, x, y):
        a = self._fit_optimize_asym(x, y)
        return self._asym_gaussian(x, *a)

    def get_fitted_values(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        resampling=0,
        threshold=0,
    ):
        """
        Apply Asymmetric function to do curve fitting from aggregated time series.
        It aims to reconstruct, smooth and extract phenological parameters from time series

        Parameters
        ----------
        eopatch : EOPatch
        feature : str
            name of the feature to process
        feature_mask : str
            name of the polygon mask
        function : np.function
            function to aggregate pixels at object level
        resampling : np.array
            dates from reconstructed time series, np.array : aggregated time series)

        Returns
        -------
        np.array : doy, np.array : fitted values
        """

        x, y, times_doy, new_x = self._init_execute(
            eopatch, feature, feature_mask, resampling, threshold
        )

        initial_value, scale, a1, a2, a3, a4, a5 = self._fit_optimize_asym(x, y)
        fitted = self._asym_gaussian(new_x, initial_value, scale, a1, a2, a3, a4, a5)
        fitted[fitted < threshold] = threshold

        return times_doy, fitted

    def execute(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        resampling=0,
    ):
        """
        Execute A-G curve fitting function

        Parameters
        ----------
        eopatch : EOPatch
        feature : str
        feature_mask : str
        resampling : int

        Returns
        -------
        np.array : doy, np.array : fitted values
        """
        # Get time series profile
        doy_o_s2, ts_mean_s2 = self.get_time_series_profile(
            eopatch, feature, feature_mask=feature_mask
        )
        # Apply curve fitting
        doy_d, doubly_res = self.get_fitted_values(
            eopatch,
            feature=feature,
            resampling=resampling,
        )

        return doy_d, doubly_res


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

    def _fit_logistic(self, x, y):
        a = self._fit_optimize_doubly(x, y)
        return self._doubly_logistic(x, *a)

    def get_fitted_values(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        resampling=0,
        threshold=0.2,
    ):
        """
        Apply Doubly Logistic function to do curve fitting from aggregated time series.
        It aims to reconstruct, smooth and extract phenological parameters from time series

        Parameters
        ----------
        eopatch : EOPatch
        feature : str
            name of the feature to process
        feature_mask : str
            name of the polygon mask
        function : np.function
            function to aggregate pixels at object level
        resampling : np.array
            dates from reconstructed time series, np.array : aggregated time series)

        Returns
        -------
        np.array : doy, np.array : fitted values
        """

        x, y, times_doy, new_x = self._init_execute(
            eopatch, feature, feature_mask, resampling, threshold
        )

        vb, ve, k, c, p, d, q = self._fit_optimize_doubly(x, y)
        fitted = self._doubly_logistic(new_x, vb, ve, k, c, p, d, q)
        fitted[fitted < threshold] = threshold

        return times_doy, fitted

    def execute(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        resampling=0,
        threshold=0.2,
    ):
        """
        Execute the smoothing at the field level

        Parameters
        ----------
        eopatch : EOPatch
        feature : str
        feature_mask : str
        resampling : int
        threshold : float

        Returns
        -------
        np.array : doy, np.array : fitted values
        """
        # Get time series profile
        doy_o_s2, ts_mean_s2 = self.get_time_series_profile(
            eopatch,
            feature,
            feature_mask,
        )

        doy_d, doubly_res = self.get_fitted_values(
            eopatch,
            feature=feature,
            resampling=resampling,
            threshold=threshold,
        )

        return doy_d, doubly_res


class FourierDiscrete(CurveFitting):
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

    def _fit_fourier(self, x_axis, y_axis, new_x):
        x_axis, y_axis = self._clean_data(x_axis, y_axis)
        popt, _ = curve_fit(self._fourier, x_axis[:, 0], y_axis[:, 0], [1.0] * 5)
        self.params = popt
        return self._fourier(new_x, *popt)

    def _fit_hants(self, x_axis, y_axis, new_x):
        x_axis, y_axis = self._clean_data(x_axis, y_axis)

        popt, _ = curve_fit(
            self._hants,
            x_axis[:, 0],
            y_axis[:, 0],
            maxfev=1e5,
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
        resampling=0,
        fft=True,
        threshold=0.2,
    ):
        """
        Apply Fourier function to do curve fitting from aggregated time series.
        It aims to reconstruct, smooth and extract phenological parameters from time series

        Parameters
        ----------
        eopatch : EOPatch
        feature : str
            name of the feature to process
        feature_mask : str
            name of the polygon mask
        function : np.function
            function to aggregate pixels at object level

        resampling : np.array
            dates from reconstructed time series, np.array : aggregated time series)

        Returns
        -------
        np.array : doy, np.array : fitted values
        """
        x, y, times_doy, new_x = self._init_execute(
            eopatch, feature, feature_mask, resampling, threshold
        )

        fitted = self._fit_hants(x, y, new_x) if fft else self._fit_fourier(x, y, new_x)
        fitted[fitted < threshold] = threshold

        return times_doy, fitted

    def execute(
        self,
        eopatch,
        feature,
        feature_mask="polygon_mask",
        resampling=0,
        fft=True,
        threshold=0,
    ):
        """
        Execute FFT

        Parameters
        ----------
        eopatch : EOPatch
        feature : str
        feature_mask : str
        resampling : int
        fft : bool
        threshold : float

        Returns
        -------
        np.array : doy, np.array : fitted values
        """
        # Get time series profile
        doy_o_s2, ts_mean_s2 = self.get_time_series_profile(
            eopatch, feature, feature_mask=feature_mask
        )

        # Apply curve fitting
        doy_d, doubly_res = self.get_fitted_values(
            eopatch,
            feature=feature,
            resampling=resampling,
            fft=fft,
            threshold=threshold,
        )

        return doy_d, doubly_res
