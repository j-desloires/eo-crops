import os
import contextlib

import dateutil
import datetime as dt
import numpy as np
from ast import literal_eval
import pandas as pd


class TempResampling:
    def __init__(
        self,
        range_dates=("2017-01-01", "2017-12-31"),
        stop=("2017-12-31"),
        smooth=False,
        id_column="key",
        varname_gdd="sum_Growing Degree days daily max/min",
        period_nas_rate=0.8,
        drop_nas_rate=0.2,
        bands=None,
        subset_id_fields=None,
    ):
        """
        Resample time series (e.g. satellite image time series and daily weather data) over accumulated GDU periods (thermal time).

        Parameters
        ----------
        range_dates : str
            Range of dates from the time series data. For satellite data, the data should be already be resampled using fixed periods from the start_date (e.g. 16- day periods from 1st January)
        stop : tuple
            Stoping date for temporal resampling. Very convenient if we train an in-season model to keep information only prior this date.
        smooth : bool
            Apply smoothing over time series when we resample into daily data in the pipeline
        id_column : str
            Column from the weather data file which refers to the identifier of the observation.
        varname_gdd : str, optional
            Name of the column for the weather dataset that refers to the accumulated GDUs
        period_nas_rate : float
            Keep a period only if it is completed at x %
        drop_nas_rate : float
            range of dates from the time series, by default yearly : ("01-01", "12-31")
        bands : tuple, optional
            range of dates from the time series, by default yearly : ("01-01", "12-31")
        subset_id_fields : tuple, optional
            range of dates from the time series, by default yearly : ("01-01", "12-31")
        """

        if bands is None:
            bands = [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
            ]

        self.id_column = id_column
        self.bands = bands
        self.range_dates = range_dates
        self.stop = stop
        if self.stop is None:
            self.stop = self.range_dates[-1]
        self.varname_gdd = varname_gdd
        self.period_nas_rate = period_nas_rate
        self.drop_nas_rate = drop_nas_rate
        self.smooth = smooth
        self.subset_id_fields = subset_id_fields
        self.sat_load = False
        self.weather_load = False
        self.meta_load = False

    def get_weather_feature(self, fname):
        """
        Get subset of column from Meteoblue data corresponding to a given feature
        """
        fnames_columns = [k[0] for k in list(self.weather_data.columns)]
        weather = [k == fname for k in fnames_columns]
        weather = self.weather_data.columns.values[weather]

        weather_df = self.weather_data[weather].copy()
        weather_df_daily = weather_df.T
        weather_df_daily.columns = self.weather_data[self.id_column]
        dates_weather = self._get_resampled_periods(1)

        if len(dates_weather) != weather_df_daily.index.shape[0]:
            start, end = int(weather_df_daily.index[0][-1]), int(
                weather_df_daily.index[-1][-1]
            )
            weather_df_daily.index = dates_weather[start : (end + 1)]
        else:
            weather_df_daily.index = dates_weather

        return weather_df_daily

    def _get_resampled_periods(self, days_range):
        """
        Get the resampled periods from the resample range of satellite data
        """
        resample_range_ = (
            self.range_dates[0],
            self.range_dates[1],
            days_range,
        )

        start_date = dateutil.parser.parse(resample_range_[0])
        end_date = dateutil.parser.parse(resample_range_[1])
        step = dt.timedelta(days=resample_range_[2])

        days = [start_date]
        while days[-1] + step < end_date:
            days.append(days[-1] + step)

        return days

    def _preprocess_meta_data(self, yield_data, feature_vector):
        """
        Retrieve ground truth data from Sentinel-2 data loader
        :param yield_data (np.array) : metadata from data loader
        :param feature_vector (list of tuples) : name of the features from ground truth
        """
        df = pd.DataFrame(yield_data)
        df.columns = [f[0] for f in feature_vector]

        # Sometimes, numpy encode string ==> need to decode
        try:
            df[self.id_column] = df[self.id_column].apply(
                lambda x: x.decode("utf-8").replace("/", "\\").split("\\")[-1]
            )
        except:
            df[self.id_column] = df[self.id_column].astype(str)

        if self.subset_id_fields is not None:
            with contextlib.suppress(UnicodeDecodeError, AttributeError):
                self.subset_id_fields = [
                    k.decode("utf-8") for k in self.subset_id_fields
                ]
            self.subset_id_fields = [
                k.replace("/", "\\").split("\\")[-1] for k in self.subset_id_fields
            ]
            df = df[df[self.id_column].isin(self.subset_id_fields)]

        df = df.sort_values(by=self.id_column)

        return df

    def load_meta_data(self, feature_vector, filepath):
        """Load metadata file that contains information of the fields"""
        if not os.path.exists(filepath):
            raise ValueError("The path {0} does not exist".format(filepath))

        yield_data = np.load(
            filepath,
            allow_pickle=True,
        )
        if len(yield_data.shape) > 2:
            yield_data = np.moveaxis(yield_data, 2, 1).squeeze()
        self.meta = self._preprocess_meta_data(yield_data, feature_vector)
        self.meta = self.meta.sort_values(by=self.id_column)
        self.meta_load = True

        return self.meta

    def load_sat_data(
        self,
        filepath,
    ):
        """
        Load 3D arrays thar contains satellite data
        """
        if not os.path.exists(filepath):
            raise ValueError("The path {0} does not exist".format(filepath))

        if self.meta_load == False:
            raise ValueError(
                "You must first load the metadata using the method load_meta"
            )

        X = np.load(filepath)
        X = np.where(np.isnan(X), np.ma.array(X, mask=np.isnan(X)).mean(axis=0), X)

        if self.id_column not in list(self.meta.columns):
            raise ValueError(
                "The column {0} is not in the meta_file".format(self.id_column)
            )

        self.X = X[
            self.meta.index,
        ]

        self.sat_load = True

        return self.X

    def load_weather_data(self, filepath):
        """
        Load weather data reformated and saved from csv file (filepath). It corresponds to the output of from (1) Meteoblue_client (2) format_data (predfin.data_extraction.ce_hub)
        See the example from workflow_yield_prediction to understand how to build those files

        Parameters
        ----------
        filepath :str
            path where the csv file is saved

        Returns
        -------
        weather_data merged with meta_data
        """
        if not os.path.exists(filepath):
            raise ValueError("The filename {0} does not exist".format(filepath))
        self.weather_data = pd.read_csv(filepath)
        weather = [k for k in list(self.weather_data.columns) if "(" in k]
        weather_renamed = [literal_eval(k) for k in weather]
        dict_rename = dict(zip(weather, weather_renamed))
        self.weather_data = self.weather_data.rename(columns=dict_rename)

        if self.id_column not in list(self.weather_data.columns):
            raise ValueError("The column {0} does not exist".format(self.id_column))

        self.weather_data[self.id_column] = self.weather_data[self.id_column].apply(
            lambda x: str(x).replace("/", "\\").split("\\")[-1]
        )
        self.weather_data = self.weather_data.sort_values(by=self.id_column)

        names_columns = {k[0] for k in list(self.weather_data.columns)}

        if self.varname_gdd not in names_columns:
            raise ValueError(
                "The column {0} is not in the weather data file".format(
                    self.varname_gdd
                )
            )

        self.weather_load = True
        # TODO : objet independant weather_data @dataclass ==> specfique à chaque fonction => vérifier si None au lieu de self.check
        # TODO Séparer => file parsing : séparer dans sous class

        if self.meta_load == False:
            raise ValueError(
                "You must first load the metadata using the method load_meta"
            )

        self.weather_data = self.weather_data[
            self.weather_data[self.id_column].isin(self.meta[self.id_column].unique())
        ]
        if self.weather_data.shape[0] == 0:
            raise ValueError(
                "Any observations are matching btw weather and S2 w.r.t the key column {0}".format(
                    self.id_column
                )
            )
        return self.weather_data

    def _get_cumsum(self, merged_df, fname):
        """
        Compute cumulated sum over GDU intervals
        """
        merged_df[fname] = merged_df.groupby([self.id_column]).cumsum()[fname]
        return merged_df

    def _get_gdd(self):
        """
        Compute cumulated sum GDD
        """
        gdd_df_daily = self.get_weather_feature(fname=self.varname_gdd)
        return gdd_df_daily.cumsum(axis=0)

    def get_sat_features(self, feature_data, fname):
        """
        Retrieve time series vegetation index from 3D array given a feature name
        """
        i = 0
        ls_feat = []
        for _, _, fname_, _, num in feature_data:
            ls_feat.append(fname_)
            if fname_ == fname:
                break
            i += num
        if fname not in ls_feat:
            raise ValueError("The name of the feature does not exist in the dataset")
        else:
            return pd.DataFrame(self.X[..., i])

    def _get_s2_band(self, fname):
        """
        Retrieve time series band from 3D array given a feature name
        """

        if fname not in self.bands:
            raise ValueError(f"Band name must be included in {str(self.bands)}")
        band_id = self.bands.index(fname)
        return pd.DataFrame(self.X[..., band_id])

    @staticmethod
    def _remove_outliers(ts_, qmin=0.05, qmax=0.98):
        """
        Remove outliers from time series, i.e. mask acquisition date if values outside quantiles
        """
        ts = ts_.copy()
        ts_[ts_ < 0] = 0
        quant_min = np.nanquantile(ts, qmin)
        quant_max = np.nanquantile(ts, qmax)
        ts[ts < quant_min] = np.nan
        ts[ts > quant_max] = np.nan
        return pd.DataFrame(ts).interpolate(method="cubicspline").values.flatten()

    def _resample_daily_satellite(
        self, original_data, id_fields, days_range=8, remove_outliers=False
    ):
        """
        Interpolate Sentinel-2 data into daily data to merge with GDU
        """
        dates_ndvi = self._get_resampled_periods(days_range)
        original_data = original_data.T
        original_data.columns = id_fields
        original_data.index = dates_ndvi

        if remove_outliers:
            for i in range(original_data.shape[1]):
                original_data.iloc[:, i] = self._remove_outliers(
                    ts_=original_data.iloc[:, i].copy()
                )

        resample = original_data.resample("D").interpolate(method="cubicspline")

        def _func_svg(x):
            from scipy.signal import savgol_filter

            return savgol_filter(x, window_length=31, deriv=0, polyorder=3)

        resample_ = resample.copy()
        if self.smooth:
            resample_ = resample_.apply(_func_svg, axis=0)

        return resample_

    def _merge_daily_features(self, df_daily, gdd_df_daily, fname):
        """
        Merge daily data over GDU to prepare from GDU intervals resampling
        """
        df_daily = df_daily.T
        df_daily.reset_index(inplace=True)

        gdd_df_daily = gdd_df_daily.T
        gdd_df_daily.reset_index(inplace=True)

        df_daily = pd.melt(df_daily, id_vars=[self.id_column])
        df_daily.columns = [self.id_column, "variable", fname]
        df_daily = df_daily[~np.isnan(df_daily[fname])]

        gdd_df_daily = pd.melt(gdd_df_daily, id_vars=[self.id_column])
        gdd_df_daily.columns = [self.id_column, "variable", "gdd"]
        gdd_df_daily = gdd_df_daily[gdd_df_daily.gdd > 0]

        merged_df = pd.merge(
            df_daily, gdd_df_daily, on=[self.id_column, "variable"], how="right"
        )

        stop_date = np.datetime64(self.stop + "T00:00:00.000000000")
        merged_df = merged_df[merged_df["variable"] < stop_date]

        return merged_df

    def _check_status(self):
        if self.sat_load is False:
            raise ValueError(
                "You must first load the satellite data using the load_S2_data() method."
            )
        if self.weather_load is False:
            raise ValueError(
                "You must first load the weather data using the load_weather_data() method."
            )

    def _check_stat(self, stat):
        if stat in ["up", "down"]:
            stat = "sum"
        elif stat not in [
            "mean",
            "min",
            "max",
            "sum",
        ]:
            raise ValueError(
                "Descriptive statistic must be 'mean', 'min', 'max', 'sum'"
            )
        return stat

    def get_gdd_value_peak(
        self,
        features_data,
        fname="Cab",
        ub_gdd=900,
        lb_gdd=400,
        days_range=8,
    ):
        """
        Get GDD and vegetation index values when the daily time series of the vegetation index is maximum
        """
        self._check_status()

        gdd_df_daily = self._get_gdd()

        if fname in self.bands:
            s2_data = self._get_s2_band(fname)
        else:
            s2_data = self.get_sat_features(features_data, fname)

        s2_data_daily = self._resample_daily_satellite(
            original_data=s2_data,
            id_fields=gdd_df_daily.columns,
            days_range=days_range,
            remove_outliers=False,
        )

        merged_df = self._merge_daily_features(s2_data_daily, gdd_df_daily, fname)
        merged_df = merged_df[merged_df["gdd"] < ub_gdd]
        merged_df = merged_df[merged_df["gdd"] > lb_gdd]

        max_values = (
            merged_df[[self.id_column, fname]]
            .groupby(self.id_column)
            .agg("max")
            .reset_index()
        )

        return pd.merge(
            max_values,
            merged_df[[self.id_column, fname, "gdd"]],
            on=[self.id_column, fname],
            how="left",
        )

    def get_last_gdd(self):
        """Get last accumulated GDD observed for each observation"""
        self._check_status()

        gdd_df_daily = self._get_gdd()
        gdd_df_daily = gdd_df_daily.T
        gdd_df_daily.reset_index(inplace=True)

        gdd_df_daily = pd.melt(gdd_df_daily, id_vars=[self.id_column])
        gdd_df_daily.columns = [self.id_column, "variable", "gdd"]
        gdd_df_daily = gdd_df_daily[gdd_df_daily.gdd > 0]

        stop_date = np.datetime64("2021-" + self.stop + "T00:00:00.000000000")
        gdd_df_daily = gdd_df_daily[gdd_df_daily["variable"] < stop_date]

        return (
            gdd_df_daily[[self.id_column, "gdd"]]
            .groupby(self.id_column)
            .agg("max")
            .reset_index()
        )

    def _remove_incomplete_periods(self, merged_df, period_name, increment):
        # Count number of days per period
        threshold_inc = int(self.period_nas_rate * increment)
        max_intervals = (
            merged_df[[self.id_column, "intervals", period_name]]
            .dropna()
            .groupby([self.id_column, "intervals"])
            .agg("max")
            .reset_index()
        )

        max_intervals[period_name] = [
            increment - (increment * (x + 1) - y)
            for x, y in zip(
                max_intervals["intervals"].values, max_intervals[period_name].values
            )
        ]

        # Get last period available per field
        last_period = (
            max_intervals[[self.id_column, "intervals"]]
            .groupby(self.id_column)
            .agg("max")
            .reset_index()
        )
        # Get number of days for the last period of each fields
        last_period = pd.merge(
            last_period, max_intervals, on=[self.id_column, "intervals"], how="left"
        )

        ## Set threshold to remove periods
        last_period["tolerance"] = threshold_inc
        ## Keep period if only it has more than # days per period
        filter_obs = last_period[last_period[period_name] < last_period["tolerance"]]

        filter_obs["field_period"] = [
            f"{x}-{y}"
            for x, y in zip(filter_obs[self.id_column], filter_obs["intervals"])
        ]

        merged_df["field_period"] = [
            f"{x}-{y}"
            for x, y in zip(merged_df[self.id_column], merged_df["intervals"])
        ]

        max_period = (
            merged_df[[self.id_column, "intervals"]]
            .groupby(self.id_column)
            .agg("max")
            .reset_index()
        )

        merged_df = merged_df[~merged_df.field_period.isin(filter_obs["field_period"])]
        merged_df = merged_df.drop(["field_period"], axis=1)

        return merged_df, max_period

    def _build_period(self, merged_df, fname, increment, n_periods, period_name, stat):
        """Build periods by incrementing a fix GDD unit (e.g every 100 GDD)"""
        bins_ = [i * increment for i in range(n_periods)]

        intervals, bins = pd.cut(
            x=merged_df[period_name], bins=bins_, right=True, labels=False, retbins=True
        )
        merged_df["intervals"] = intervals

        merged_df, max_period = self._remove_incomplete_periods(
            merged_df,
            period_name=period_name,
            increment=increment,
        )
        #########################################################################
        pivot_merged_df = pd.pivot_table(
            merged_df,
            values=[fname],
            index=[self.id_column],
            aggfunc=stat,
            columns=["intervals"],
        ).reset_index()

        pivot_merged_df.columns.values[0] = self.id_column
        pivot_merged_df = pivot_merged_df.dropna(
            thresh=int(pivot_merged_df.shape[0] * self.drop_nas_rate), axis=1
        )
        return pivot_merged_df, max_period

    def _resample_feature_over_gdd(
        self, df_daily, gdd_df_daily, fname, stat="mean", cumsum=False, increment=75
    ):
        """
        Resample time series data into GDU interval given a number of periods and a statistic (e.g. mean)
        """
        stat = self._check_stat(stat)
        merged_df = self._merge_daily_features(df_daily, gdd_df_daily, fname)

        stop = np.max(merged_df["gdd"].values)
        n_periods = round(stop / increment) + 1

        if cumsum:
            merged_df = self._get_cumsum(merged_df, fname)

        pivot_merged_df, max_period = self._build_period(
            merged_df, fname, increment, n_periods, "gdd", stat
        )

        return pivot_merged_df, max_period

    def _resample_feature_over_time(
        self,
        df_daily,
        gdd_df_daily,
        fname,
        stat="mean",
        cumsum=False,
        increment=10,
    ):
        """
        Resample time series data into GDU interval given a number of periods and a statistic (e.g. mean)
        """
        stat = self._check_stat(stat)
        doy_cummulated = gdd_df_daily.copy()
        doy_cummulated[doy_cummulated > 0] = 1
        doy_cummulated = doy_cummulated.cumsum(axis=0)
        merged_df = self._merge_daily_features(df_daily, doy_cummulated, fname)
        merged_df = merged_df.rename(columns={"gdd": "doy"})
        stop = np.max(merged_df["doy"].values)
        n_periods = round(stop // increment) + 1

        list_df = []
        for keys in merged_df[self.id_column].unique():
            subset = merged_df[merged_df[self.id_column].isin([keys])]
            subset["doy"] = list(range(subset.shape[0]))
            list_df.append(subset)

        merged_df = pd.concat(list_df, axis=0)

        if cumsum:
            merged_df = self._get_cumsum(merged_df, fname)

        pivot_merged_df, max_period = self._build_period(
            merged_df, fname, increment, n_periods, "doy", stat
        )

        return pivot_merged_df, max_period

    def fill_missing_columns(self, output, na_value=None):
        """Fill nas values since some fields do not match with GDU intervals (especially for high intervals)"""
        row, cols = list(np.where(output.isnull()))
        cols = sorted(set(list(cols)))
        if na_value:
            output.fillna(na_value, inplace=True)
        else:
            for col in cols:
                output.iloc[:, col].fillna(output.iloc[:, col - 1], inplace=True)

        output = output.droplevel(axis=1, level=0)
        output.columns.values[0] = self.id_column

        return output

    @staticmethod
    def _reformat_columns(df):
        cols_ = [(int(k[0]), k[1]) for k in df.columns]
        df.columns = cols_
        cols_.sort()
        df = df[cols_]
        return df

    def resample_s2(
        self,
        features_data,
        fname,
        stat="mean",
        increment=120,
        thermal_time=True,
        period_length=8,
        cumsum=False,
        remove_outliers=True,
    ):
        """
        Resample satellite data over periods (thermal or calendar) from the planting date

        Parameters
        ----------
        features_data : list
            list of S2 features (S2 data loader)
        fname : str
            name of the S2 feature to resample over GDU intervals
        stat : str
            aggregation function over periods
        increment : int
            number of units between each period (accumulated gdd or days)
        thermal_time : bool
            Resample over thermal time or calendar
        days_range : int
            S2 resampling resolution (8 days by default) from the original file in load_sat_data
        cumsum : bool
            compute cumulated cum of the feature rather than the mean over GDU intervals
        remove_outliers : bool
            remove outliers from time series using quantiles (0.02) ==> cloud

        Returns
        -------
        pd.DataFrame : dataset with vegetation indices resampled
        """
        self._check_status()

        gdd_df_daily = self._get_gdd()

        if fname in self.bands:
            s2_data = self._get_s2_band(fname)
        else:
            s2_data = self.get_sat_features(features_data, fname)

        s2_data_daily = self._resample_daily_satellite(
            original_data=s2_data,
            id_fields=gdd_df_daily.columns,
            days_range=period_length,
            remove_outliers=remove_outliers,
        )

        dict_parameters = dict(
            df_daily=s2_data_daily,
            gdd_df_daily=gdd_df_daily,
            fname=fname,
            stat=stat,
            cumsum=cumsum,
            increment=increment,
        )

        if thermal_time:
            output, max_period = self._resample_feature_over_gdd(**dict_parameters)
        else:
            output, max_period = self._resample_feature_over_time(**dict_parameters)

        return output, max_period

    def resample_weather(
        self,
        fname,
        stat,
        increment=120,
        thermal_time=True,
    ):
        """
        Resample weather data over periods (thermal or calendar) from the planting date

        Parameters
        ----------

        fname : str
            name of the S2 feature to resample over GDU intervals
        stat : str
            Aggregation over periods (e.g. mean)
        increment : int
            Number of units between each period (accumulated gdd or days)
        thermal_time : bool
            Resample over thermal time or calendar

        Returns
        -------
        pd.DataFrame : dataset with weather data resampled
        """

        if self.weather_load is False:
            raise ValueError(
                "You must first load the weather data using the load_weather_data() method."
            )

        gdd_df_daily = self._get_gdd()
        weather_df_daily = self.get_weather_feature(fname)

        dict_parameters = dict(
            df_daily=weather_df_daily,
            gdd_df_daily=gdd_df_daily,
            fname=fname,
            stat=stat,
            increment=increment,
        )

        output, _ = (
            self._resample_feature_over_gdd(**dict_parameters)
            if thermal_time
            else self._resample_feature_over_time(**dict_parameters)
        )

        return output
