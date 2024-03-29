import pandas as pd
import dateutil
import datetime as dt

import numpy as np
import datetime


class CEHubFormatting:
    def __init__(
        self,
        shapefile,
        id_column,
        year_column,
        resample_range=("-01-01", "-12-31", 1),
        planting_date_column=None,
    ):
        """
        :param input_file (pd.DataFrame) : input file with fields in-situ data
        :param id_column (str): Name of the column that contains ids of the fields to merge with CEHub data
        :param planting_date_column (str): Name of the column with planting date in doy format
        :param havest_date_column (str): Name of the column with harvest date in doy format
        :param year_column (str) : Name of the column with the yearly season associated to each field
        :param resample_range (tuple): Interval of date to resample time series given a fixed period length (e.g. 8 days)
        """

        self.resample_range = resample_range
        self.id_column = id_column
        self.planting_date_column = planting_date_column

        self.year_column = year_column
        self.resample_range = resample_range
        self.input_file = shapefile.drop_duplicates(
            subset=[self.id_column, self.year_column]
        )

        if self.planting_date_column is not None:
            if self.planting_date_column not in list(shapefile.columns):
                raise ValueError(
                    "The column "
                    + self.planting_date_column
                    + " is not in the input file"
                )

            self.input_file = self.input_file.rename(
                columns={self.planting_date_column: "planting_date"}
            )

            self._apply_convert_doy("planting_date")

    def _get_descriptive_period(self, df, stat="mean"):
        """
        Compute descriptive statistics given period
        """
        dict_stats = dict(mean=np.nanmean, max=np.nanmax, min=np.nanmin, sum=np.nansum)

        df["value"] = df["value"].astype("float32")
        df_agg = (
            df[["variable", "period", "location", "value", self.year_column]]
            .groupby(["variable", "period", "location", self.year_column])
            .agg(dict_stats[stat])
        )
        df_agg.reset_index(inplace=True)
        df_agg = df_agg.rename(
            columns={"value": stat + "_value", "location": self.id_column}
        )
        return df_agg

    def _get_cumulated_period(self, df):
        """
        Compute the cumulative sum given period.
        """

        df_cum = pd.DataFrame()
        for var in df.variable.unique():
            df_subset = df[df.variable == var]
            df_agg = (
                df_subset[["location", "period", self.year_column, "variable", "value"]]
                .groupby(["location", self.year_column, "variable", "period"])
                .sum()
            )
            df_agg = df_agg.groupby(level=0).cumsum().reset_index()
            df_agg = df_agg.rename(
                columns={"value": "cumsum_value", "location": self.id_column}
            )
            df_cum = df_cum.append(df_agg)

        return df_cum

    def _get_resampled_periods(self, year="2021"):
        """
        Get the resampled periods from the resample range
        """
        resample_range_ = (
            str(year) + self.resample_range[0],
            str(year) + self.resample_range[1],
            self.resample_range[2],
        )

        start_date = dateutil.parser.parse(resample_range_[0])
        end_date = dateutil.parser.parse(resample_range_[1])
        step = dt.timedelta(days=resample_range_[2])

        days = [start_date]
        while days[-1] + step < end_date:
            days.append(days[-1] + step)
        return days

    def _format_periods(self, periods):
        df_resampled = pd.melt(periods, id_vars="period").rename(
            columns={"value": "timestamp", "variable": self.year_column}
        )

        # Left join periods to the original dataframe
        df_resampled["timestamp"] = [str(k) for k in df_resampled["timestamp"].values]
        df_resampled["timestamp"] = [
            np.datetime64(f"{str(year)}-" + "-".join(k.split("-")[1:]))
            for year, k in zip(
                df_resampled[self.year_column], df_resampled["timestamp"]
            )
        ]

        return df_resampled

    def _get_periods(self, df_cehub_):
        """
        Assign the periods to the file obtained through CEHub
        """

        def _get_year(x):
            return x[:4]

        def _convert_date(x):
            return dateutil.parser.parse(x[:-5])

        df_cehub = df_cehub_.copy()

        # Assign period ids w.r.t the date from the dataframe
        df_cehub["timestamp"] = [str(k) for k in df_cehub["timestamp"]]

        # Assign dates to a single year to retrieve periods
        df_cehub[self.year_column] = df_cehub["timestamp"].apply(lambda x: _get_year(x))
        df_cehub["timestamp"] = df_cehub["timestamp"].apply(lambda x: _convert_date(x))

        dict_year = {}
        for year in df_cehub[self.year_column].drop_duplicates().values:
            dict_year[year] = self._get_resampled_periods()

        periods = pd.DataFrame(dict_year)

        periods = periods.reset_index().rename(columns={"index": "period"})
        df_resampled = self._format_periods(periods)

        df = pd.merge(
            df_resampled, df_cehub, on=["timestamp", self.year_column], how="right"
        )

        fill_nas = (
            df[["period", "location"]]
            .groupby("location")
            .apply(
                lambda group: group.interpolate(
                    method="pad", limit=self.resample_range[-1]
                )
            )
        )

        df["period"] = fill_nas["period"]

        return df, df[["period", "timestamp"]].drop_duplicates()

    def _apply_convert_doy(self, feature):
        """
        Convert dates from CEhub format into day of the year
        """

        def _convert_doy_to_date(doy, year):
            date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(doy - 1)
            return np.datetime64(date)

        self.input_file[feature] = [
            _convert_doy_to_date(doy, year)
            for doy, year in zip(
                self.input_file[feature], self.input_file[self.year_column]
            )
        ]

    def _add_growing_stage(self, periods_df, feature="planting_date"):
        """
        Retrive the date from weather data associated with a given growing stage (doy format) from the input file
        The objective is to not take into account observations before sowing date of after harvest date in the statistics
        """

        return (
            pd.merge(
                periods_df,
                self.input_file[[feature, self.id_column]].copy(),
                left_on="timestamp",
                right_on=feature,
                how="right",
            )
            .rename(columns={"period": "period_" + feature})
            .drop(["timestamp"], axis=1)
        )

    def _init_df(self, df):
        """
        Initialize weather dataframe into periods to do the period calculations
        """
        df = df[~df.variable.isin(["variable"])]
        df = df.drop_duplicates(subset=["location", "timestamp", "variable"])

        df = df[df.location.isin(self.input_file[self.id_column].unique())]

        # Reformat into time series only if it is a dynamic variable
        df, periods_df = self._get_periods(df_cehub_=df.copy())
        df["value"] = df["value"].astype("float32")

        if self.planting_date_column is not None:
            periods_sowing = self._add_growing_stage(
                periods_df, feature="planting_date"
            )
            df = pd.merge(
                df[
                    [
                        "period",
                        "timestamp",
                        "location",
                        "variable",
                        "value",
                        self.year_column,
                    ]
                ],
                periods_sowing,
                left_on="location",
                right_on=self.id_column,
                how="left",
            )

            # Observations before planting date are assigned to np.nan
            df.loc[df.timestamp < df.planting_date, ["value"]] = np.nan

        return df

    def _prepare_output_file(self, df_stats, stat="mean"):
        """
        Prepare output dataframe with associated statistics over the periods.
        The output will have the name of the feature and its corresponding period (tuple)
        """
        df_pivot = pd.pivot_table(
            df_stats,
            values=[stat + "_value"],
            index=[self.id_column, self.year_column],
            columns=["variable", "period"],
            dropna=False,
        )

        df_pivot.reset_index(inplace=True)
        df_pivot.columns = [
            "-".join([str(x) for x in col]).strip() for col in df_pivot.columns.values
        ]
        df_pivot = df_pivot.rename(
            columns={
                self.id_column + "--": self.id_column,
                self.year_column + "--": self.year_column,
            }
        )
        df_pivot = df_pivot.sort_values(
            by=[self.id_column, self.year_column]
        ).reset_index(drop=True)
        return df_pivot

    def _get_temperature_difference(self, min_weather, max_weather):
        """
        Compute difference between minimum and maximum temperature observed for each period
        """
        diff_weather = min_weather.copy()

        tempMax = max_weather.loc[
            max_weather.variable.isin(["Temperature"]),
            ["period", "timestamp", self.id_column, "value"],
        ].rename(columns={"value": "value_max"})

        diff_weather = pd.merge(
            diff_weather,
            tempMax,
            on=["period", "timestamp", self.id_column],
            how="left",
        )

        diff_weather["value"] = diff_weather["value_max"] - diff_weather["value"]
        diff_weather["variable"] = "Temperature difference"

        return diff_weather

    def format_static_variable(self, df_weather, return_pivot=False):
        """
        Format static features from a given output of CEHub
        Parameters
        ----------
        df_weather (pd.DataFrame) : cehub dataframe with stat as daily descriptive statistics
        return_pivot (bool):

        Returns
        -------
        """
        df_weather = df_weather[~df_weather.variable.isin(["variable"])]
        df_weather = df_weather.drop_duplicates(
            subset=["location", "timestamp", "variable"]
        )
        df_weather = df_weather[
            df_weather.location.isin(self.input_file[self.id_column].unique())
        ]

        df_agg = (
            df_weather[["variable", "location", "value"]]
            .groupby(["variable", "location"])
            .agg("mean")
        )
        df_agg.reset_index(inplace=True)
        df_agg = df_agg.rename(
            columns={"value": "static_value", "location": self.id_column}
        )
        if return_pivot:
            df_agg = pd.pivot_table(
                df_agg,
                values=["static_value"],
                index=[self.id_column],
                columns=["variable"],
                dropna=False,
            )
            df_agg.reset_index(inplace=True)
            df_agg.columns = [
                "-".join([str(x) for x in col]).strip() for col in df_agg.columns.values
            ]

        return df_agg

    def execute(self, df_weather, stat=None, return_pivot=False):
        """
        Execute the workflow to get the dataframe aggregated into periods from CEHub data
        :param df_weather (pd.DataFrame) : cehub dataframe with stat as daily descriptive statistics
        :param return_pivot
        :return:
            pd.DataFrame with mean, min, max, sum aggregated into periods defined w.r.t the resample_range
        """
        # If resampling range in days >1, it means that we have periods. Therefore, we need to aggregate them
        if self.resample_range[-1] > 1 and stat not in [
            "mean",
            "min",
            "max",
            "sum",
            "cumsum",
        ]:
            raise ValueError(
                "Descriptive statistic must be 'mean', 'min', 'max', 'sum' or 'cumsum'"
            )

        init_weather = self._init_df(df=df_weather.copy())

        if stat == "cumsum":
            df_stats = self._get_cumulated_period(df=init_weather)
        else:
            df_stats = self._get_descriptive_period(df=init_weather, stat=stat)

        df_stats = df_stats.sort_values(
            by=[self.id_column, self.year_column, "variable"]
        )

        if not return_pivot:
            return df_stats

        output = self._prepare_output_file(df_stats=df_stats, stat=stat)
        output.columns = ["".join(k.split("value-")) for k in output.columns]
        output.columns = [
            tuple(k.split("-")) if k != self.id_column else k for k in output.columns
        ]
        output.columns = [
            (k[0], float(k[1])) if (type(k) is tuple and len(k) > 1) else k
            for k in output.columns
        ]

        return output
