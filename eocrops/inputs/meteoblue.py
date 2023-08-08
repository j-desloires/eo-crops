import aiohttp
import asyncio
from io import StringIO

import pandas as pd
import dateutil
import datetime as dt

import numpy as np
import datetime


class WeatherDownload:
    def __init__(
        self,
        api_key,
        shapefile,
        id_column,
        year_column,
        queryBackbone=None,
    ):
        """
        Define the request to download weather data from ERA5 using meteoblue api

        Parameters
        ----------
        api_key : str
                 Meteoblue API
        shapefile : gpd.GeoDataFrame
                    Shapefile with point or polygon geometry. Each observation is a request to obtain the associated meteorological data.
        id_column : str
                   Column from the shapefile containing the identifier of each observation. The output file will have this identifier also.
        year_column : str
                  Column from the shapefile containing the year (str) of each observation to download the weather data.
        queryBackbone : dict
               Dictionary containing the query backbone of meteoblue. By default, it is None as we defined it an example.

        """

        self.api_key = api_key

        if "geometry" not in shapefile.columns:
            raise ValueError("Must provide a shapefile with polygon data")

        shapefile["coordinates"] = [
            (geom.centroid.x, geom.centroid.y) for geom in shapefile["geometry"].values
        ]

        self.ids = shapefile[id_column].astype(str).values
        self.coordinates = shapefile["coordinates"].values
        self.years = shapefile[year_column].astype(str).values

        self.url_query = "http://my.meteoblue.com/dataset/query"
        self.url_queue = "http://queueresults.meteoblue.com/"

        if queryBackbone is None:
            queryBackbone = {
                "units": {
                    "temperature": "C",
                    "velocity": "km/h",
                    "length": "metric",
                    "energy": "watts",
                },
                "timeIntervalsAlignment": None,
                "runOnJobQueue": True,
                "oneTimeIntervalPerGeometry": True,
                "checkOnly": False,
                "requiresJobQueue": False,
                "geometry": {"type": "GeometryCollection", "geometries": None},
                "format": "csvIrregular",  # best format
                "timeIntervals": None,
            }
        self.queryBackbone = queryBackbone

    async def _get_jobIDs_from_query(self, query, time_interval=("01-01", "12-31")):
        """
        Get unique job id from a given query and a time interval
        """

        async def _make_ids(ids, coordinates, dates):
            for i, (id, coord, date) in enumerate(zip(ids, coordinates, dates)):
                yield i, id, coord, date

        jobIDs = []

        async for i, id, coord, date in _make_ids(
            self.ids, self.coordinates, self.years
        ):
            await asyncio.sleep(
                0.5
            )  # query spaced by 05 seconds => 2 queries max per queueTime (limit = 5)
            start_time, end_time = (
                f"{str(date)}-" + time_interval[0],
                f"{str(date)}-" + time_interval[1],
            )
            self.queryBackbone["geometry"]["geometries"] = [
                dict(type="MultiPoint", coordinates=[coord], locationNames=[id])
            ]

            self.queryBackbone["timeIntervals"] = [
                "{0}T+00:00/{1}T+00:00".format(start_time, end_time)
            ]
            self.queryBackbone["queries"] = query

            async with aiohttp.ClientSession() as session:
                # prepare the coroutines that post
                async with session.post(
                    self.url_query,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    params={"apikey": self.api_key},
                    json=self.queryBackbone,
                ) as response:
                    data = await response.json()
                    print(data)
                await session.close()
            jobIDs.append(data["id"])
        # now execute them all at once
        return jobIDs

    async def _get_request_from_jobID(self, jobID, sleep=1, limit=5):
        """
        Get data given a single jobID from the list of jobIDs _get_jobIDs_from_query
        """

        await asyncio.sleep(sleep)
        # limit amount of simultaneously opened connections you can pass limit parameter to connector
        conn = aiohttp.TCPConnector(limit=limit, ttl_dns_cache=300)
        session = aiohttp.ClientSession(
            connector=conn
        )  # ClientSession is the heart and the main entry point for all client API operations.
        # session contains a cookie storage and connection pool, thus cookies and connections are shared between HTTP requests sent by the same session.

        async with session.get(self.url_queue + jobID) as response:
            print("Status:", response.status)
            print("Content-type:", response.headers["content-type"])
            urlData = await response.text()
            print(response)
            await session.close()
        df = pd.read_csv(StringIO(urlData), sep=",", header=None)
        df["jobID"] = jobID
        return df

    @staticmethod
    async def _gather_with_concurrency(n, *tasks):
        semaphore = asyncio.Semaphore(n)

        async def sem_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*(sem_task(task) for task in tasks))

    @staticmethod
    def _format_data(df_stats):
        cols = df_stats.iloc[0, :]
        df_output = df_stats.iloc[1:, :]
        df_output.columns = cols
        return df_output

    def execute(self, query, time_interval=("01-01", "12-31"), conc_req=5):
        """
        Download Meteoblue data given a query and a time interval

        Parameters
        ----------
        query : dict
            Dictionary of agroclimatic variables
        time_interval : tuple
            Time interval to extract the data
        conc_req : int
            Maximum number of requests in concurrency

        Returns
        -------
        pd.DataFrame with weather data associated to each observation.
        """

        loop = asyncio.new_event_loop()
        try:
            jobIDs = loop.run_until_complete(
                self._get_jobIDs_from_query(query, time_interval)
            )
            time.sleep(2)
            dfs = loop.run_until_complete(
                self._gather_with_concurrency(
                    conc_req,
                    *[
                        self._get_request_from_jobID(jobID, i / 100)
                        for i, jobID in enumerate(jobIDs)
                    ],
                )
            )
        finally:
            print("close")
            loop.close()

        return self._format_data(pd.concat(dfs, axis=0))
