from sentinelhub import (
    DataCollection,
)

from eolearn.io import SentinelHubEvalscriptTask
from eocrops.utils import base_functions as utils
from eolearn.core import linearly_connect_tasks, EOWorkflow, FeatureType, OutputTask


class CollectionsSH:
    def __init__(self, config, resolution=10):
        """
        This method aims to download Copernicus products that are defined by SentinelHub collections freely available

        Parameters
        ----------
        config : sentinelhub.config
                 SentinelHub configuration
        resolution : int
            Spatial resolution of the downloaded products
        """
        self.config = config
        self.resolution = resolution

    def get_task_PPI(self):
        """Download Productivity and Phenology Index"""
        # https://collections.sentinel-hub.com/vegetation-phenology-and-productivity-parameters-season-1/
        byoc = DataCollection.define_byoc(
            collection_id="67c73156-095d-4f53-8a09-9ddf3848fbb6",
            name="PPI",
            service_url="https://creodias.sentinel-hub.com",
            is_timeless=False,
        )

        evalscript = """
                function setup() {
                    return {
                       input: ["SOSD", "EOSD", "LSLOPE", "RSLOPE", "MAXD",
                               "AMPL", "MAXV", "SPROD", "TPROD",
                               "QFLAG", "dataMask"],
                       output: [
                       {id: "SOSD", bands: 1, sampleType: SampleType.UINT16},
                       {id: "EOSD", bands: 1, sampleType: SampleType.UINT16},
                       {id: "LSLOPE", bands: 1, sampleType: SampleType.FLOAT32},
                       {id: "RSLOPE", bands: 1, sampleType: SampleType.FLOAT32},
                       {id: "MAXD", bands: 1, sampleType: SampleType.UINT16},
                       {id: "AMPL", bands: 1, sampleType: SampleType.UINT16},
                       {id: "MAXV", bands: 1, sampleType: SampleType.FLOAT32},
                       {id: "SPROD", bands: 1, sampleType: SampleType.UINT16},
                       {id: "TPROD", bands: 1, sampleType: SampleType.UINT16},
                       {id: "QFLAG", bands: 1, sampleType: SampleType.UINT16},              
                       {id:"dataMask", bands:1, sampleType: SampleType.UINT16 }]
                    }
                }

               function evaluatePixel(sample) {
                   return {
                       SOSD: [sample.SOSD],
                       EOSD: [sample.EOSD],
                       LSLOPE: [sample.LSLOPE],
                       RSLOPE: [sample.RSLOPE],
                       MAXD: [sample.MAXD],
                       AMPL: [sample.AMPL],
                       MAXV: [sample.MAXV],
                       SPROD: [sample.SPROD],
                       TPROD: [sample.TPROD],
                       QFLAG: [sample.QFLAG],
                       dataMask : [sample.dataMask]
                   }
               }
        """

        input_task = SentinelHubEvalscriptTask(
            features=[
                (FeatureType.DATA_TIMELESS, "SOSD", "SOSD"),
                (FeatureType.DATA_TIMELESS, "EOSD", "EOSD"),
                (FeatureType.DATA_TIMELESS, "RSLOPE", "RSLOPE"),
                (FeatureType.DATA_TIMELESS, "LSLOPE", "LSLOPE"),
                (FeatureType.DATA_TIMELESS, "MAXD", "MAXD"),
                (FeatureType.DATA_TIMELESS, "AMPL", "AMPL"),
                (FeatureType.DATA_TIMELESS, "MAXV", "MAXV"),
                (FeatureType.DATA_TIMELESS, "SPROD", "SPROD"),
                (FeatureType.DATA_TIMELESS, "TPROD", "TPROD"),
                (FeatureType.DATA_TIMELESS, "QFLAG", "QFLAG"),
                (FeatureType.DATA_TIMELESS, "dataMask", "dataMask"),
            ],
            data_collection=DataCollection.PPI,
            evalscript=evalscript,
            resolution=self.resolution,
            config=self.config,
        )

        return input_task

    def get_task_WC(self):
        """Download World Cover (10m) : only for 2020-2021"""
        # https://collections.sentinel-hub.com/worldcover/
        byoc = DataCollection.define_byoc(
            collection_id="0b940c63-45dd-4e6b-8019-c3660b81b884",
            name="WC",
            service_url=" http://services.sentinel-hub.com",
            is_timeless=False,
        )

        evalscript = """
        function setup() {
          return {
            input: ["Map", "dataMask"],
            output: [
                {id: "WC", bands: 1, sampleType: SampleType.FLOAT32},
                {id:"dataMask", bands:1, sampleType: SampleType.UINT8 }
            ]
          }
        }

        function evaluatePixel(sample) {
            return {
                WC: [sample.Map],
                dataMask : [sample.dataMask]
            }
        }
        """
        # https://collections.sentinel-hub.com/vegetation-phenology-and-productivity-parameters-season-1/
        input_task = SentinelHubEvalscriptTask(
            features=[(FeatureType.DATA, "WC"), (FeatureType.MASK, "dataMask")],
            data_collection=byoc,
            resolution=self.resolution,
            config=self.config,
            evalscript=evalscript,
        )
        return input_task

    def get_task_GLC(self):
        """Download Global Land Cover (100m)"""
        # 100 m Land Cover products for the years 2015 - 2019
        # https://collections.sentinel-hub.com/worldcover/
        byoc = DataCollection.define_byoc(
            collection_id="f0a97620-0e88-4c1f-a1ac-bb388fabdf2c",
            name="GLC",
            service_url="https://creodias.sentinel-hub.com",
            is_timeless=False,
        )

        evalscript = """
        function setup() {
          return {
            input: ["Discrete_Classification_map"],
            output: [
                {id: "GLC", bands: 1, sampleType: SampleType.FLOAT32},
                {id:"dataMask", bands:1, sampleType: SampleType.UINT8 }
            ]
          }
        }

        function evaluatePixel(sample) {
            return {
                GLC: [sample.Discrete_Classification_map],
                dataMask : [sample.dataMask]
            }
        }
        """
        # https://collections.sentinel-hub.com/vegetation-phenology-and-productivity-parameters-season-1/
        input_task = SentinelHubEvalscriptTask(
            features=[(FeatureType.DATA, "GLC"), (FeatureType.MASK, "dataMask")],
            data_collection=DataCollection.GLC,
            resolution=self.resolution,
            config=self.config,
            evalscript=evalscript,
        )
        return input_task

    def execute(self, input_task, shapefile, time_stamp):
        """
        For a given input task using get methods, get the pixel informations into EOPatch

        Parameters
        ----------
        input_task : SentinelHubEvalscriptTask
            Input task for the collection
        shapefile : gpd.GeoDataFrame
            Shapefile containing the field boundaries
        time_stamp : tuple
            Time periods, if it gathers two years, you would have two timestamps

        Returns
        -------
        EOPatch
        """
        field_bbox = get_bounding_box(shapefile)

        # With eolearn 1.0
        output_task = OutputTask("eopatch")
        workflow_nodes = linearly_connect_tasks(input_task, output_task)
        workflow = EOWorkflow(workflow_nodes)
        result = workflow.execute(
            {workflow_nodes[0]: {"bbox": field_bbox, "time_interval": time_stamp}}
        )

        return result.outputs["eopatch"]
