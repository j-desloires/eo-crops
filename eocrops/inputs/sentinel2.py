import os
import multiprocessing

from eolearn.core import (
    SaveTask,
    linearly_connect_tasks,
    EOWorkflow,
    FeatureType,
    OutputTask,
)

from sentinelhub import DataCollection
from eolearn.io import SentinelHubDemTask, SentinelHubEvalscriptTask
from eolearn.core import OverwritePermission

import eocrops.tasks.preprocessing as preprocessing
from eocrops.utils import base_functions as utils
import eocrops.tasks.vegetation_indices as vegetation_indices
import eocrops.inputs.utils_sh as utils_sh


def workflow_instructions_S2L2A(
    config,
    time_stamp,
    coverage_predicate,
    path_out=None,
    polygon=None,
    interpolation=None,
    n_threads=multiprocessing.cpu_count() - 1,
):
    """
    Define the request of image from SentinelHub API using Sentinel-2 L2a product, available from 2017 with ~5 days revisit and 10-20 meters resolution

    Parameters
    ----------
    coverage_predicate: float
                        Upper bound of fraction of pixels contaminated by clouds
                        Images with higher cloud percentage will be removed
    time_stamp : tuple
                 first and last date to download the picture (e.g ('2017-01-01', '2017-12-31') for a 2017
    config : sentinelhub.config
             SentinelHub configuration
    path_out : str
               Path to save the EOPatch. It can be also a AWS S3 bucket path if s3Bucket is specified .
    polygon : gpd.GeoDataFrame
              Shapefile loaded using geopandas that contains the field(s) boundary(ies) of the area
    interpolation : dictionary
                    Specify if we interpolate cloudy pixels ({"interpolate": True}) and resample time series over fixed periods (e.g. 16 days : {"interpolate": True, , "period_length": 16})
    n_threads : int
                Number of threads used to download the EOPatch

    Returns
    -------
    EOPatch

    """

    if interpolation is None:
        interpolation = {"interpolate": False, "period_length": None}

    # Request format to download Landsat8 L2A products
    time_difference = datetime.timedelta(hours=2)

    # Request format to download L8 products
    evalscript = """
        function setup() {
            return {
                input: [{
                    bands: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12',
                    'viewZenithMean','sunZenithAngles','viewAzimuthMean', 'sunAzimuthAngles', 'CLM', "dataMask"],
                    units: ["REFLECTANCE",  "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE",  "REFLECTANCE", "REFLECTANCE",
                    "DEGREES", "DEGREES","DEGREES","DEGREES","DN","DN"]
                }],
                output: [
                    { id:"BANDS", bands:10, sampleType: SampleType.FLOAT32 },
                    { id:"ILLUMINATION", bands:4, sampleType: SampleType.FLOAT32 },
                    { id:"CLM", bands:1, sampleType: SampleType.UINT8 },
                    { id:"IS_DATA", bands:1, sampleType: SampleType.UINT8 }
                ]
            }
        }
        function evaluatePixel(sample) {
            return {
            BANDS: [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08,sample.B8A, 
            sample.B11, sample.B12], 
            ILLUMINATION : [sample.viewZenithMean,sample.sunZenithAngles,sample.viewAzimuthMean, sample.sunAzimuthAngles],
            CLM : [sample.CLM],
            IS_DATA :  [sample.dataMask]

            };
        }
    """

    input_task = SentinelHubEvalscriptTask(
        features=[
            (FeatureType.DATA, "BANDS", "BANDS-S2-L2A"),
            (FeatureType.DATA, "ILLUMINATION", "ILLUMINATION"),
            (FeatureType.MASK, "IS_DATA"),
            (FeatureType.MASK, "CLM"),
        ],
        data_collection=DataCollection.SENTINEL2_L2A,
        evalscript=evalscript,
        resolution=10,
        maxcc=coverage_predicate,
        time_difference=time_difference,
        config=config,
        max_threads=n_threads,
    )

    add_dem = SentinelHubDemTask("DEM", resolution=10, config=config)
    add_polygon_mask = preprocessing.PolygonMask(polygon)
    cloud_mask = utils_sh.ValidDataS2()

    add_coverage = utils_sh.AddValidDataCoverage()

    add_valid_mask = utils_sh.AddValidDataMaskTask(
        predicate=utils_sh.calculate_valid_data_mask
    )

    remove_cloudy_scenes = eolearn.features.SimpleFilterTask(
        (eolearn.core.FeatureType.MASK, "VALID_DATA"),
        utils_sh.ValidDataCoveragePredicate(coverage_predicate),
    )

    vis = vegetation_indices.VegetationIndicesS2(
        "BANDS-S2-L2A", mask_data=bool(1 - interpolation["interpolate"])
    )

    norm = vegetation_indices.EuclideanNorm("ECNorm", "BANDS-S2-L2A")

    if not interpolation["interpolate"]:
        linear_interp = utils_sh.EmptyTask()

    else:
        if "period_length" not in interpolation.keys():
            resampled_range = None
        elif interpolation["period_length"] is not None:
            resampled_range = (
                time_stamp[0],
                time_stamp[1],
                interpolation["period_length"],
            )

        copy_features = [
            (FeatureType.MASK, "CLM"),
            (FeatureType.DATA_TIMELESS, "DEM"),
            (FeatureType.MASK_TIMELESS, "MASK"),
        ]

        linear_interp = preprocessing.InterpolateFeatures(
            resampled_range=resampled_range, copy_features=copy_features
        )

    if path_out is None:
        save = utils_sh.EmptyTask()

    else:
        os.makedirs(path_out, exist_ok=True)
        save = SaveTask(
            path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH
        )

    output_task = OutputTask("eopatch")
    workflow_nodes = linearly_connect_tasks(
        input_task,
        cloud_mask,
        add_valid_mask,
        add_coverage,
        remove_cloudy_scenes,
        add_dem,
        add_polygon_mask,
        vis,
        norm,
        linear_interp,
        save,
        output_task,
    )
    workflow = EOWorkflow(workflow_nodes)

    field_bbox = utils.get_bounding_box(polygon)
    result = workflow.execute(
        {workflow_nodes[0]: {"bbox": field_bbox, "time_interval": time_stamp}}
    )

    return result.outputs["eopatch"]
