import os
import datetime
import eolearn

from sentinelhub import DataCollection

from eolearn.core import OverwritePermission
from eolearn.io import SentinelHubInputTask, SentinelHubDemTask

import eocrops.tasks.preprocessing as preprocessing
import eocrops.utils.base_functions as utils
import multiprocessing

import eocrops.inputs.utils_sh as utils_sh

from eolearn.core import (
    linearly_connect_tasks,
    SaveTask,
    EOWorkflow,
    FeatureType,
    OutputTask,
)


def workflow_instructions_S1IW(
    config,
    time_stamp,
    path_out=None,
    polygon=None,
    backCoeff="GAMMA0_TERRAIN",
    orbit_direction="ASC",
    speckle_lee_window=3,
    n_threads=multiprocessing.cpu_count() - 1,
):
    """
    Define the request of image from SentinelHub API using Sentinel-1 IW-GRD product

    Parameters
    ----------
    config : sentinelhub.config
             SentinelHub configuration
    time_stamp : tuple
                 first and last date to download the picture (e.g ('2017-01-01', '2017-12-31') for a 2017
    path_out : str
               Path to save the EOPatch. It can be also a AWS S3 bucket path if s3Bucket is specified .
    polygon : gpd.GeoDataFrame
              Shapefile loaded using geopandas that contains the field(s) boundary(ies) of the area
    backCoeff : str
           Backscatter coefficient ('GAMMA0_TERRAIN', 'BETA0', 'SIGMA0_ELLIPSOID' or 'GAMMA0_ELLIPSOID')
    orbit_direction : str
           Orbit direction ('ASC', 'DESC', or 'BOTH')
    speckle_lee_window : int
       Window (in pixels) to apply spatial speckle filtering
    n_threads : int
                Number of threads used to download the EOPatch
    Returns
    -------
    EOPatch

    """

    if backCoeff not in [
        "GAMMA0_TERRAIN",
        "BETA0",
        "SIGMA0_ELLIPSOID",
        "GAMMA0_ELLIPSOID",
    ]:
        raise ValueError(
            "Backscatter coefficient can only be 'GAMMA0_TERRAIN', 'BETA0', 'SIGMA0_ELLIPSOID' or 'GAMMA0_ELLIPSOID'"
        )
    if orbit_direction not in ["ASC", "DES", "BOTH"]:
        raise ValueError("orbit  can only be 'ASC', 'DES' or 'BOTH")

    # Request format to download Sentinel-1 IW GRD products
    time_difference = datetime.timedelta(hours=2)

    if orbit_direction == "ASC":
        data_collection = DataCollection.SENTINEL1_IW_ASC
    elif orbit_direction == "DESC":
        data_collection = DataCollection.SENTINEL1_IW_DES
    else:
        data_collection = DataCollection.SENTINEL1_IW

    input_task = SentinelHubInputTask(
        data_collection=data_collection,
        bands=["VV", "VH"],
        bands_feature=(FeatureType.DATA, "BANDS-S1-IW"),
        additional_data=[
            (FeatureType.MASK, "dataMask", "IS_DATA"),
            (FeatureType.DATA, "localIncidenceAngle"),
        ],
        resolution=10,
        time_difference=time_difference,
        config=config,
        max_threads=n_threads,
        aux_request_args={
            "dataFilter": {"acquisitionMode": "IW"},
            "processing": {
                "backCoeff": backCoeff,
                "speckleFilter": {
                    "type": "LEE",
                    "windowSizeX": speckle_lee_window,
                    "windowSizeY": speckle_lee_window,
                },
                "orthorectify": True,
                "demInstance": "COPERNICUS",
                "mosaicking": "ORBIT",
            },
        },
    )

    add_polygon_mask = preprocessing.PolygonMask(polygon)

    add_dem = SentinelHubDemTask("DEM", resolution=10, config=config)

    if path_out is None:
        save = utils_sh.EmptyTask()
    else:
        os.makedirs(path_out, exist_ok=True)
        save = SaveTask(
            path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH
        )

    output_task = OutputTask("eopatch")

    workflow_nodes = linearly_connect_tasks(
        input_task, add_dem, add_polygon_mask, save, output_task
    )
    workflow = EOWorkflow(workflow_nodes)

    field_bbox = utils.get_bounding_box(polygon)
    result = workflow.execute(
        {workflow_nodes[0]: {"bbox": field_bbox, "time_interval": time_stamp}}
    )

    return result.outputs["eopatch"]
