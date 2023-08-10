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

import eocrops.utils.base_functions as utils
from eolearn.geometry.morphology import ErosionTask

from eolearn.core import RemoveFeatureTask
from eolearn.core import FeatureType, EOTask


class PolygonMask(EOTask):
    """
    EOTask that performs rasterization from an inputs shapefile into :
    - data_timeless feature 'FIELD_ID' (0 nodata; 1,...,N for each observation of the shapefile ~ object IDs)
    - mask_timeless feature 'MASK' (0 if pixels outside the polygon(s) from the shapefile, 1 otherwise

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

        # Get an ID for each polygon from the inputs shapefile
        self.geodataframe["FIELD_ID"] = list(range(1, self.geodataframe.shape[0] + 1))

        if self.geodataframe.shape[0] > 1:
            bbox = self.geodataframe.geometry.total_bounds
            MASK = sentinelhub.BBox(
                bbox=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], crs=self.geodataframe.crs
            )
            self.geodataframe["MASK"] = MASK.geometry
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
                (ftype, fname)
                for (ftype, fname) in eopatch.get_features()
                if ftype == FeatureType.DATA
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
            # eopatch.remove_feature(FeatureType.MASK, "CLM")

        return eopatch
