import os
import numpy as np
import pandas as pd

import sentinelhub
from sentinelhub import CRS

import rasterio
import warnings


def save_numpy(path, array, name_array):
    """Save a numpy array"""
    with open(os.path.join(path, name_array + ".npy"), "wb") as f:
        np.save(f, array)


def MetaInfos(saving_path, N):
    """Get meta info from a .tif using rasterio. Nodata is set to np.nan"""
    with rasterio.open(saving_path) as src0:
        meta = src0.meta
        meta["nodata"] = np.nan
        meta["dtype"] = "float32"

    meta.update(count=N)
    meta.update(nodata=np.nan)

    return meta


def WriteTiff(array, saving_path, meta, dim=1):
    """Write a tiff file from a numpy array using rasterio"""
    with rasterio.open(saving_path, "w", **meta) as dst:
        if dim > 1:
            for id in range(dim):
                dst.write_band(id + 1, array[:, :, id].astype(np.float32))
        else:
            dst.write_band(1, array.astype(np.float32))


def check_crs(reference_file_):
    """Check crs from a GeoDataFrame and transform it into UTM"""
    reference_file = reference_file_.copy()
    reference_file_crs = reference_file.crs
    if reference_file_crs is None:
        reference_file.set_crs(epsg=4326, inplace=True)
        warnings.warn(
            "Your inputs GeoDataFrame should have a CRS! By default, it will set to WGS84"
        )
    elif str(reference_file_crs).split(":")[-1] != "4326":
        reference_file.to_crs(epsg=4326, inplace=True)

    xmin, ymin, xmax, ymax = reference_file.geometry.iloc[0].bounds
    utm_crs = str(CRS.get_utm_from_wgs84(xmin, ymin))
    reference_file.to_crs(utm_crs, inplace=True)

    return reference_file


def get_bounding_box(shapefile):
    """Get the bounding box of a given polygon for sentinelhub request"""
    shapefile = check_crs(shapefile)
    xmin, ymin, xmax, ymax = shapefile.geometry.total_bounds
    return sentinelhub.BBox(bbox=[(xmin, ymin), (xmax, ymax)], crs=str(shapefile.crs))


def get_resampled_periods(start, end, year, days_range=1):
    """
    Get the resampled periods from the resample range
    """
    import dateutil
    import datetime as dt

    resample_range_ = (str(year) + start, str(year) + end, days_range)

    start_date = dateutil.parser.parse(resample_range_[0])
    end_date = dateutil.parser.parse(resample_range_[1])
    step = dt.timedelta(days=resample_range_[2])

    days = [start_date]
    while days[-1] + step < end_date:
        days.append(days[-1] + step)
    days = [str(day).split(" ")[0] for day in days]
    return days


def concatenate_outputs(ds, output, fname_, id_column="path"):
    """Perform a loop on multiple dataframes on axis=0 and concatenate them"""
    print(fname_)

    new_cols = output.columns.to_flat_index()
    new_cols = [
        (k[0], int(k[1])) if k not in [id_column, (id_column, "")] else k
        for k in new_cols
    ]
    output.columns = new_cols
    output = output.rename(columns={(id_column, ""): id_column})

    if ds.empty:
        ds = ds.append(output)
    else:
        ds = pd.merge(ds, output, on=id_column, how="left")
    return ds
