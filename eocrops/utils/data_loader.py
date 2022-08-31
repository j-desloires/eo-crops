from eolearn.core import EOPatch, FeatureType, AddFeatureTask

import glob
import numpy as np

import os

from eolearn.geometry import ErosionTask
import eocrops.tasks.preprocessing as preprocessing
import warnings
from scipy import interpolate as interpolate
###########################################################################################################

class EOPatchDataset:
    def __init__(self,
                 root_dir, features_data,
                 suffix='', resampling = None,
                 range_doy = (1, 365),
                 function=np.nanmedian):
        """
        root_dir (str) : root path where EOPatch are saved
        features_data (list of tuples) : features to aggregate in the dataset
        suffix (str) : suffix of the EOPatch file names to read only a subset of EOPatch (e.g. '_S2'). This is very useful if you have several data sources in the same root_dir.
        resampling (dict) : resampling period to make EOPatch at the same time scale and timely comparables (e.g. 8-days period)
        range_doy (tuple) : suset of the time series w.r.t a range of day of the year (e.g. between the 1st day and the 365th day)
        function (np.functon) : function to aggegate pixels not masked into a single time series
        """

        import tensorflow as tf
        import tensorflow_datasets as tfds
        global tf
        global tfds

        self.root_dir = root_dir

        self.features_data = features_data
        self.suffix = suffix
        self.mask = (FeatureType.MASK_TIMELESS, 'MASK')
        self.range_doy = range_doy

        if resampling is None:
            resampling = dict(start = '-01-01', end = '-12-31', day_periods = 8)
            warnings.warn(
                'You must specify a resampling periods to make your observations comparable. The default is set to ' + str(resampling))
        self.resampling = resampling
        self.function = function

        try:
            self.AUTOTUNE = tf.data.AUTOTUNE
        except:
            self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def _instance_tf_ds(self):
        '''
        initalize tf.data.Dataset w.r.t the file names in self.root_dir_or_list
        '''
        file_pattern = os.path.join(self.root_dir, '*' + self.suffix)
        files = glob.glob(file_pattern)
        if len(files) == 0:
            raise ValueError('No file in the root directory ' + self.root_dir + " ending with " + self.suffix)
        files.sort()
        self.dataset = tf.data.Dataset.from_tensor_slices(files)
        self.vector_dataset = tf.data.Dataset.from_tensor_slices(files)

    @staticmethod
    def _interpolate_feature(eopatch, features,  **kwargs):
        '''
        Perform gapfilling over a new time window or not.
        '''
        kwargs['features'] = features
        interp = preprocessing.InterpolateFeatures( **kwargs)
        eopatch = interp.execute(eopatch)
        return eopatch

    def _resamping_timeseries(self, arr, doy):
        '''
        Resample time series over a new periods after double logistic curve fitting
        '''

        xnew = np.arange(0, 365, self.resampling['day_periods'])
        start = np.where(xnew >= doy[0])[0][0]
        end = np.where(xnew <= doy[-1])[0][-1]

        flinear_cspline = interpolate.Akima1DInterpolator(doy, arr)
        ylinear_cspline = flinear_cspline(xnew[start:end])

        before_season = np.repeat(ylinear_cspline[0], xnew[:start].shape[0])
        before_season = before_season.reshape(before_season.shape[0] // ylinear_cspline.shape[1],
                                              ylinear_cspline.shape[1])

        after_season = np.repeat(ylinear_cspline[-1], xnew[end:].shape[0])
        after_season = after_season.reshape(after_season.shape[0] // ylinear_cspline.shape[1],
                                              ylinear_cspline.shape[1])

        return np.concatenate([before_season, ylinear_cspline, after_season], axis=0)

    def _execute_gap_filling(self, eopatch,
                             resampled_range,
                             copy_features,
                             algorithm = 'linear'):

        '''Gap filling after data extraction, very useful if did not include it in the data extraction workflow'''
        kwargs = dict(copy_features=copy_features,
                      resampled_range=resampled_range,
                      algorithm = algorithm)

        features = []
        for ftype, fname, _, _, _ in self.features_data:
            features.append((FeatureType.DATA, fname))
        new_eopatch = self._interpolate_feature(eopatch=eopatch, features=features, **kwargs)

        return new_eopatch

    def _prepare_eopatch(self, patch, resampled_range, algorithm = 'linear'):

        polygon_mask = (patch.data_timeless['FIELD_ID']>0).astype(np.int32)
        add_feature = AddFeatureTask(self.mask)
        add_feature.execute(eopatch=patch, data=polygon_mask.astype(bool))
        #patch.add_feature(self.mask[0], self.mask[1], polygon_mask.astype(bool))

        erode = ErosionTask(mask_feature=self.mask, disk_radius=1)
        erode.execute(patch)

        patch = self._execute_gap_filling(eopatch=patch,
                                          resampled_range=resampled_range,
                                          algorithm=algorithm,
                                          copy_features=[self.mask])
        return patch

    @staticmethod
    def _retrieve_range_doy(path, meta_file, range_doy, path_column,
                            planting_date_column, harvest_date_column):

        meta_file_subset = meta_file[meta_file[path_column] == path]
        if meta_file_subset.shape[0] == 0:
            raise ValueError('We cannot find a correspond path for the patch in the meta file')
        else:
            if planting_date_column is not None:
                if planting_date_column not in meta_file_subset.columns:
                    raise ValueError('The column ' + planting_date_column + ' is not in the meta file')
                else:
                    range_doy[0] = meta_file_subset[planting_date_column].values[0]
            if harvest_date_column is not None:
                if harvest_date_column not in meta_file_subset.columns:
                    raise ValueError('The column ' + harvest_date_column + ' is not in the meta file')
                else:
                    range_doy[1] = meta_file_subset[harvest_date_column].values[0]

        return range_doy


    def _read_patch(self, path, algorithm = 'linear',
                    doubly_logistic = False, return_params = False,
                    fit_resampling = False, meta_file = None,
                    path_column = None,
                    planting_date_column = None, harvest_date_column = None,
                    window_planting = 0, window_harvest = 0):

        """ TF op for reading an eopatch at a given path. """
        def _func(path):
            path = path.numpy().decode('utf-8')
            # Load only relevant features
            ################################################################
            #path = os.path.join(root_dir, os.listdir(root_dir)[0])
            patch = EOPatch.load(path)
            if not doubly_logistic:
                year = str(patch.timestamp[0].year)
                start, end = year + self.resampling['start'], year + self.resampling['end']
                resampled_range = (start, end, self.resampling['day_periods'])
            else:
                resampled_range = None

            patch = self._prepare_eopatch(patch, resampled_range, algorithm)
            range_doy = self.range_doy

            if meta_file is not None:
                range_doy = self._retrieve_range_doy(path, meta_file, list(range_doy),
                                                     path_column, planting_date_column,
                                                     harvest_date_column)

            curve_fitting = preprocessing.CurveFitting(range_doy = tuple([int(range_doy[0]) - window_planting,
                                                                          int(range_doy[1]) + window_harvest]))
            #################################################################
            data = []
            #self = curve_fitting
            for feat_type, feat_name, _, dtype, _ in self.features_data:
                if doubly_logistic:
                    if fit_resampling:
                        resampling = self.resampling['day_periods']
                    else:
                        resampling = 0

                    doy, arr = curve_fitting.execute(eopatch = patch,
                                                     feature=feat_name,
                                                     feature_mask=self.mask[-1],
                                                     resampling=resampling)

                    params = curve_fitting.params
                    if return_params:
                        data.append(params)
                    else:
                        if not fit_resampling:
                            arr = self._resamping_timeseries(arr.reshape(arr.shape[0], 1), doy)
                        data.append(arr)
                else:
                    arr = curve_fitting.get_time_series_profile(eopatch=patch,
                                                                feature=feat_name,
                                                                feature_mask=self.mask[-1],
                                                                function=self.function)
                    data.append(arr)

            return data

        #################################################################
        out_types = [tf.as_dtype(data[3]) for data in self.features_data]
        data = tf.py_function(_func, [path], out_types)

        out_data = {}
        for f_data, feature in zip(self.features_data, data):
            feat_type, feat_name, out_name, dtype, _ = f_data
            feature.set_shape(feature.get_shape())
            out_data[out_name] = feature

        return out_data

    def _read_vector_data(self, path, column_path, vector_data, features_list):
        """
        TF op for reading an eopatch at a given path.
        It must have a column with the corresponding path
        """

        def _func(path):
            path = path.numpy().decode('utf-8')
            vector_data_ = vector_data.copy()
            vector_data_ = vector_data_[vector_data_[column_path] == path]

            data = []
            for feat, dtype_ in features_list:
                data.append(np.array(vector_data_[feat].astype(dtype_)))
            return data

        data = tf.py_function(_func, [path], [feat[1] for feat in features_list])
        out_data = {}
        for fname, feature in zip(features_list, data):
            feature.set_shape(feature.get_shape())
            out_data[fname] = feature
        return out_data

    @staticmethod
    def _format_feature(out_feature):
        out_df = [np.concatenate([np.expand_dims(value, axis=1)
                                  if len(value.shape) == 1 else value
                                  for key, value in dicto.items()], axis=-1)
                  for dicto in out_feature]

        out_df = [np.expand_dims(k, axis=0) for k in out_df]
        return np.concatenate(out_df, axis=0)

    def get_eopatch_tfds(self, algorithm = 'linear',
                         doubly_logistic = False, return_params = False, fit_resampling = False,
                         meta_file=None,
                         path_column=None,
                         planting_date_column=None, harvest_date_column=None,
                         window_planting = 0, window_harvest = 0
                         ):
        '''
        Aggregate all the EOPatch files into a single 3D array, where each observation is summarized muiltivariate time series (e.g. median NDVI and NDWI) of one EOPatch

        Parameters
        ----------
        algorithm (str): name of the algorithm for gapfilling (linear or cubic)
        '''

        self._instance_tf_ds()
        ds_numpy = self.dataset.map(lambda x : self._read_patch(path = x,
                                                                algorithm=algorithm,
                                                                doubly_logistic = doubly_logistic,
                                                                return_params=return_params,
                                                                fit_resampling = fit_resampling,
                                                                meta_file=meta_file,
                                                                path_column=path_column,
                                                                planting_date_column=planting_date_column,
                                                                harvest_date_column=harvest_date_column,
                                                                window_planting = window_planting,
                                                                window_harvest=window_harvest
                                                                ),
                                    num_parallel_calls=self.AUTOTUNE)
        out_feature = list(ds_numpy)
        return self._format_feature(out_feature)


    def get_vector_tfds(self, vector_data, features_list, column_path):
        '''
        Get ground truth data from a given dataframe into a 2D array. Each observation will match with the aggregation of EOPatch
        '''

        self._instance_tf_ds()
        out_labels = list(self.vector_dataset.map(
            lambda path : self._read_vector_data(
                path, column_path, vector_data, features_list),
            num_parallel_calls=self.AUTOTUNE))

        npy_labels = self._format_feature(out_labels)
        npy_labels = npy_labels.reshape(npy_labels.shape[0], npy_labels.shape[-1])

        return npy_labels



