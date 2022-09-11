from eolearn.core import EOPatch, FeatureType, AddFeatureTask

import glob
import numpy as np

import os

from eolearn.geometry import ErosionTask
import eocrops.tasks.preprocessing as preprocessing
import warnings
from eocrops.tasks.vegetation_indices import VegetationIndicesS2 as s2_vis
###########################################################################################################

class EOPatchDataset:
    def __init__(self,
                 root_dir, features_data,
                 suffix='', resampling = None,
                 range_doy = (1, 365),
                 bands_name = 'BANDS-S2-L2A',
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
            resampling = dict(start = '01-01', end = '12-31', day_periods = 8)
            warnings.warn(
                'You must specify a resampling periods to make your observations comparable. The default is set to ' + str(resampling))
        self.resampling = resampling
        self.function = function
        self.bands_name = bands_name

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


    def _execute_gap_filling(self, eopatch,
                             resampled_range,
                             copy_features,
                             algorithm = 'linear'):

        """
        Gap-filling to interpolate missing pixels and/or resample time series
        Parameters
        ----------
        eopatch (EOPatch) : patch to preprocess
        resampled_range (tuple) : resampled range (e.g. ("2020-01-01", "2020-12-12"))
        copy_features (list) : original features to keep in the EOPatch
        algorithm (str) : type of algorithm to do gap-filling

        Returns EOPatch
        -------

        """
        kwargs = dict(copy_features=copy_features,
                      resampled_range=resampled_range,
                      algorithm = algorithm)

        features = [
            (FeatureType.DATA, fname)
            for ftype, fname, _, _, _ in self.features_data
        ]

        return self._interpolate_feature(eopatch=eopatch, features=features, **kwargs)

    def _prepare_eopatch(self, patch, resampled_range, algorithm = 'linear', disk_radius = 1):
        '''
        Preprocess EOPatch (making pixels, gap filling and temporal resampling)
        Parameters
        ----------
        patch (EOPatch) : patch to preprocess
        resampled_range (tuple) : period to resample the patch
        algorithm (str) : typle of algorithm for gap filling (linear or cubic)
        disk_radius (int) : Number of pixels to erode w.r.t field boundaries

        Returns EOPatch
        -------

        '''
        if algorithm not in ['linear', 'cubic']:
            raise ValueError('Algorithm can only be "linear" of "cubic"')

        polygon_mask = (patch.data_timeless['FIELD_ID']>0).astype(np.int32)
        add_feature = AddFeatureTask(self.mask)
        add_feature.execute(eopatch=patch, data=polygon_mask.astype(bool))
        #patch.add_feature(self.mask[0], self.mask[1], polygon_mask.astype(bool))

        erode = ErosionTask(mask_feature=self.mask, disk_radius=disk_radius)
        erode.execute(patch)

        add_vis = s2_vis(biophysical=False, feature_name=self.bands_name)
        add_vis.execute(eopatch=patch)

        patch = self._execute_gap_filling(eopatch=patch,
                                          resampled_range=resampled_range,
                                          algorithm=algorithm,
                                          copy_features=[self.mask])
        return patch

    @staticmethod
    def _retrieve_range_doy(path, meta_file, range_doy, path_column,
                            planting_date_column, harvest_date_column):
        '''
        Get for each EOPatch the start and end of the season.

        Parameters
        ----------
        path (str) : path of the EOPatch saved
        meta_file (pd.DataFrame) : DataFrame with additional information for each EOPatch
        range_doy (tuple) : approximate range of starting and ending of the season if not planting and harvest dates available
        path_column (str) : column name from the meta_file with EOPatch paths
        planting_date_column (str) : column name from the meta_file with EOPatch paths
        harvest_date_column (str) : column name from the meta_file with EOPatch paths

        Returns (tuple)  start of the season and the end of the season.
        -------

        '''

        meta_file_subset = meta_file[meta_file[path_column] == path]
        if meta_file_subset.shape[0] == 0:
            raise ValueError('We cannot find a correspond path for the patch in the meta file')
        if planting_date_column is not None:
            if planting_date_column not in meta_file_subset.columns:
                raise ValueError('The column ' + planting_date_column + ' is not in the meta file')
            range_doy[0] = meta_file_subset[planting_date_column].values[0]
            if np.isnan(range_doy[0]):
                range_doy[0] = np.nanmedian(meta_file[planting_date_column].values)

        if harvest_date_column is not None:
            if harvest_date_column not in meta_file_subset.columns:
                raise ValueError('The column ' + harvest_date_column + ' is not in the meta file')
            range_doy[1] = meta_file_subset[harvest_date_column].values[0]
            if np.isnan(range_doy[1]):
                range_doy[1] = np.nanmedian(meta_file[harvest_date_column].values)

        return range_doy


    def _read_patch(self, path, algorithm = 'linear',
                    doubly_logistic = False,
                    asym_gaussian = False,
                    return_params = False,
                    meta_file = None,
                    path_column = None,
                    planting_date_column = None, harvest_date_column = None,
                    window_planting = 0, window_harvest = 0):

        '''
        Read and preprocess EOPatch given a path.
        Asymmetric and logistic function can be fitted (https://www.sciencedirect.com/science/article/abs/pii/S0034425712001629)

        Parameters
        ----------
        path (str) : path where the EOPatch is saved
        algorithm (str) : type of algorithm to interpolate missing pixels
        doubly_logistic (bool) : reconstruct time series using doubly logistic fitting
        asym_gaussian (bool) : reconstruct time series using asymmetric gaussian
        return_params (bool) : return parameters estimated from doubly or assymetric functions to get growth metrics
        meta_file (pd.DataFrame) : DataFrame with meta info regarding planting and dates. Useful for thresholding before preprocessing.
        path_column (str) : column from meta_file which give the EOPatch path for a given observation.
        planting_date_column (str) : column from meta_file where planting date column (DOY) is referred
        harvest_date_column (str) : column from meta_file where harvest date column (DOY) is referred
        window_planting (int) : window before planting date to fit asymmetric ou doubly before N days
        window_harvest (int) : window after planting date to fit asymmetric ou doubly after N days

        Returns (np.arrray) : 3D np.array (1, t, d) ~ EOPatch resampled
        -------

        '''
        def _func(path):
            path = path.numpy().decode('utf-8')
            # Load only relevant features
            ################################################################
            #path = os.path.join(root_dir, os.listdir(root_dir)[0])
            patch = EOPatch.load(path)

            if doubly_logistic or asym_gaussian:
                resampled_range = None
            else:
                year = str(patch.timestamp[0].year)
                start, end = '{0}-{1}'.format(year, self.resampling['start']),\
                             '{0}-{1}'.format(year, self.resampling['end'])
                resampled_range = (start, end, self.resampling['day_periods'])

            patch = self._prepare_eopatch(patch, resampled_range, algorithm)
            range_doy = self.range_doy

            if meta_file is not None:
                range_doy = self._retrieve_range_doy(path, meta_file, list(range_doy),
                                                     path_column, planting_date_column,
                                                     harvest_date_column)

            range_doy = tuple([int(range_doy[0]) - window_planting,
                               int(range_doy[1]) + window_harvest])

            curve_fitting = preprocessing.DoublyLogistic(range_doy = range_doy)

            if asym_gaussian:
                curve_fitting = preprocessing.AsymmetricGaussian(range_doy = range_doy)

            #################################################################
            data = []

            for feat_type, feat_name, _, dtype, _ in self.features_data:
                if doubly_logistic or asym_gaussian:

                    resampling = self.resampling['day_periods']

                    doy, arr = curve_fitting.execute(eopatch = patch,
                                                     feature=feat_name,
                                                     feature_mask=self.mask[-1],
                                                     resampling=resampling)

                    params = curve_fitting.params
                    if return_params:
                        data.append(params)
                    else:
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
        '''
        Subset the vector file to match each EOPatch

        Parameters
        ----------
        vector_data (pd.DataFrame) : dataframe with meta info regarding each EOPatch saved
        features_list (list) : list of features to read
        column_path (str) : column name with the corresponding path of each EOPatch

        Returns (np.array) : np.array from the input file
        -------

        '''

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
                         doubly_logistic = False,
                         asym_gaussian = False,
                         return_params = False,
                         meta_file=None,
                         path_column=None,
                         planting_date_column=None,
                         harvest_date_column=None,
                         window_planting = 0, window_harvest = 0
                         ):
        '''

        Parameters
        ----------
        algorithm (str) : type of algorithm to interpolate missing pixels
        doubly_logistic (bool) : reconstruct time series using doubly logistic fitting
        asym_gaussian (bool) : reconstruct time series using asymmetric gaussian
        return_params (bool) : return parameters estimated from doubly or assymetric functions to get growth metrics
        meta_file (pd.DataFrame) : DataFrame with meta info regarding planting and dates. Useful for thresholding before preprocessing.
        path_column (str) : column from meta_file which give the EOPatch path for a given observation.
        planting_date_column (str) : column from meta_file where planting date column (DOY) is referred
        harvest_date_column (str) : column from meta_file where harvest date column (DOY) is referred
        window_planting (int) : window before planting date to fit asymmetric ou doubly before N days
        window_harvest (int) : window after planting date to fit asymmetric ou doubly after N days

        Returns Returns (np.arrray) : 3D np.array (N, t, d) ~ EOPatch aggregated and resampled into the same np.array
        -------

        '''


        self._instance_tf_ds()
        ds_numpy = self.dataset.map(lambda x : self._read_patch(path = x,
                                                                algorithm=algorithm,
                                                                doubly_logistic = doubly_logistic,
                                                                asym_gaussian = asym_gaussian,
                                                                return_params=return_params,
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
        Read the vector file into a numpy array.
        Each observation will match the eopatch_tfds and will be converted into a np.array
        Parameters
        ----------
        vector_data (pd.DataFrame) : dataframe with meta info regarding each EOPatch saved
        features_list (list) : list of features to read
        column_path (str) : column name with the corresponding path of each EOPatch

        Returns (np.array) : np.array from the input file
        -------

        '''

        self._instance_tf_ds()
        out_labels = list(self.vector_dataset.map(
            lambda path : self._read_vector_data(
                path, column_path, vector_data, features_list),
            num_parallel_calls=self.AUTOTUNE))

        npy_labels = self._format_feature(out_labels)
        npy_labels = npy_labels.reshape(npy_labels.shape[0], npy_labels.shape[-1])

        return npy_labels
