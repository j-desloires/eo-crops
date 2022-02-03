from eolearn.core import  FeatureType, EOTask
import numpy as np



class VegetationIndicesVHRS(EOTask):
    def __init__(self, feature_name) :
        self.feature_name = feature_name

    def calcul_ratio_vegetation_indices(self):
        self.NDVI = (self.B4 - self.B3)/(self.B4 + self.B3)
        self.NDWI = (self.B2 - self.B4)/(self.B2 + self.B4)
        #self.MSAVI2 = (2*self.B4 + 1 - ((2*self.B4 +1)^2)**0.5 - 8*(self.B4 - self.B3))/2
        self.VARI = (self.B2 - self.B3)/(self.B2 + self.B3 - self.B1)

    def execute(self, eopatch, **kwargs):
        arr0 = eopatch.data[self.feature_name]

        # Raw data
        self.B1 = arr0[..., 0]
        self.B2 = arr0[..., 1]
        self.B3 = arr0[..., 2]
        self.B4 = arr0[..., 3]
        #VIS
        self.calcul_ratio_vegetation_indices()
        eopatch.add_feature(FeatureType.DATA, "NDVI", self.NDVI[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "NDWI", self.NDWI[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "VARI", self.VARI[..., np.newaxis])

        return eopatch




class VegetationIndicesS2(EOTask) :
    '''Define a class of vegetation indices, which are computed from the metadata of sentinel2 images extracted'''

    def __init__(self, feature_name, mask_data=True) :
        self.feature_name = feature_name
        self.mask_data = mask_data

    def get_vegetation_indices(self) :
        '''Define vegetation indices which are simply ratio of spectral bands'''
        self.NDVI = (self.B8A-self.B04)/(self.B8A+self.B04)
        self.NDWI = (self.B8A-self.B11)/(self.B8A+self.B11)


    def execute(self, eopatch) :
        '''Add those vegeation indices to the eo-patch for futur use'''

        bands_array = eopatch.data[self.feature_name]
        illumination_array = eopatch.data['ILLUMINATION']

        valid_data_mask = eopatch.mask['VALID_DATA'] if self.mask_data else eopatch.mask['IS_DATA']

        if 'polygon_mask' in list(eopatch.mask_timeless.keys()) :

            bands_array = np.ma.array(bands_array,
                                      dtype=np.float32,
                                      mask=np.logical_or(~valid_data_mask.astype(np.bool), np.isnan(bands_array)),
                                      fill_value=np.nan)
            bands_array = bands_array.filled()

        # Raw data
        self.B02 = bands_array[..., 0]
        self.B03 = bands_array[..., 1]
        self.B04 = bands_array[..., 2]
        self.B05 = bands_array[..., 3]
        self.B06 = bands_array[..., 4]
        self.B07 = bands_array[..., 5]
        self.B08 = bands_array[..., 6]
        self.B8A = bands_array[..., 7]
        self.B11 = bands_array[..., 8]
        self.B12 = bands_array[..., 9]
        self.viewZenithMean = illumination_array[..., 0]
        self.sunZenithAngles = illumination_array[..., 1]
        self.viewAzimuthMean = illumination_array[..., 2]
        self.sunAzimuthAngles = illumination_array[..., 3]
        self.get_vegetation_indices()

        eopatch.add_feature(FeatureType.DATA, "NDVI", self.NDVI[..., np.newaxis])
        eopatch.add_feature(FeatureType.DATA, "NDWI", self.EVI2[..., np.newaxis])

        return eopatch




class EuclideanNorm(EOTask) :
    """
    The tasks calculates Euclidian Norm of all bands within an array:
    norm = sqrt(sum_i Bi**2),
    where Bi are the individual bands within user-specified feature array.
    """

    def __init__(self, feature_name, in_feature_name) :
        self.feature_name = feature_name
        self.in_feature_name = in_feature_name

    def execute(self, eopatch) :
        arr = eopatch.data[self.in_feature_name]
        norm = np.sqrt(np.sum(arr**2, axis=-1))

        eopatch.add_feature(FeatureType.DATA, self.feature_name, norm[..., np.newaxis])
        return eopatch




