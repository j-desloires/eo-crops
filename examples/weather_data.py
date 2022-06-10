import os
import pandas as pd

import eocrops
from eocrops.input.meteoblue import CEHUBExtraction, CEHubFormatting

os.getcwd()

input_file = pd.read_csv('./examples/layers/Data-Points-Valid-Burkina-MeteoBlue.csv')
input_file['coordinates'] = [(x,y) for x,y in zip(input_file['Longitude'], input_file['Latitude'])]
input_file['Id_location'] = input_file['Id_location'].astype(str)

queryBackbone = {
        "units": {
        "temperature": "C",
        "velocity": "km/h",
        "length": "metric",
        "energy": "watts"
    },
    "timeIntervalsAlignment": None,
    "runOnJobQueue": True,
    "oneTimeIntervalPerGeometry": True,
    "checkOnly": False,
    "requiresJobQueue": False,
    "geometry": {
        "type": "GeometryCollection",
        "geometries": None
    },
    "format": "csvIrregular", # best format
    "timeIntervals":  None
}

input_file_mean_agg = input_file[input_file['Aggregation'].isin(['mean'])]

pipeline_cehub = CEHUBExtraction(api_key = 'syng63gdwiuhiudw',
                                 queryBackbone = queryBackbone,
                                 ids = input_file_mean_agg['Id_location'].values,
                                 coordinates=  input_file_mean_agg['coordinates'].values,
                                 years = input_file_mean_agg['Annee'].values)

stat = 'mean'

query = [{"domain": "ERA5", "gapFillDomain": "NEMS4",
          "timeResolution": "daily",
          "codes": [
              {"code": 52, "level": "2 m above gnd", "aggregation": stat},  # Relative Humidity
              {"code":  11, "level": "2 m above gnd", "aggregation": stat}, # air temperature (°C)
              {"code":  32, "level": "2 m above gnd", "aggregation": stat}, # Wind Speed
              {"code": 180, "level": "sfc", "aggregation": stat}, #wind gust
              {"code":  256, "level": "sfc","aggregation": stat}, # Diffuse Shortwave Radiation
              {"code":  56, "level": "2 m above gnd","aggregation": stat}, # Vapor Pressure Deficit
              {"code":  260, "level": "2 m above gnd","aggregation": stat}, # FAO Reference Evapotranspiration,
              {"code":  261, "level": "sfc", "aggregation": stat}, # Evapotranspiration
              {"code":  52, "level": "2 m above gnd","aggregation": stat}, # Relative humidity
          ],
}]


df_output = pipeline_cehub.execute(query = query,  time_interval = ('01-01', '12-31'))



