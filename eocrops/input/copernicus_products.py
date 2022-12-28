from sentinelhub import (
    DataCollection,
)

from eolearn.io import SentinelHubEvalscriptTask

from eocrops.utils import utils as utils
from eolearn.core import linearly_connect_tasks, EOWorkflow, FeatureType, OutputTask


class VegetationPPI:
    def __init__(self, config, name="PPI"):
        self.config = config
        self.name = name

    def execute(self, shapefile):

        byoc = DataCollection.define_byoc(
            collection_id="67c73156-095d-4f53-8a09-9ddf3848fbb6",
            name=self.name,
            service_url="https://creodias.sentinel-hub.com",
            is_timeless=True,
        )

        evalscript = """
        function setup() {
          return {
            input: ["SOSD", "EOSD", "LSLOPE", "RSLOPE", "SPROD", "MAXD", "dataMask"],
            output: [
                {id: "PPI", bands: 6, sampleType: SampleType.FLOAT32},
                {id:"dataMask", bands:1, sampleType: SampleType.UINT8 }
            ]
          }
        }
        
        function evaluatePixel(sample) {
            return {
                PPI: [sample.SOSD, sample.EOSD, sample.LSLOPE, sample.RSLOPE, sample.SPROD, sample.MAXD],
                dataMask : [sample.dataMask]
            }
        }
        """

        input_task = SentinelHubEvalscriptTask(
            features=[(FeatureType.DATA, "PPI"), (FeatureType.MASK, "dataMask")],
            data_collection=byoc,
            resolution=10,
            config=self.config,
            evalscript=evalscript,
        )

        output_task = OutputTask("eopatch")
        workflow_nodes = linearly_connect_tasks(input_task, output_task)
        workflow = EOWorkflow(workflow_nodes)

        field_bbox = utils.get_bounding_box(shapefile)
        result = workflow.execute({workflow_nodes[0]: {"bbox": field_bbox}})
        patch = result.eopatch()

        ls_features = ["SOSD", "EOSD", "LSLOPE", "RSLOPE", "SPROD", "MAXD"]

        dict_output = {}
        for i, name in enumerate(ls_features):
            dict_output[name] = patch.data["PPI"][..., i]

        return dict_output
