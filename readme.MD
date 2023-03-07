A small python module that brings together the features of SentinelHub, the eo-learn python library and OrfeoToolBox for basic downloading and processing of Earth observation data for crop monitoring. 


# Installation

To install the package, you must first clone the git repository to the desired folder

```bash
git clone git@github.com:j-desloires/eo-crops.git
```

Then, open Anaconda prompt and create the environment from environment.yml

```
cd eo-crops
conda env create -f environment.yml
conda activate eo-crops
pip install .
```


If you want to run a jupyter notebook, you need to install jupyter on the environment. Then, you will be able to select the kernel.

```
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=eo-crops
jupyter notebook
```

# Dependencies

You should have OrfeoToolBox installed (https://www.orfeo-toolbox.org/CookBook/Installation.html) if you would like to apply pansharpening and multitemporal speckle filtering (Quegan). 

Also, you should have a Sentinelhub account with credentials. The latter can be sponsored using [Network of Resources program](https://nor-discover.cloudeo.group/Service/EDC-Sentinel-Hub/SponsoringWizardPricelist)

# Citation

```
@misc{eocrops_2022,
  title={eo-crops Python package for processing Earth observations for crop monitoring},
  author={Johann Desloires},
  year={2022},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/j-desloires/eo-crops}},
}
```
