FROM continuumio/miniconda3:23.3.1-0
LABEL authors="j-desloires"


# Set the PATH to include Miniconda
ENV PATH /root/miniconda3/bin:$PATH

# Update conda and install any additional packages
#RUN conda install --name base -c conda-forge mamba
COPY environment.yml requirements.txt requirements-dev.txt  ./

RUN conda config --set ssl_verify no
RUN conda env create -f environment.yml
RUN echo "conda activate $(head -1 environment.yml | cut -d' ' -f2)" >> ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH
ENV CONDA_DEFAULT_ENV $(head -1 environment.yml | cut -d' ' -f2)

SHELL ["/bin/bash", "--login", "-c"]
#################################################################################################################
WORKDIR /eocrops
# Unittest
## Scripts
COPY ./tests ./tests
## Files used in the tests
COPY data ./data

# Examples for the documentation
COPY ./examples/ ./examples

# Modules
COPY ./eocrops ./eocrops

COPY environment.yml requirements.txt requirements-doc.txt setup.py README.md  ./
COPY ./docs ./docs

RUN conda activate $(head -1 environment.yml | cut -d' ' -f2)
RUN pip install --no-cache-dir -e .

# Test python imports
RUN python -c "import eocrops"
RUN python -c "import eolearn"
