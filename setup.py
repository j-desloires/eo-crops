import os
from setuptools import setup, find_packages


def parse_requirements(file):
    return sorted(
        (
            {
                line.partition("#")[0].strip()
                for line in open(os.path.join(os.path.dirname(__file__), file))
            }
            - set("")
        )
    )


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="eocrops",
    version="1.0.0",
    packages=[
        "eocrops",
        "eocrops.inputs",
        "eocrops.tasks",
        "eocrops.climatools",
        "eocrops.utils",
    ],
    url="https://github.com/j-desloires/eo-crops",
    author="Johann Desloires",
    author_email="johann.desloires@gmail.com",
    long_description=open("README.md", encoding="utf8").read(),
    python_requires=">=3, <4",
    extras_require={"PROD": parse_requirements("requirements.txt")},
    # Used for whole testing before pip finalized the install
    test_suite="eocrops.tests.suite",
    # Classifiers list used when deploying to PY.org
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: Not Distributable",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
