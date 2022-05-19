lumos-ncpt-tools
------------

lumos-ncpt-tools is a bare-bones Python package for working with data from the NeuroCognitive Performance Test (NCPT; Lumos Labs, Inc.). NCPT data from ~750,000 adults are freely available at [placeholder]. For more details about the dataset, check out the Data Descriptor publication: [placeholder].   


Quick start guide
------------

1) Fork the repo from the command line:

```
git clone https://github.com/pauljaffe/lumos-ncpt-tools
```

2) Install Poetry to manage the dependencies (see https://python-poetry.org/docs/). After installing, make sure that Poetry's bin directory is in the 'PATH' environment variable by running `source $HOME/.poetry/env` from the command line. 

3) Install the lumos-ncpt-tools dependencies: Run `poetry install` from the command line within the local lumos-ncpt-tools repo. The complete set of dependencies is listed in pyproject.toml.

4) Download the data linked above. 

5) Check out demo.ipynb for an overview of how to use the package to analyze the NCPT data.  


Reproduce the figures and analyses from the paper
------------

The easiest and recommended way to reproduce the results from the paper is as follows:

1) Do steps 1-4 above.

2) Change the paths in make_paper.py to the local copy of the data and a folder to save the figures. 

3) Run the script to make the figures and reproduce the analyses from the top-level directory of the lumos-ncpt-tools repo:

```
poetry run python3 make_paper.py
```