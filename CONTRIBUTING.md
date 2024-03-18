# Development
mypy .
autopep8 --recursive --in-place .
isort .


## Coding basics

Package installation

Install package with package dependencies:

`pip install -e .`


Install package with development dependencies:

`pip install -e .[dev]`
