ARG VARIANT
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

ARG PROGNOSTICA_PYPI

RUN apt-get update && apt-get install -y openjdk-17-jre && apt-get clean

### Add root commands above ###
USER vscode

RUN pip config set global.extra-index-url $PROGNOSTICA_PYPI

# add some useful history entries
RUN echo "pip install -e .[dev]" >> ~/.bash_history && \   
    echo "isort ." >> ~/.bash_history && \
    echo "autopep8 --recursive --in-place ." >> ~/.bash_history
