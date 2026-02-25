ARG VARIANT
FROM mcr.microsoft.com/vscode/devcontainers/python:3-${VARIANT}

RUN apt-get update && apt-get install -y openjdk-21-jre && apt-get clean

### Add root commands above ###
USER vscode

RUN . /usr/local/share/nvm/nvm.sh && nvm install 20

RUN pip install -U jupyter pytest ipytest mypy autopep8 isort

# add some useful history entries
RUN echo "pip install -e .[dev]" >> ~/.bash_history && \
    echo "isort ." >> ~/.bash_history && \
    echo "autopep8 --recursive --in-place ." >> ~/.bash_history
