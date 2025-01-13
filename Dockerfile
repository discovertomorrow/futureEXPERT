FROM python:3.12
WORKDIR /app
COPY ./requirements-dev.txt ./dist/
RUN python3 -m pip install pip-tools
RUN pip-sync ./dist/requirements-dev.txt
COPY ./example_data/ ./example_data/
COPY ./use_cases/ ./use_cases/
COPY ./run-e2e.sh ./generate-notebook-list.sh .
COPY ./notebooks/ ./notebooks/
COPY ./dist/futureexpert-*.whl ./dist/
RUN python3 -m pip install $(ls dist/*.whl)[dev]
