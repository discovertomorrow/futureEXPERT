# futureEXPERT

futureEXPERT offers high-quality forecasts for data experts with ease.

In case you don't want to use this Python client or access futureEXPERT via API, check out our frontend solution [futureNOW](https://www.future-forecasting.de/).

## Registration

If you do not have an account for [future](https://now.future-forecasting.de) yet, click [here](https://launch.future-forecasting.de/) to register for a free account.

## Installation

In order to use futureEXPERT, you need a Python environment with Python 3.9 or higher.

The futureEXPERT package can be directly installed with `pip` from our GitHub repository.

```
pip install git+https://github.com/discovertomorrow/futureexpert
```

## Getting started

To get started with futureEXPERT we recommend checking out the jupyter notebook [getting started](notebooks/getting_started.ipynb) to help you with your first steps.

## Ready-made use case templates

Utilize our use case templates to get started with your own business application right away.

- [Demand Planning](use_cases/demand_planning/demand_planning.ipynb) 
- [Sales Forecasting](use_cases/sales_forecasting/sales_forecasting.ipynb)

## Advanced usage

- [checkin configuration options](notebooks/checkin_configuration_options.ipynb) - Different options to prepare your data to time series.

An example on how to forecast with covariates is available in the jupyter notebook [forecast with covariates](notebooks/forecast_with_covariates.ipynb)

- [Advanced workflow FORECAST](notebooks/advanced_workflow.ipynb) - For more control about the single steps for generating a forecast.
- [Using covariates - MATCHER and FORECAST](notebooks/cov_matcher_and_forecast.ipynb?ref_type=heads) - Using covariates: Leverage MATCHER to identify predictive covariates, get ranking of all covariates with the best time lag & incorporate the result into your FORECAST.
- [Using covariates from POOL](notebooks/using_covariates_from_POOL.ipynb) - How to use potential influencing factors from POOL.

- [Working with results](notebooks/working_with_results.ipynb) - Overview of forecast result functions (e.g. export, plotting) and how to use them; further detailed information about the results (e.g. summary of forecasting methods).

- [API documentation](https://discovertomorrow.github.io/futureEXPERT) - Further information about all features and configurations.

## Contributing

You can report issues or send pull requests in our [GitHub project](https://github.com/discovertomorrow/futureexpert).

## Wiki for prognostica employees

Further information for prognostica employees can be found [here](https://git.prognostica.de/prognostica/future/futureapp/futureexpert/-/wikis/home)
