# futureEXPERT

_futureEXPERT_ is a flexible Python toolkit designed to significantly simplify the process of building professional forecasting solutions.
It is built upon a **Smart Build** principle: a clear division of tasks that makes powerful forecasting accessible with ease, even without a deep data science background.
* **You focus on the "what"**: Designing the solution tailored to your specific domain and requirements, connecting your data and integrating the results into your workflow.
* **_futureEXPERT_ handles the "how"**: The complex methodological and technical details, from data preparation to forecast generation, are abstracted away for you.


The workflow is handled by four distinct modules:

1. *CHECK-IN*: Prepares your time series data. This module validates, cleans, and transforms your input data to ensure it's ready for forecasting.
2. *POOL*: Provides a library of curated external variables (e.g., economic indicators, weather data). You can search this continuously updated collection to find useful covariates for your forecast.
3. *MATCHER*: Ranks covariates to find the most impactful ones for your data. It takes your own covariates or variables from the *POOL*, determines their optimal time lag, and measures their predictive value against a baseline model.
4. *FORECAST*: Generates the final forecast. This module automatically selects the best model (from statistical, ML, and AI methods) for each time series and can incorporate the top-performing covariates identified by *MATCHER*.

The simplest workflow only contains *CHECK-IN* and *FORECAST* is described in the jupyter notebook [getting started](./notebooks/getting_started.ipynb).

In case you don't want to use this Python client or access futureEXPERT via API, check out our frontend solution [futureNOW](https://www.future-forecasting.de/).

## Registration

If you do not have an account for [future](https://now.future-forecasting.de) yet, click [here](https://launch.future-forecasting.de/) to register for a free account.

## Installation

In order to use futureEXPERT, you need a Python environment with Python 3.9 or higher.

The futureEXPERT package can be directly installed with `pip` from our GitHub repository.

```
pip install -U futureexpert
```

## Getting started

To get started with futureEXPERT we recommend checking out the jupyter notebook [getting started](./notebooks/getting_started.ipynb) to help you with your first steps. Also check our [quick start video tutorial](https://www.future-forecasting.de/video/getting-started/).


## Ready-made use case templates

Utilize our use case templates to get started with your own business application right away.

- [Demand Planning](./use_cases/demand_planning/demand_planning.ipynb)
- [Sales Forecasting](./use_cases/sales_forecasting/sales_forecasting.ipynb)

## Advanced usage

- [checkin configuration options](./notebooks/checkin_configuration_options.ipynb) - Different options to prepare your data to time series.

- [Advanced workflow FORECAST](./notebooks/advanced_workflow.ipynb) - For more control about the single steps for generating a forecast.
- [Using covariates for FORECAST](./notebooks/forecast_with_covariates.ipynb) - Create forecasts with covariates by using your own data of influencing factors.
- [Using covariates - MATCHER and FORECAST](./notebooks/cov_matcher_and_forecast.ipynb?ref_type=heads) - Using covariates: Leverage MATCHER to identify predictive covariates, get ranking of all covariates with the best time lag & incorporate the result into your FORECAST.
- [Using covariates from POOL](./notebooks/using_covariates_from_POOL.ipynb) - How to use potential influencing factors from POOL.
- [Cluster time series with ASSOCIATOR](./notebooks/associator.ipynb) - Identifying clusters of similar time series patterns and trend behaviour.
- [Working with results](./notebooks/working_with_results.ipynb) - Overview of forecast result functions (e.g. export, plotting) and how to use them; further detailed information about the results (e.g. summary of forecasting methods).

- [API documentation](https://discovertomorrow.github.io/futureEXPERT) - Further information about all features and configurations.

## Video tutorials

Check out our video tutorials for a quick introduction to various aspects of futureEXPERT.

- [Getting started](https://www.future-forecasting.de/video/getting-started/) from registration to first forecasts within minutes.
- [CHECK-IN](https://www.future-forecasting.de/video/check-in/) your data and create time series for your forecasting use case.

## Contributing

You can report issues or send pull requests in our [GitHub project](https://github.com/discovertomorrow/futureexpert).

## Wiki for prognostica employees

Further information for prognostica employees can be found [here](https://git.prognostica.de/prognostica/future/futureapp/futureexpert/-/wikis/home)
