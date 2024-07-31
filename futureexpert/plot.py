import datetime
from collections import defaultdict
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futureexpert.result_models import ForecastResult, Model

progColor = pd.DataFrame({
    'darkblue': ['#003652', '#34506c', '#62738a', '#949cae', '#c8cad5'],
    'cyan': ['#009ee3', '#00b0ea', '#57c5f2', '#a2d9f6', '#d3edfc'],
    'green': ['#58b396', '#85c3ab', '#a9d2bf', '#c9e1d5', '#e4f0ea'],
    'yellow': ['#f1bb69', '#f5ca89', '#f7d7a8', '#fae4c5', '#fcf2e3'],
    'violet': ['#a74b97', '#b671ad', '#c797c4', '#d8bada', '#ebddee'],
    'red': ['#d12b39', '#db5d59', '#e48a7f', '#eeb2a8', '#f6dad3']
})

# set the font globally
# plt.rcParams.update({'font.sans-serif':'Regular'})
mpl.rcParams['axes.titlesize'] = 12
plt.style.use('seaborn-v0_8-whitegrid')


def filter_models(models: list[Model], ranks: Optional[list[int]] = [1], model_names: Optional[list[str]] = None,) -> list[Model]:
    """Filter models based on the given criteria.

    Parameters
    ----------
    models
        List of models.
    model_names
        Names of the models to filtered by.
    ranks
        Ranks of the models to filtered by.
    """
    if model_names:
        models = [mo for mo in models if mo.model_name in model_names]
    if ranks:
        models = [mo for mo in models if mo.model_selection.ranking and mo.model_selection.ranking.rank_position in ranks]

    return models


def plot_forecast(result: ForecastResult,
                  plot_last_x_data_points_only: Optional[int] = None,
                  model_names: Optional[list[str]] = None,
                  ranks: Optional[list[int]] = [1]) -> Any:
    """Plots actuals and forecast from a single time series.

    Parameters
    ----------
    forecasts
        Forecasting results of a single time series and model.
    plot_last_x_data_points_only
        Number of data points of the actuals that should be shown in the plot.
    model_names
        Names of the models to plot.
    ranks
        Ranks of the models to plot.
    """

    actuals = result.input.actuals
    plot_models = filter_models(result.models, ranks, model_names)

    for model in plot_models:
        forecast = model.forecasts

        name = actuals.name
        model_name = model.model_name
        assert model.model_selection.ranking, 'No ranking, plotting not possible.'
        model_rank = model.model_selection.ranking.rank_position

        fc_date = [fc.time_stamp_utc for fc in forecast]
        # forecast values
        fc_value = [fc.point_forecast_value for fc in forecast]
        # forecast intervals
        fc_upper_value = [fc.upper_limit_value for fc in forecast]
        fc_lower_value = [fc.lower_limit_value for fc in forecast]

        values = actuals.values
        if plot_last_x_data_points_only is not None:
            values = actuals.values[-plot_last_x_data_points_only:]

        ac_date = [fc.time_stamp_utc for fc in values]
        ac_value = [fc.value for fc in values]

        df_ac = pd.DataFrame({'date': ac_date, 'actuals': ac_value})

        df_fc = pd.DataFrame({'date': fc_date, 'fc': fc_value, 'upper': fc_upper_value, 'lower': fc_lower_value})
        df_concat = pd.concat([df_ac, df_fc], axis=0).reset_index(drop=True)
        # connected forecast line with actual line
        df_concat.loc[len(ac_date)-1, 'fc'] = df_concat.loc[len(ac_date)-1, 'actuals']
        df_concat.loc[len(ac_date)-1, 'upper'] = df_concat.loc[len(ac_date)-1, 'actuals']
        df_concat.loc[len(ac_date)-1, 'lower'] = df_concat.loc[len(ac_date)-1, 'actuals']
        df_concat.date = pd.to_datetime(df_concat.date)

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        fig.suptitle(f'Forecast for {name}', fontsize=16)
        ax.set_title(f'using {model_name} (Rank {model_rank})')

        # plot
        ax.plot(df_concat.date, df_concat.actuals, color=progColor.loc[0, 'darkblue'], label='Time Series')
        ax.plot(df_concat.date, df_concat.fc, color=progColor.loc[0, 'cyan'], label='Forecast')

        if not any(v is None for v in df_concat.lower):
            ax.fill_between(df_concat.date, df_concat.lower, df_concat.upper,
                            color=progColor.loc[2, 'cyan'], alpha=0.30, label='Prediction Interval')

        # legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

        # style
        ax.set_frame_on(False)
        ax.tick_params(axis='both', labelsize=10)

        # margin
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.show()


def plot_backtesting(result: ForecastResult,
                     iteration: int = 1,
                     plot_last_x_data_points_only: Optional[int] = None,
                     model_names: Optional[list[str]] = None,
                     ranks: Optional[list[int]] = [1]) -> Any:
    """Plots actuals and backtesting results from a single time series.

    Parameters
    ----------
    forecasts
        Forecasting and backtesting results of a single time series and model.
    iteration
        Iteration of the backtesting forecast.
    plot_last_x_data_points_only
        Number of data points of the actuals that should be shown in the plot.
    model_names
        Names of the models to plot.
    ranks
        Ranks of the models to plot.
    """

    actuals = result.input.actuals
    plot_models = filter_models(result.models, ranks, model_names)

    for model in plot_models:
        forecast = model.model_selection.backtesting
        model_name = model.model_name
        assert model.model_selection.ranking, 'No ranking, plotting not possible.'
        model_rank = model.model_selection.ranking.rank_position

        word_len_dict = defaultdict(list)

        for word in forecast:
            word_len_dict[word.fc_step].append(word)

        iterations = max([len(word_len_dict[x]) for x in word_len_dict])
        if iteration > iterations:
            raise ValueError('Selected iteration was not calculated.')

        bt_round = [word_len_dict[x][iteration] for x in word_len_dict]

        backtesting_dates = [ac.time_stamp_utc for ac in bt_round]
        backtesting_fc = [ac.point_forecast_value for ac in bt_round]
        backtesting_upper = [ac.upper_limit_value for ac in bt_round]
        backtesting_lower = [ac.lower_limit_value for ac in bt_round]

        values = actuals.values
        if plot_last_x_data_points_only is not None:
            values = actuals.values[-plot_last_x_data_points_only:]

        actual_dates = [ac.time_stamp_utc for ac in values]
        actual_values = [ac.value for ac in values]

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        fig.suptitle(f'Backtesting of {actuals.name} Iteration: {iteration}', fontsize=16)
        ax.set_title(f'using {model_name} (Rank {model_rank})')

        # plot
        ax.plot(actual_dates, actual_values, color=progColor.loc[0, "darkblue"],  label='Time Series')
        ax.plot(backtesting_dates, backtesting_fc, color=progColor.loc[0, "cyan"], label='Forecast')
        if None not in backtesting_lower and None not in backtesting_upper:
            ax.fill_between(backtesting_dates, backtesting_lower, backtesting_upper,
                            color=progColor.loc[2, "cyan"], alpha=0.30, label='Prediction Interval')

        # legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

        # style
        ax.set_frame_on(False)

        # margin
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.show()
