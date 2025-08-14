"""Contains all the functionality to plot the checked in time series and the forecast and backtesting results."""
import copy
import datetime
import math
from collections import defaultdict
from typing import Any, Final, Hashable, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from futureexpert.forecast import (ChangedStartDate,
                                   ChangedValue,
                                   ChangePoint,
                                   ForecastResult,
                                   Model,
                                   Outlier,
                                   TimeSeriesCharacteristics)
from futureexpert.shared_models import Covariate, CovariateRef, TimeSeries

prog_color = pd.DataFrame({
    'darkblue': ['#003652', '#34506c', '#62738a', '#949cae', '#c8cad5'],
    'cyan': ['#009ee3', '#00b0ea', '#57c5f2', '#a2d9f6', '#d3edfc'],
    'green': ['#58b396', '#85c3ab', '#a9d2bf', '#c9e1d5', '#e4f0ea'],
    'gold': ['#f1bb69', '', '', '', ''],
    'darkgreen': ['#088670', '', '', '', ''],
    'lightgreen': ['#AAD18B', '', '', '', ''],
    'turquois': ['#34D1AE', '', '', '', ''],
    'marine': ['#1E5798', '', '', '', ''],
    'yellow': ['#f1bb69', '#f5ca89', '#f7d7a8', '#fae4c5', '#fcf2e3'],
    'violet': ['#a74b97', '#b671ad', '#c797c4', '#d8bada', '#ebddee'],
    'darkviolet': ['#6E5099', '', '', '', ''],
    'blueviolet': ['#8198F0', '', '', '', ''],
    'red': ['#d12b39', '#db5d59', '#e48a7f', '#eeb2a8', '#f6dad3'],
    'greyblue': ['#6f7593', '#8c91a9', '#a9acbe', '#c5c8d4', '#e2e3e9']
})

cov_column_color = [prog_color.loc[0, "violet"], prog_color.loc[0, "darkviolet"], prog_color.loc[0, "blueviolet"],
                    prog_color.loc[0, "lightgreen"], prog_color.loc[0, "turquois"], prog_color.loc[0, "darkgreen"],
                    prog_color.loc[0, "marine"]
                    ]
transparent_rgba = 'rgba(0,0,0,0)'
legend_position = {'loc': 'upper left'}

plot_labels = {'time_series': 'Time Series',
               'original_outlier': 'Original Outlier',
               'replace_value': 'Replacement Values',
               'removed_values': 'Removed Values',
               'few_observations': 'Few Observations',
               'pi': 'Prediction Interval'}

granularity_to_pd_alias: Final[dict[str, str]] = {
    'yearly': 'YS',
    'quarterly': 'QS',
    'monthly': 'MS',
    'weekly': 'W',
    'daily': 'D',
    'hourly': 'h',
    'halfhourly': '30min'
}
# set the font globally
# plt.rcParams.update({'font.sans-serif':'Regular'})
mpl.rcParams['axes.titlesize'] = 12
plt.style.use('seaborn-v0_8-whitegrid')


def filter_models(models: list[Model],
                  ranks: Optional[list[int]] = [1],
                  model_names: Optional[list[str]] = None,) -> list[Model]:
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


def _create_interactive_time_series_plot(df_ac: pd.DataFrame, name: str) -> None:
    """Creates a interactive plot for time series data.

    Parameters
    ----------
    df_ac
        Actuals data frame containing dates an values.
    name
        Name of the time series used as title.
    """
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        title_text=name,
        title_x=0.5
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )

    fig.add_trace(go.Scatter(x=df_ac.date, y=df_ac.actuals, mode='lines+markers',
                             name=plot_labels['time_series'],
                             line={'color': prog_color.loc[0, 'darkblue']},
                             marker={'color': prog_color.loc[0, 'darkblue'], 'size': 2},
                             connectgaps=False,
                             yaxis='y1'
                             ))

    covariate_column = [col for col in df_ac if col.startswith('covariate_lag')]
    if len(covariate_column) > 0:
        for idx, cov in enumerate(covariate_column):
            fig.add_trace(go.Scatter(x=df_ac.date, y=df_ac[cov],
                                     connectgaps=False,
                                     yaxis=f'y{idx+2}',
                                     mode='lines+markers', name=str(cov).replace('covariate_lag_', ''),
                                     marker={'color': cov_column_color[idx % len(cov_column_color)], 'size': 2},
                                     line={'color': cov_column_color[idx % len(cov_column_color)]}))

        count_y_axis = len(covariate_column) + 1
        fig.update_layout(
            {
                f"yaxis{'' if ax == 0 else ax+1}": {
                    "showticklabels": False,
                    "overlaying": None if ax == 0 else "y",
                }
                for ax in range(count_y_axis)
            }
        )

    fig.show()


def _create_static_time_series_plot(df_ac: pd.DataFrame,  name: str) -> None:
    """Creates a static plot for time series data.

    Parameters
    ----------
    df_ac
        Actuals data frame containing dates an values.
    name
        Name of the time series used as title.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    fig.suptitle(name, fontsize=16)
    ax.set_frame_on(False)
    ax.tick_params(axis='both', labelsize=10)

    covariate_column = [col for col in df_ac if col.startswith('covariate_lag')]

    ax.plot(df_ac.date, df_ac.actuals,
            marker='.', markersize=1,
            color=prog_color.loc[0, "darkblue"],
            label=plot_labels['time_series'] if len(covariate_column) > 0 else None)

    if len(covariate_column) > 0:
        _add_covariates_to_static_plot(ax, covariate_column, df_ac)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.show()


def plot_time_series(ts: TimeSeries,
                     covariate: Optional[list[Covariate]] = None,
                     plot_last_x_data_points_only: Optional[int] = None,
                     as_interactive: bool = False) -> None:
    """Plots actuals from a single time series. Optional a Covariate can be plotted next to it.

    Parameters
    ----------
    ts
        time series data
    covariate
        covariate data
    plot_last_x_data_points_only
        Number of data points of the actuals that should be shown in the plot.
    as_interactive
        Plots the data as an interactive plot or as static image.
    """
    df_ac = _prepare_actuals(actuals=ts, plot_last_x_data_points_only=plot_last_x_data_points_only)
    name = ts.name

    if covariate:
        df_ac = _add_covariates(df_ac, covariate, covariate,
                                _calculate_max_covariate_date(ts.granularity, df_ac.date.max()))

    if as_interactive:
        _create_interactive_time_series_plot(df_ac, name)
    else:
        _create_static_time_series_plot(df_ac, name)


def _calculate_max_covariate_date(granularity: str, start_date: datetime.datetime) -> datetime.datetime:
    """Calculates 60 data points into the future based on the start_date. Used for cropping covariates in plots.

    Parameters
    ----------
    granularity
        Granularity of the time series.
    start_date
         Date from which 60 steps should be calculated into the future. Usually last datapoint of actuals.
    """
    result: list[datetime.datetime] = pd.date_range(start=start_date,
                                                    periods=60,
                                                    freq=granularity_to_pd_alias.get(granularity))

    return result[-1]


def _fill_missing_values_for_plot(granularity: str,
                                  df: pd.DataFrame) -> pd.DataFrame:
    """Finds missing values in data and explicity fills them with nan. This is required for accruately
    displaying missing data in plots.

    Parameters
    ----------
    granularity
        Granularity of the time series.
    df
        Data with potentially missing values. Timestamp data needs to be in a column called date. Can handle any
        other number of columns.

    Returns
    -------
    Dataframe in the same structure as before, but with potentially added rows with nan values.
    """
    full_date_range = pd.date_range(start=df['date'].min(),
                                    end=df['date'].max(),
                                    freq=granularity_to_pd_alias.get(granularity))

    # Reindex the dataframe to fill in missing dates, leaving actuals as NaN for missing entries
    return df.set_index('date').reindex(full_date_range).reset_index().rename(columns={'index': 'date'})


def _calculate_few_observation_borders(actual_dates: list[datetime.datetime],
                                       few_observations: list[ChangePoint]
                                       ) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Calculates the time frame where few observations where found.

    Parameters
    ----------
    actual_dates
        Dates of the actuals.
    few_observations
        List of all detected few observation change points.

    Returns
    -------
    List where each entry contains the borders of a few observation time frame.
    """

    if len(few_observations) == 0:
        return []

    matched_few_observations = []
    few_observations.sort(key=lambda x: x.time_stamp_utc)
    for i, obs in enumerate(few_observations):
        if obs.change_point_type == 'FEW_OBSERVATIONS_LEFT':
            obs.time_stamp_utc = actual_dates[actual_dates.index(obs.time_stamp_utc) - 1]
    if few_observations[0].change_point_type == 'FEW_OBSERVATIONS_LEFT':
        few_observations.insert(0, ChangePoint(time_stamp_utc=actual_dates[0],
                                               change_point_type='FEW_OBSERVATIONS_RIGHT'))
    if few_observations[-1].change_point_type == 'FEW_OBSERVATIONS_RIGHT':
        few_observations.append(ChangePoint(time_stamp_utc=actual_dates[-1],
                                            change_point_type='FEW_OBSERVATIONS_LEFT'))
    for i in range(0, len(few_observations), 2):
        left_border = few_observations[i].time_stamp_utc
        right_border = few_observations[i+1].time_stamp_utc
        matched_few_observations.append((left_border, right_border))
    return matched_few_observations


def _add_level_shifts(df_ac: pd.DataFrame, level_shifts: list[ChangePoint]) -> pd.DataFrame:
    """Add the level shifts to the plot data.

    Parameters
    ----------
    df_ac
        Actuals data frame containing dates an values.
    level_shifts
        List of all detected level shift change points.

    Returns
    -------
    The input dataframe added with a column containing the level shift information.
    """

    if len(level_shifts) == 0:
        return df_ac
    index_last_actual = len(df_ac.index)-1
    df_level_shift = pd.DataFrame({'date': [x.time_stamp_utc for x in level_shifts],
                                   'level_shift':  'x'})

    df_ac = df_ac.merge(df_level_shift, on='date', how='left')
    shift_index = df_ac[df_ac.level_shift.notnull()].index.tolist()
    shift_index.append(index_last_actual)
    shift_index.append(0)
    shift_index.sort()

    for idx, x in enumerate(shift_index):
        if idx + 1 < len(shift_index):
            # only calculate mean for all values before the shift
            mean = df_ac.loc[x:shift_index[idx+1]-1, 'actuals'].mean()
            df_ac.loc[x:shift_index[idx+1], 'level_shift'] = mean

    return df_ac


def _add_outliers(df_ac: pd.DataFrame, outliers: Sequence[Outlier], changed_values: Sequence[ChangedValue]) -> pd.DataFrame:
    """Add outliers information to the plot data.

    Parameters
    ----------
    df_ac
        Actuals data frame containing dates an values.
    outliers
        List of all detected outliers.
    changed_values
        List auf all changed values.

    Returns
    -------
    The input dataframe added with columns containing the outlier information.
    """

    if len(outliers) == 0:
        return df_ac

    detected_outlier = pd.DataFrame({'date': [x.time_stamp_utc for x in outliers],
                                     'original_outlier': [x.original_value for x in outliers]})
    outlier = [x for x in changed_values if x.change_reason == 'outlier']
    changed_outliers = pd.DataFrame({'date': [x.time_stamp_utc for x in outlier],
                                     'replace_outlier': [x.changed_value for x in outlier]})

    df_outlier = detected_outlier.merge(changed_outliers, on='date', how='left', validate='1:1')
    if df_outlier.shape[0] == 0:
        return df_ac
    df_ac = df_ac.merge(df_outlier, on='date', how='left')
    replaced_outlier_index = df_ac[df_ac.replace_outlier.notnull()].index.tolist()
    df_ac.loc[replaced_outlier_index, 'actuals'] = df_ac.loc[replaced_outlier_index, 'replace_outlier']

    df_ac['outlier_connection'] = df_ac['original_outlier']
    outlier_indices = df_ac[df_ac.original_outlier.notnull()].index
    for idx in outlier_indices:
        if idx > 0:  # Connect to previous point
            df_ac.loc[idx-1, 'outlier_connection'] = df_ac.loc[idx-1, 'actuals']
        if idx < len(df_ac) - 1:  # Connect to next point
            df_ac.loc[idx+1, 'outlier_connection'] = df_ac.loc[idx+1, 'actuals']

    return df_ac


def _add_replaced_missings(df_ac: pd.DataFrame, changed_values: Sequence[ChangedValue]) -> pd.DataFrame:
    """Add replaced_missings to the plot data.

    Parameters
    ----------
    df_ac
        Actuals data frame containing dates an values.
    changed_values
        List auf all changed values.

    Returns
    -------
    The input dataframe added with columns containing the replaced missing information.
    """
    replaced_missings = [x for x in changed_values if x.change_reason == 'na_value']

    if len(replaced_missings) == 0:
        return df_ac

    df_replaced_missings = pd.DataFrame({'date': [x.time_stamp_utc for x in replaced_missings],
                                         'replaced_missing': [x.changed_value for x in replaced_missings]})
    df_ac = df_ac.merge(df_replaced_missings, on='date', how='left')
    not_null_index = df_ac[df_ac.replaced_missing.notnull()].index.tolist()
    df_ac.loc[not_null_index, 'actuals'] = df_ac.loc[not_null_index, 'replaced_missing']

    return df_ac


def _add_changed_start_date(df_ac: pd.DataFrame, changed_start_date: Optional[ChangedStartDate]) -> pd.DataFrame:
    """Adjust actuals based on changed_start_date.

    Parameters
    ----------
    df_ac
        Actuals data frame containing dates an values.
    changed_start_date
        Details about a changed start date of the time series.
    """
    # If changed_start_date ist not part of the plot scope (ts was shorten by plot_x_last_data_points)
    # skip logic to prevent an unnecessary entry in the legend
    if changed_start_date is not None and df_ac.date.min() < changed_start_date.changed_start_date:
        df_ac['removed_actuals'] = df_ac['actuals']
        df_ac.loc[df_ac.date > changed_start_date.changed_start_date, 'removed_actuals'] = np.NaN
        df_ac.loc[df_ac.date < changed_start_date.changed_start_date, 'actuals'] = np.NaN
    return df_ac


def _add_covariates(df_plot: pd.DataFrame,
                    model_covariates: Sequence[Union[CovariateRef, Covariate]],
                    input_covariates: Sequence[Covariate],
                    max_date: datetime.date) -> pd.DataFrame:
    """Add covariates to the plot data.

    Parameters
    ----------
    df_plot
        Data frame containing dates values of actuals, forecast and various preprocessing information.
    model_covariates
        CovariateRef that were used in the model
    input_covariates
        Complete covariate data that were used in the report.
    max_date
        Max date to which covaraites are plotted.
    """

    for covariate_ref in model_covariates:
        if covariate_ref is None:
            continue
        if isinstance(covariate_ref, Covariate):
            covariate = covariate_ref
        else:
            covariate = next((x for x in input_covariates if x.ts.name == covariate_ref.name))

        cov_date = [value.time_stamp_utc for value in covariate.ts.values]
        cov_value = [value.value for value in covariate.ts.values]

        df_cov = pd.DataFrame({'date': cov_date, 'covariate': cov_value, 'covariate_lag': cov_value})
        df_plot = pd.merge(df_plot, df_cov, on='date', how='outer', validate='1:1').reset_index(drop=True)
        df_plot = df_plot.sort_values(by='date')
        df_plot.covariate_lag = df_plot.covariate_lag.shift(covariate.lag)
        columns_new_name_mapping = {'covariate': 'covariate_' + covariate.ts.name,
                                    'covariate_lag': 'covariate_lag_' + covariate.ts.name + ' and lag ' + str(covariate_ref.lag)}
        df_plot = df_plot.rename(columns=columns_new_name_mapping)

    # remove all covariate values from before the start of actuals
    min_value = df_plot[df_plot['actuals'].notna()][['date']].min()
    df_plot = df_plot[(df_plot['date'] >= min_value.iloc[0]) & (df_plot['date'] <= max_date)]
    df_plot.reset_index(drop=True, inplace=True)

    return df_plot


def _add_covariates_to_static_plot(ax: mpl.axes.Axes, covariate_column: list[Hashable], df_plot: pd.DataFrame) -> None:
    """Add covariates to the plot.

    Parameters
    ----------
    ax
        Matplotlip axes
    covariate_column
        List of column names for laged covariates.
    df_plot
        Data frame containing dates values of actuals, forecast, covariates and various preprocessing information
    """
    cov_axis: dict[int, mpl.axes.Axes] = {}
    lines, labels = ax.get_legend_handles_labels()

    for idx, cov in enumerate(covariate_column):
        cov_axis[idx] = ax.twinx()  # type: ignore
        cov_axis[idx].grid(False)
        cov_axis[idx].set_frame_on(False)
        cov_axis[idx].tick_params(axis='both', labelsize=10)
        cov_axis[idx].plot(df_plot.date, df_plot[cov], color=cov_column_color[idx % len(
            cov_column_color)], label=str(cov).replace('covariate_lag_', ''))

        cov_axis[idx].set_axis_off()

        lines2, labels2 = cov_axis[idx].get_legend_handles_labels()
        lines = lines + lines2
        labels = labels + labels2

    ax.legend(lines, labels, bbox_to_anchor=(1, 1), loc=legend_position['loc'])


def _prepare_actuals(actuals: TimeSeries, plot_last_x_data_points_only: Optional[int] = None) -> pd.DataFrame:
    """Converts the actual data into a pandas DataFrame.

    Parameters
    ----------
    actuals
        Time series data.
    plot_last_x_data_points_only
        Number of data points of the actuals that should be shown in the plot.
    """

    values = actuals.values
    if plot_last_x_data_points_only is not None:
        values = actuals.values[-plot_last_x_data_points_only:]

    actual_dates = [fc.time_stamp_utc for fc in values]
    actual_values = [fc.value for fc in values]

    # if the data has missing values, make sure to explicitly store them as nan. Otherwise the plot function will
    # display interpolated values. To use help function, a temporary df is needed.
    df_ac = pd.DataFrame({'date': actual_dates, 'actuals': actual_values})
    df_ac = _fill_missing_values_for_plot(granularity=actuals.granularity, df=df_ac)

    return df_ac


def _add_forecast(df_ac: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Add forecast to the plot data.

    Parameters
    ----------
    df_ac
        Data frame containing dates values of actuals, forecast and various preprocessing information.
    model
        Result data of one forecasting model.
    """

    forecast = model.forecasts
    index_last_actual = len(df_ac.index)-1

    fc_date = [fc.time_stamp_utc for fc in forecast]
    # forecast values
    fc_value = [fc.point_forecast_value for fc in forecast]
    # forecast intervals
    fc_upper_value = [fc.upper_limit_value for fc in forecast]
    fc_lower_value = [fc.lower_limit_value for fc in forecast]

    df_fc = pd.DataFrame({'date': fc_date, 'fc': fc_value, 'upper': fc_upper_value, 'lower': fc_lower_value})

    df_concat = pd.concat([df_ac, df_fc], axis=0).reset_index(drop=True)

    # connected forecast line with actual line
    df_concat.loc[index_last_actual, 'fc'] = df_concat.loc[index_last_actual, 'actuals']
    df_concat.loc[index_last_actual, 'upper'] = df_concat.loc[index_last_actual, 'actuals']
    df_concat.loc[index_last_actual, 'lower'] = df_concat.loc[index_last_actual, 'actuals']
    df_concat.date = pd.to_datetime(df_concat.date)
    df_concat.sort_values('date', inplace=True)
    df_concat.reset_index(drop=True, inplace=True)
    return df_concat


def _prepare_backtesting(model: Model, iteration: int) -> pd.DataFrame:
    """Converts the backtesting data of one model into a pandas DataFrame.

    Parameters
    ----------
    model
        Result data of one forecasting model.
    iteration
        Iteration of the backtesting forecast.
    """

    fc_step_dict = defaultdict(list)

    for bt in model.model_selection.backtesting:
        fc_step_dict[bt.fc_step].append(bt)

    highest_calculated_iteration = max([len(fc_step_dict[x]) for x in fc_step_dict])
    if iteration > highest_calculated_iteration:
        raise ValueError('Selected iteration was not calculated.')

    bt_round = [fc_step_dict[x][iteration] for x in fc_step_dict]

    backtesting_dates = [ac.time_stamp_utc for ac in bt_round]
    backtesting_fc = [ac.point_forecast_value for ac in bt_round]
    backtesting_upper = [ac.upper_limit_value for ac in bt_round]
    backtesting_lower = [ac.lower_limit_value for ac in bt_round]

    return pd.DataFrame({'date': backtesting_dates,
                         'fc': backtesting_fc,
                         'lower': backtesting_lower,
                         'upper': backtesting_upper})


def _calculate_replaced_missing_borders(df_ac: pd.DataFrame) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Reshapes the replaced missing information for the interactive plot.

    Parameters
    ----------
    df_ac
        DataFrame containing actuals and various preprocessing results.

    Returns
    -------
    Tupel with start and end date of a block with missing values
    """
    missing_borders: list[tuple[datetime.datetime, datetime.datetime]] = []
    current_missing_block_start = None
    min_relevant_actuals = df_ac[df_ac['actuals'].notna()].min()
    for index, row in df_ac[df_ac['date'] >= min_relevant_actuals['date']].iterrows():

        if not math.isnan(row['replaced_missing']):
            if current_missing_block_start is None:
                start_index = 0 if index == 0 else index - 1
                current_missing_block_start = df_ac.iloc[start_index]['date']

            end_index = index if index+1 == len(df_ac.index) else index+1
            if math.isnan(df_ac.iloc[end_index]['replaced_missing']) or \
                    current_missing_block_start is not None and index+1 == len(df_ac.index):
                assert current_missing_block_start is not None, 'Start date for replaced missing block is missing.'
                missing_borders.append((current_missing_block_start, df_ac.iloc[end_index]['date']))
                current_missing_block_start = None
    return missing_borders


def create_multiple_yaxis(count_y_axis: int) -> dict[str, Any]:
    """Create multiple yaxis for interactive plots.

    Parameters
    ----------
    count_y_axis
        Number of how many yaxis are needed.
    """
    return {
        f"yaxis{'' if ax == 0 else ax+1}": {
            "showticklabels": False,
            "overlaying": None if ax == 0 else "y",
        }
        for ax in range(count_y_axis)
    }


def _create_static_forecast_plot(title: str,
                                 subtitle: str,
                                 df_concat: pd.DataFrame,
                                 plot_prediction_intervals: bool,
                                 plot_few_observations: list[tuple[datetime.datetime, datetime.datetime]]) -> None:
    """Creates a static plot for the forecasting results.

    Parameters
    ----------
    title
        Title of the plot.
    subtitle
        Subtitle of the plot.
    df_concat
        DataFrame containing actuals and various preprocessing results.
    df_bt
        DataFrame containing the backtesting forecasts.
    plot_prediction_intervals
        Shows prediction intervals.
    plot_few_observations
        Information about few observation time frames.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    ax.set_title(subtitle)

    ax.plot(df_concat.date, df_concat.actuals,
            marker='.', markersize=1,
            color=prog_color.loc[0, 'darkblue'], label=plot_labels['time_series'])
    ax.plot(df_concat.date, df_concat.fc, color=prog_color.loc[0, 'cyan'], label='Forecast')

    if 'removed_actuals' in df_concat.columns:
        ax.plot(df_concat.date, df_concat.removed_actuals,
                color=prog_color.loc[3, 'greyblue'], label=plot_labels['removed_values'])
    if 'replaced_missing' in df_concat.columns:
        ax.fill_between(df_concat.date, df_concat.actuals.min(), df_concat.actuals.max(),
                        where=df_concat.replaced_missing.notnull(),
                        color=prog_color.loc[2, 'red'], alpha=0.30, label='Missings')

    if 'original_outlier' in df_concat.columns:
        ax.plot(df_concat.date, df_concat.original_outlier, 'o-',
                color=prog_color.loc[0, 'red'], label=plot_labels['original_outlier'])
        ax.plot(df_concat.date, df_concat.replace_outlier, 'o-',
                color=prog_color.loc[0, 'green'], label=plot_labels['replace_value'])
        ax.plot(df_concat.date, df_concat.outlier_connection, '-',
                color=prog_color.loc[4, 'red'], zorder=1)

    if 'level_shift' in df_concat.columns:
        ax.plot(df_concat.date, df_concat.level_shift, color=prog_color.loc[0, 'gold'], label='Levels')

    for idx, time_frame in enumerate(plot_few_observations):

        ax.fill_between(df_concat.date, df_concat.actuals.min(), df_concat.actuals.max(),
                        where=(df_concat.date >= time_frame[0]) & (df_concat.date <= time_frame[1]),
                        color=prog_color.loc[1, 'greyblue'], alpha=0.30,
                        label=plot_labels['few_observations'] if idx == 0 else None)

    if plot_prediction_intervals and not any(v is None for v in df_concat.lower):
        ax.fill_between(df_concat.date, df_concat.lower, df_concat.upper,
                        color=prog_color.loc[2, 'cyan'], alpha=0.30, label=plot_labels['pi'])

    covariate_column = [col for col in df_concat if col.startswith('covariate_lag')]
    if len(covariate_column) > 0:
        _add_covariates_to_static_plot(ax, covariate_column, df_concat)
    else:
        ax.legend(loc=legend_position['loc'], bbox_to_anchor=(1, 1))

    ax.set_frame_on(False)
    ax.tick_params(axis='both', labelsize=10)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.show()


def _create_interactive_forecast_plot(title: str,
                                      subtitle: str,
                                      df_concat: pd.DataFrame,
                                      plot_prediction_intervals: bool,
                                      missing_borders: list[tuple[datetime.datetime, datetime.datetime]],
                                      plot_few_observations: list[tuple[datetime.datetime, datetime.datetime]]) -> None:
    """Creates a interactive plot for the forecasting results.

    Parameters
    ----------
    title
        Title of the plot.
    subtitle
        Subtitle of the plot.
    df_concat
        DataFrame containing actuals and various preprocessing results.
    plot_prediction_intervals
        Shows prediction intervals.
    missing_borders
        Information about missing time frames.
    plot_few_observations
        Information about few observation time frames.
    """

    fig = go.Figure()

    fig.update_layout(
        plot_bgcolor='white',
        title=f'{title}<br><sup>{subtitle}</sup>',
        title_x=0.5,
        title_font={'size': 16},
        xaxis_title='Date',
        yaxis_title='Value'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )

    fig.add_trace(go.Scatter(x=df_concat.date,
                             y=df_concat.actuals,
                             connectgaps=False,
                             mode='lines+markers', name=plot_labels['time_series'],
                             marker={'color': prog_color.loc[0, 'darkblue'], 'size': 2},
                             line={'color': prog_color.loc[0, 'darkblue']}))

    if 'removed_actuals' in df_concat.columns:
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.removed_actuals,
                                 connectgaps=False,
                                 mode='lines', name=plot_labels['removed_values'], line={'color': prog_color.loc[3, 'greyblue']}))

    if plot_prediction_intervals and not any(v is None for v in df_concat.lower):
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.upper,
                                 line_color=transparent_rgba,
                                 mode='lines',
                                 hoverinfo='skip',
                                 showlegend=False))
        pi_color = mpl.colors.to_rgba(prog_color.loc[2, "cyan"], alpha=0.3)
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.lower,
                                 line_color=transparent_rgba,
                                 mode='lines', fill='tonexty', name=plot_labels['pi'],
                                 fillcolor=f"rgba({pi_color[0]}, {pi_color[1]}, {pi_color[2]}, {pi_color[3]})"))

    fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.fc,
                             mode='lines', name='Forecast', line={'color': prog_color.loc[0, 'cyan']}))

    if 'replaced_missing' in df_concat.columns:
        missing_color = mpl.colors.to_rgba(prog_color.loc[2, 'red'], alpha=0.3)
        missing_color_str = f"rgba({missing_color[0]}, {missing_color[1]}, {missing_color[2]}, {missing_color[3]})"

        for idx, time_frame in enumerate(missing_borders):
            missing_dates = df_concat.date[(df_concat.date >= time_frame[0]) & (df_concat.date <= time_frame[1])]
            fig.add_trace(go.Scatter(x=missing_dates, y=[df_concat.actuals.max()] * len(missing_dates),
                                     line_color=transparent_rgba,
                                     mode='lines',
                                     hoverinfo='skip',
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=missing_dates, y=[df_concat.actuals.min()] * len(missing_dates),
                                     mode='lines', fill='tonexty',
                                     line_color=transparent_rgba,
                                     legendgroup='2',
                                     showlegend=True if idx == 0 else False,
                                     name='Missings',
                                     fillcolor=missing_color_str))

    if 'original_outlier' in df_concat.columns:
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.outlier_connection,
                                 mode='lines', name='Outlier Connection', showlegend=False,
                                 line={'color': prog_color.loc[4, 'red']}))
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.original_outlier,
                                 mode='markers+lines', name=plot_labels['original_outlier'],
                                 line={'color': prog_color.loc[0, 'red']}, marker={'symbol': 'circle'}))
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.replace_outlier,
                                 mode='markers+lines', name=plot_labels['replace_value'],
                                 line={'color': prog_color.loc[0, 'green']}, marker={'symbol': 'circle'}))

    if 'level_shift' in df_concat.columns:
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.level_shift,
                                 mode='lines', name='Levels', line={'color': prog_color.loc[0, 'gold']}))

    for idx, time_frame in enumerate(plot_few_observations):
        few_obs_color = mpl.colors.to_rgba(prog_color.loc[1, 'greyblue'], alpha=0.3)
        few_obs_color_str = f"rgba({few_obs_color[0]}, {few_obs_color[1]}, {few_obs_color[2]}, {few_obs_color[3]})"
        few_obs_dates = df_concat.date[(df_concat.date >= time_frame[0]) & (df_concat.date <= time_frame[1])]

        fig.add_trace(go.Scatter(x=few_obs_dates, y=[df_concat.actuals.max()] * len(few_obs_dates),
                                 line_color=transparent_rgba,
                                 mode='lines',
                                 hoverinfo='skip',
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=few_obs_dates,
                                 y=[df_concat.actuals.min()] * len(few_obs_dates),
                                 line_color=transparent_rgba,
                                 legendgroup='1',
                                 showlegend=True if idx == 0 else False,
                                 mode='lines', fill='tonexty', name=plot_labels['few_observations'],
                                 fillcolor=few_obs_color_str))

    covariate_column = [col for col in df_concat if col.startswith('covariate_lag')]
    if len(covariate_column) > 0:
        for idx, cov in enumerate(covariate_column):
            fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat[cov],
                                     connectgaps=False,
                                     yaxis=f'y{idx+2}',
                                     mode='lines+markers', name=str(cov).replace('covariate_lag_', ''),
                                     marker={'color': cov_column_color[idx % len(cov_column_color)], 'size': 2},
                                     line={'color': cov_column_color[idx % len(cov_column_color)]}))

        count_y_axis = len(covariate_column) + 1
        fig.update_layout(create_multiple_yaxis(count_y_axis))

    fig.update_layout(
        showlegend=True,
        legend={'x': 1.05, 'y': 1, 'traceorder': 'normal', 'orientation': 'v'}
    )

    fig.show()


def _prepare_characteristics(result: ForecastResult,
                             prepared_actuals: pd.DataFrame) -> TimeSeriesCharacteristics:
    """Returns a copy of the characteristics and removes all outliers, missing values and
    change points that are not within the time frame of the actuals. Usually happens when
    plot_last_x_data_points_only is used.

    Parameters
    ----------
    result
        Forecasting results of a single time series and model.
    prepared_actuals
        Dataframe containing the actuals that are reduced to the last x data points.


    Returns
    -------
    Timeseries characteristics with potentially removed values.
    """
    ts_characteristics = result.ts_characteristics.model_copy(deep=True)

    # Get the earliest date from prepared_actuals
    min_valid_date = min(d.date() for d in prepared_actuals.date)

    if ts_characteristics.outliers:
        ts_characteristics.outliers = [
            x for x in ts_characteristics.outliers
            if x.time_stamp_utc.date() >= min_valid_date
        ]

    if ts_characteristics.change_points:
        ts_characteristics.change_points = [
            x for x in ts_characteristics.change_points
            if x.time_stamp_utc.date() >= min_valid_date
        ]

    if ts_characteristics.missing_values:
        ts_characteristics.missing_values = [
            x for x in ts_characteristics.missing_values
            if x.time_stamp_utc.date() >= min_valid_date
        ]

    return ts_characteristics


def plot_forecast(result: ForecastResult,
                  plot_last_x_data_points_only: Optional[int] = None,
                  model_names: Optional[list[str]] = None,
                  ranks: Optional[list[int]] = [1],
                  plot_prediction_intervals: bool = True,
                  plot_outliers: bool = False,
                  plot_change_points: bool = False,
                  plot_replaced_missings: bool = False,
                  plot_covariates: bool = False,
                  as_interactive: bool = False) -> None:
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
    plot_prediction_intervals
        Shows prediction intervals.
    plot_outliers
        Shows outlieres and replacement values.
    plot_change_points
        Shows change point like level shifts and few observations.
    plot_replaced_missings
        Shows replaced missing values.
    plot_covariates
        Shows the covariates that where used in the model.
    as_interactive
        Plots the data in an interactive plot or as static image.
    """

    df_ac = _prepare_actuals(actuals=result.input.actuals,
                             plot_last_x_data_points_only=plot_last_x_data_points_only)
    characteristics = _prepare_characteristics(result, df_ac)
    name = result.input.actuals.name
    plot_models = filter_models(result.models, ranks, model_names)

    plot_few_observations = []
    if plot_change_points:
        change_points = characteristics.change_points or []
        few_observations = [copy.deepcopy(x) for x in change_points if x.change_point_type.startswith('FEW_OBS')]

        plot_few_observations = _calculate_few_observation_borders(df_ac.date.tolist(), few_observations)

        level_shifts = [x for x in change_points if x.change_point_type == 'LEVEL_SHIFT']
        df_ac = _add_level_shifts(df_ac, level_shifts)

    if plot_outliers:
        outliers = characteristics.outliers or []
        df_ac = _add_outliers(df_ac, outliers, result.changed_values)

    df_ac = _add_changed_start_date(df_ac, result.changed_start_date)

    missing_borders = []
    if plot_replaced_missings:
        df_ac = _add_replaced_missings(df_ac, result.changed_values)
        if 'replaced_missing' in df_ac.columns:
            missing_borders = _calculate_replaced_missing_borders(df_ac)

    for model in plot_models:
        assert model.model_selection.ranking, 'No ranking, plotting not possible.'
        df_concat = _add_forecast(df_ac, model)
        title = f'Forecast for {name}'
        subtitle = f'using {model.model_name} (Rank {model.model_selection.ranking.rank_position})'

        if plot_covariates and len(model.covariates) > 0:
            df_concat = _add_covariates(df_concat, model.covariates, result.input.covariates, df_concat.date.max())

        if as_interactive:
            _create_interactive_forecast_plot(title=title,
                                              subtitle=subtitle,
                                              df_concat=df_concat,
                                              missing_borders=missing_borders,
                                              plot_prediction_intervals=plot_prediction_intervals,
                                              plot_few_observations=plot_few_observations)
        else:
            _create_static_forecast_plot(title=title,
                                         subtitle=subtitle,
                                         df_concat=df_concat,
                                         plot_prediction_intervals=plot_prediction_intervals,
                                         plot_few_observations=plot_few_observations)


def _create_interactive_backtesting_plot(title: str,
                                         subtitle: str,
                                         df_concat: pd.DataFrame,
                                         df_bt: pd.DataFrame,
                                         missing_borders: list[tuple[datetime.datetime, datetime.datetime]],
                                         plot_prediction_intervals: bool,
                                         plot_few_observations:  list[tuple[datetime.datetime, datetime.datetime]]) -> None:
    """Creates a interactive plot for the backtesting results.

    Parameters
    ----------
    title
        Title of the plot.
    subtitle
        Subtitle of the plot.
    df_concat
        DataFrame containing actuals and various preprocessing results.
    df_bt
        DataFrame containing the backtesting forecasts.
    missing_borders
        Information about missing time frames.
    plot_prediction_intervals
        Shows prediction intervals.
    plot_few_observations
        Information about few observation time frames.
    """
    fig = go.Figure()

    fig.update_layout(
        plot_bgcolor='white',
        title=f'{title}<br><sup>{subtitle}</sup>',
        title_x=0.5,
        title_font={'size': 16},
        xaxis_title='Date',
        yaxis_title='Value'
    )

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )

    fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.actuals,
                             connectgaps=False,
                             mode='lines+markers', name=plot_labels['time_series'],
                             marker={'color': prog_color.loc[0, 'darkblue'], 'size': 2},
                             line={'color': prog_color.loc[0, 'darkblue']}))

    if 'removed_actuals' in df_concat.columns:
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.removed_actuals,
                                 connectgaps=False,
                                 mode='lines', name=plot_labels['removed_values'], line={'color': prog_color.loc[0, 'greyblue']}))

    if plot_prediction_intervals and not any(v is None for v in df_bt.lower):
        fig.add_trace(go.Scatter(x=df_bt.date, y=df_bt.upper,
                                 line_color=transparent_rgba,
                                 mode='lines',
                                 hoverinfo='skip',
                                 showlegend=False))
        pi_color = mpl.colors.to_rgba(prog_color.loc[2, "cyan"], alpha=0.3)
        fig.add_trace(go.Scatter(x=df_bt.date, y=df_bt.lower,
                                 line_color=transparent_rgba,
                                 mode='lines', fill='tonexty', name=plot_labels['pi'],
                                 fillcolor=f"rgba({pi_color[0]}, {pi_color[1]}, {pi_color[2]}, {pi_color[3]})"))

    fig.add_trace(go.Scatter(x=df_bt.date, y=df_bt.fc,
                             mode='lines', name='Forecast', line={'color': prog_color.loc[0, 'cyan']}))

    if 'replaced_missing' in df_concat.columns:
        missing_color = mpl.colors.to_rgba(prog_color.loc[2, 'red'], alpha=0.3)
        missing_color_str = f"rgba({missing_color[0]}, {missing_color[1]}, {missing_color[2]}, {missing_color[3]})"
        for idx, time_frame in enumerate(missing_borders):
            missing_dates = df_concat.date[(df_concat.date >= time_frame[0]) & (df_concat.date <= time_frame[1])]
            fig.add_trace(go.Scatter(x=missing_dates, y=[df_concat.actuals.max()] * len(missing_dates),
                                     line_color=transparent_rgba,
                                     mode='lines',
                                     hoverinfo='skip',
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=missing_dates, y=[df_concat.actuals.min()] * len(missing_dates),
                                     mode='lines', fill='tonexty',
                                     line_color=transparent_rgba,
                                     legendgroup='2',
                                     showlegend=True if idx == 0 else False,
                                     name='Missings',
                                     fillcolor=missing_color_str))

    if 'original_outlier' in df_concat.columns:
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.outlier_connection,
                                 mode='lines', name='Outlier Connection', showlegend=False,
                                 line={'color': prog_color.loc[4, 'red']}))
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.original_outlier,
                                 mode='markers+lines', name=plot_labels['original_outlier'],
                                 line={'color': prog_color.loc[0, 'red']}, marker={'symbol': 'circle'}))
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.replace_outlier,
                                 mode='markers+lines', name=plot_labels['replace_value'],
                                 line={'color': prog_color.loc[0, 'green']}, marker={'symbol': 'circle'}))

    if 'level_shift' in df_concat.columns:
        fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat.level_shift,
                                 mode='lines', name='Levels', line={'color': prog_color.loc[0, 'gold']}))

    for idx, time_frame in enumerate(plot_few_observations):
        few_obs_color = mpl.colors.to_rgba(prog_color.loc[1, 'greyblue'], alpha=0.3)
        few_obs_color_str = f"rgba({few_obs_color[0]}, {few_obs_color[1]}, {few_obs_color[2]}, {few_obs_color[3]})"
        few_obs_dates = df_concat.date[(df_concat.date >= time_frame[0]) & (df_concat.date <= time_frame[1])]

        fig.add_trace(go.Scatter(x=few_obs_dates, y=[df_concat.actuals.max()] * len(few_obs_dates),
                                 line_color=transparent_rgba,
                                 mode='lines',
                                 hoverinfo='skip',
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=few_obs_dates,
                                 y=[df_concat.actuals.min()] * len(few_obs_dates),
                                 line_color=transparent_rgba,
                                 legendgroup='1',
                                 showlegend=True if idx == 0 else False,
                                 mode='lines', fill='tonexty', name=plot_labels['few_observations'],
                                 fillcolor=few_obs_color_str))

    covariate_column = [col for col in df_concat if col.startswith('covariate_lag')]
    if len(covariate_column) > 0:
        for idx, cov in enumerate(covariate_column):
            fig.add_trace(go.Scatter(x=df_concat.date, y=df_concat[cov],
                                     connectgaps=False,
                                     yaxis=f'y{idx+2}',
                                     mode='lines+markers', name=str(cov).replace('covariate_lag_', ''),
                                     marker={'color': cov_column_color[idx % len(cov_column_color)], 'size': 2},
                                     line={'color': cov_column_color[idx % len(cov_column_color)]}))

        count_y_axis = len(covariate_column) + 1
        fig.update_layout(create_multiple_yaxis(count_y_axis))

    fig.update_layout(
        showlegend=True,
        legend={'x': 1.05, 'y': 1, 'traceorder': 'normal', 'orientation': 'v'}
    )

    fig.show()


def _create_static_backtesting_plot(title: str,
                                    subtitle: str,
                                    df_concat: pd.DataFrame,
                                    df_bt: pd.DataFrame,
                                    plot_prediction_intervals: bool,
                                    plot_few_observations:  list[tuple[datetime.datetime, datetime.datetime]]) -> None:
    """Creates a static plot for the backtesting results.

    Parameters
    ----------
    title
        Title of the plot.
    subtitle
        Subtitle of the plot.
    df_concat
        DataFrame containing actuals and various preprocessing results.
    df_bt
        DataFrame containing the backtesting forecasts.
    plot_prediction_intervals
        Shows prediction intervals.
    plot_few_observations
        Information about few observation time frames.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    fig.suptitle(title, fontsize=16)
    ax.set_title(subtitle)

    ax.plot(df_concat.date, df_concat.actuals,
            marker='.', markersize=1,
            color=prog_color.loc[0, "darkblue"],  label=plot_labels['time_series'])
    ax.plot(df_bt.date, df_bt.fc, color=prog_color.loc[0, "cyan"], label='Forecast')

    if 'removed_actuals' in df_concat.columns:
        ax.plot(df_concat.date, df_concat.removed_actuals,
                color=prog_color.loc[3, 'greyblue'], label=plot_labels['removed_values'])

    if 'replaced_missing' in df_concat.columns:
        ax.fill_between(df_concat.date, df_concat.actuals.min(), df_concat.actuals.max(),
                        where=df_concat.replaced_missing.notnull(),
                        color=prog_color.loc[2, 'red'], alpha=0.30, label='Missings')

    if 'original_outlier' in df_concat.columns:
        ax.plot(df_concat.date, df_concat.original_outlier, 'o-',
                color=prog_color.loc[0, 'red'], label=plot_labels['original_outlier'])
        ax.plot(df_concat.date, df_concat.replace_outlier, 'o-',
                color=prog_color.loc[0, 'green'], label=plot_labels['replace_value'])
        ax.plot(df_concat.date, df_concat.outlier_connection, '-',
                color=prog_color.loc[4, 'red'], zorder=1)

    if 'level_shift' in df_concat.columns:
        ax.plot(df_concat.date, df_concat.level_shift, color=prog_color.loc[0, 'gold'], label='Levels')

    for idx, time_frame in enumerate(plot_few_observations):
        ax.fill_between(df_concat.date, df_concat.actuals.min(), df_concat.actuals.max(),
                        where=(df_concat.date >= time_frame[0]) & (df_concat.date <= time_frame[1]),
                        color=prog_color.loc[1, 'greyblue'], alpha=0.30,
                        label=plot_labels['few_observations'] if idx == 0 else None)

    if plot_prediction_intervals and None not in df_bt.lower and None not in df_bt.upper:
        ax.fill_between(df_bt.date, df_bt.lower, df_bt.upper,
                        color=prog_color.loc[2, "cyan"], alpha=0.30, label=plot_labels['pi'])

    covariate_column = [col for col in df_concat if col.startswith('covariate_lag')]
    if len(covariate_column) > 0:
        _add_covariates_to_static_plot(ax, covariate_column, df_concat)
    else:
        ax.legend(loc=legend_position['loc'], bbox_to_anchor=(1, 1))

    ax.set_frame_on(False)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.show()


def plot_backtesting(result: ForecastResult,
                     iteration: int = 1,
                     plot_last_x_data_points_only: Optional[int] = None,
                     model_names: Optional[list[str]] = None,
                     ranks: Optional[list[int]] = [1],
                     plot_prediction_intervals: bool = True,
                     plot_outliers: bool = False,
                     plot_change_points: bool = False,
                     plot_replaced_missings: bool = False,
                     plot_covariates: bool = False,
                     as_interactive: bool = False) -> None:
    """Plots actuals and backtesting results from a single time series.

    Parameters
    ----------
    result
        Forecasting and backtesting results of a single time series and model.
    iteration
        Iteration of the backtesting forecast.
    plot_last_x_data_points_only
        Number of data points of the actuals that should be shown in the plot.
    model_names
        Names of the models to plot.
    ranks
        Ranks of the models to plot.
    plot_prediction_intervals
        Shows prediction intervals.
    plot_outliers
        Shows outlieres and replacement values.
    plot_change_points
        Shows change point like level shifts and few observations.
    plot_replaced_missings
        Shows replaced missing values.
    plot_covariates
        Shows the covariates that where used in the model.
    as_interactive
        Plots the data in an interactive plot or as static image.
    """
    plot_models = filter_models(result.models, ranks, model_names)
    df_ac = _prepare_actuals(result.input.actuals, plot_last_x_data_points_only)
    characteristics = _prepare_characteristics(result, df_ac)

    plot_few_observations = []
    if plot_change_points:

        change_points = characteristics.change_points or []
        few_observations = [copy.deepcopy(x) for x in change_points if x.change_point_type.startswith('FEW_OBS')]

        plot_few_observations = _calculate_few_observation_borders(df_ac.date.tolist(), few_observations)

        level_shifts = [x for x in change_points if x.change_point_type == 'LEVEL_SHIFT']
        df_ac = _add_level_shifts(df_ac, level_shifts)

    if plot_outliers:
        outliers = characteristics.outliers or []
        df_ac = _add_outliers(df_ac, outliers, result.changed_values)

    df_ac = _add_changed_start_date(df_ac, result.changed_start_date)

    missing_borders = []
    if plot_replaced_missings:
        df_ac = _add_replaced_missings(df_ac, result.changed_values)
        if 'replaced_missing' in df_ac.columns:
            missing_borders = _calculate_replaced_missing_borders(df_ac)

    for model in plot_models:

        assert model.model_selection.backtesting, \
            'Cannot create backtesting plot, at least one model has no backtesting results available.'
        assert model.model_selection.ranking, 'No ranking, plotting not possible.'
        df_bt = _prepare_backtesting(model, iteration)

        title = f'Backtesting of {result.input.actuals.name} - Iteration: {iteration}'
        subtitle = f'using {model.model_name} (Rank {model.model_selection.ranking.rank_position})'

        if plot_covariates and len(model.covariates) > 0:
            df_concat = _add_covariates(df_ac, model.covariates, result.input.covariates,
                                        _calculate_max_covariate_date(result.input.actuals.granularity, df_ac.date.max()))
        else:
            df_concat = df_ac

        if as_interactive:
            _create_interactive_backtesting_plot(title=title,
                                                 subtitle=subtitle,
                                                 df_concat=df_concat, df_bt=df_bt,
                                                 missing_borders=missing_borders,
                                                 plot_prediction_intervals=plot_prediction_intervals,
                                                 plot_few_observations=plot_few_observations)
        else:
            _create_static_backtesting_plot(title=title,
                                            subtitle=subtitle,
                                            df_concat=df_concat,
                                            df_bt=df_bt,
                                            plot_prediction_intervals=plot_prediction_intervals,
                                            plot_few_observations=plot_few_observations)
