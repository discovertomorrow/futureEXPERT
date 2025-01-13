"""Contains all the functionality to plot the checked in time series and the forecast and backtesting results."""
import copy
import datetime
from collections import defaultdict
from typing import Hashable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from futureexpert.forecast import ChangedValue, ChangePoint, ForecastResult, Model, Outlier
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

legend_position = {'loc': 'upper left'}

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


def plot_time_series(ts: TimeSeries,
                     covariate: Optional[Covariate] = None,
                     plot_last_x_data_points_only: Optional[int] = None) -> None:
    """Plots actuals from a single time series. Optional a Covariate can be plotted next to it.

    Parameters
    ----------
    ts
        time series data
    covariate
        covariate data
    plot_last_x_data_points_only
        Number of data points of the actuals that should be shown in the plot.
    """

    actual_dates = [fc.time_stamp_utc for fc in ts.values]
    actual_values = [fc.value for fc in ts.values]

    if plot_last_x_data_points_only is not None:
        actual_dates = actual_dates[-plot_last_x_data_points_only:]
        actual_values = actual_values[-plot_last_x_data_points_only:]

    name = ts.name
    df_ac = pd.DataFrame({'date': actual_dates, 'actuals': actual_values})
    df_ac = _fill_missing_values_for_plot(granularity=ts.granularity, df=df_ac)

    if covariate:
        cov_date = [value.time_stamp_utc for value in covariate.ts.values]
        cov_value = [value.value for value in covariate.ts.values]
        df_cov = pd.DataFrame({'date': cov_date, 'covariate': cov_value, 'covariate_lag': cov_value})
        df_ac = pd.merge(df_ac, df_cov, on='date', how='outer', validate='1:1').reset_index(drop=True)
        df_ac = df_ac.sort_values(by='date')
        df_ac.covariate_lag = df_ac.covariate_lag.shift(covariate.lag)

        # remove all covariate values from befor the start of actuals
        min_value = df_ac[df_ac['actuals'].notna()][['date']].min()
        df_ac = df_ac[df_ac['date'] >= min_value[0]]
        df_ac.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    fig.suptitle(name, fontsize=16)
    ax.set_frame_on(False)
    ax.tick_params(axis='both', labelsize=10)

    # plot
    ax.plot(df_ac.date, df_ac.actuals, color=prog_color.loc[0, "darkblue"])
    if covariate:
        ax.set_title(f'with covariate: {covariate.ts.name} and lag {covariate.lag}')
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.set_frame_on(False)
        ax2.tick_params(axis='both', labelsize=10)
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
        ax2.yaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
        ax2.plot(df_ac.date, df_ac.covariate_lag, color=prog_color.loc[0, 'violet'], label=covariate.ts.name)

    # margin
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.show()


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

    granularity_to_pd_alias = {
        'yearly': 'YS',
        'quarterly': 'QS',
        'monthly': 'MS',
        'weekly': 'W',
        'daily': 'D',
        'hourly': 'h',
        'halfhourly': '30min'
    }
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
    matched_few_observations = []
    few_observations.sort(key=lambda x: x.time_stamp_utc)
    for idx, few_observation in enumerate(few_observations):

        left_border = None
        right_border = None

        if few_observation.change_point_type == 'FEW_OBSERVATIONS_RIGHT':

            if idx == len(few_observations)-1:
                right_border = actual_dates[-1]
            else:
                ref_point = next((k for k in few_observations[idx:]
                                  if k.change_point_type == 'FEW_OBSERVATIONS_LEFT'), None)
                if ref_point:
                    right_border = ref_point.time_stamp_utc
            left_border = few_observation.time_stamp_utc

        if few_observation.change_point_type == 'FEW_OBSERVATIONS_LEFT' and idx == 0:
            right_border = few_observation.time_stamp_utc
            left_border = actual_dates[0]

        if left_border and right_border:
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
    if df_outlier.shape[0] > 0:
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


def _add_covariates(df_plot: pd.DataFrame,
                    model_covariates: Sequence[CovariateRef],
                    input_covariates: Sequence[Covariate]) -> pd.DataFrame:
    """Add covariates to the plot data.

    Parameters
    ----------
    df_plot
        Data frame containing dates values of actuals, forecast and various preprocessing information.
    model_covariates
        CovariateRef that were used in the model
    input_covariates
        Complete covariate data that were used in the report.
    """
    for covariate_ref in model_covariates:
        if covariate_ref is None:
            continue
        covariate = next((x for x in input_covariates if x.ts.name == covariate_ref.name))

        cov_date = [value.time_stamp_utc for value in covariate.ts.values]
        cov_value = [value.value for value in covariate.ts.values]

        df_cov = pd.DataFrame({'date': cov_date, 'covariate': cov_value, 'covariate_lag': cov_value})
        df_plot = pd.merge(df_plot, df_cov, on='date', how='outer', validate='1:1').reset_index(drop=True)
        df_plot = df_plot.sort_values(by='date')
        df_plot.covariate_lag = df_plot.covariate_lag.shift(covariate.lag)
        columns_new_name_mapping = {'covariate': 'covariate_' + covariate_ref.name,
                                    'covariate_lag': 'covariate_lag_' + covariate_ref.name + ' and lag ' + str(covariate_ref.lag)}
        df_plot = df_plot.rename(columns=columns_new_name_mapping)

    # remove all covariate values from before the start of actuals
    min_value = df_plot[df_plot['actuals'].notna()][['date']].min()
    df_plot = df_plot[df_plot['date'] >= min_value.iloc[0]]
    df_plot.reset_index(drop=True, inplace=True)

    return df_plot


def _add_covariates_to_plot(ax: mpl.axes.Axes, covariate_column: list[Hashable], df_plot: pd.DataFrame) -> None:
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


def plot_forecast(result: ForecastResult,
                  plot_last_x_data_points_only: Optional[int] = None,
                  model_names: Optional[list[str]] = None,
                  ranks: Optional[list[int]] = [1],
                  plot_prediction_intervals: bool = True,
                  plot_outliers: bool = False,
                  plot_change_points: bool = False,
                  plot_replaced_missings: bool = False,
                  plot_covariates: bool = False) -> None:
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
    """

    actuals = result.input.actuals
    plot_models = filter_models(result.models, ranks, model_names)

    # prepare actual values
    values = actuals.values
    if plot_last_x_data_points_only is not None:
        values = actuals.values[-plot_last_x_data_points_only:]

    actual_dates = [fc.time_stamp_utc for fc in values]
    actual_values = [fc.value for fc in values]
    df_ac = pd.DataFrame({'date': actual_dates, 'actuals': actual_values})
    df_ac = _fill_missing_values_for_plot(granularity=result.input.actuals.granularity, df=df_ac)
    index_last_actual = len(df_ac.index)-1

    plot_few_observations = []
    if plot_change_points:

        change_points = result.ts_characteristics.change_points or []
        few_observations = [copy.deepcopy(x) for x in change_points if x.change_point_type.startswith('FEW_OBS')]

        plot_few_observations = _calculate_few_observation_borders(actual_dates, few_observations)

        level_shifts = [x for x in change_points if x.change_point_type == 'LEVEL_SHIFT']
        df_ac = _add_level_shifts(df_ac, level_shifts)

    if plot_outliers:
        outliers = result.ts_characteristics.outliers or []
        df_ac = _add_outliers(df_ac, outliers, result.changed_values)

    if plot_replaced_missings:
        df_ac = _add_replaced_missings(df_ac, result.changed_values)

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

        df_fc = pd.DataFrame({'date': fc_date, 'fc': fc_value, 'upper': fc_upper_value, 'lower': fc_lower_value})

        df_concat = pd.concat([df_ac, df_fc], axis=0).reset_index(drop=True)

        # connected forecast line with actual line
        df_concat.loc[index_last_actual, 'fc'] = df_concat.loc[index_last_actual, 'actuals']
        df_concat.loc[index_last_actual, 'upper'] = df_concat.loc[index_last_actual, 'actuals']
        df_concat.loc[index_last_actual, 'lower'] = df_concat.loc[index_last_actual, 'actuals']
        df_concat.date = pd.to_datetime(df_concat.date)
        df_concat.sort_values('date', inplace=True)
        df_concat.reset_index(drop=True, inplace=True)

        if plot_covariates and len(model.covariates) > 0:
            df_concat = _add_covariates(df_concat, model.covariates, result.input.covariates)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'Forecast for {name}', fontsize=16)
        ax.set_title(f'using {model_name} (Rank {model_rank})')

        # plot
        ax.plot(df_concat.date, df_concat.actuals, color=prog_color.loc[0, 'darkblue'], label='Time Series')
        ax.plot(df_concat.date, df_concat.fc, color=prog_color.loc[0, 'cyan'], label='Forecast')
        if 'replaced_missing' in df_concat.columns:
            ax.fill_between(df_concat.date, min(df_concat.actuals), max(df_concat.actuals),
                            where=df_concat.replaced_missing.notnull(),
                            color=prog_color.loc[2, 'red'], alpha=0.30, label='Missings')

        if 'original_outlier' in df_concat.columns:
            ax.plot(df_concat.date, df_concat.original_outlier, 'o-',
                    color=prog_color.loc[0, 'red'], label='Original Outlier')
            ax.plot(df_concat.date, df_concat.replace_outlier, 'o-',
                    color=prog_color.loc[0, 'green'], label='Replacement Values')
            ax.plot(df_concat.date, df_concat.outlier_connection, '-',
                    color=prog_color.loc[4, 'red'], zorder=1)

        if 'level_shift' in df_concat.columns:
            ax.plot(df_concat.date, df_concat.level_shift, color=prog_color.loc[0, 'gold'], label='Levels')

        for idx, time_frame in enumerate(plot_few_observations):
            ax.fill_between(df_concat.date, min(df_concat.actuals), max(df_concat.actuals),
                            where=(df_concat.date >= time_frame[0]) & (df_concat.date < time_frame[1]),
                            color=prog_color.loc[1, 'greyblue'], alpha=0.30, label='Few Observations' if idx == 0 else None)

        if plot_prediction_intervals and not any(v is None for v in df_concat.lower):
            ax.fill_between(df_concat.date, df_concat.lower, df_concat.upper,
                            color=prog_color.loc[2, 'cyan'], alpha=0.30, label='Prediction Interval')

        covariate_column = [col for col in df_concat if col.startswith('covariate_lag')]
        if len(covariate_column) > 0:
            _add_covariates_to_plot(ax, covariate_column, df_concat)
        else:
            # legend
            ax.legend(loc=legend_position['loc'], bbox_to_anchor=(1, 1))

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
                     ranks: Optional[list[int]] = [1],
                     plot_prediction_intervals: bool = True,
                     plot_outliers: bool = False,
                     plot_change_points: bool = False,
                     plot_replaced_missings: bool = False,
                     plot_covariates: bool = False) -> None:
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
    """

    actuals = result.input.actuals
    plot_models = filter_models(result.models, ranks, model_names)

    values = actuals.values
    if plot_last_x_data_points_only is not None:
        values = actuals.values[-plot_last_x_data_points_only:]

    actual_dates = [ac.time_stamp_utc for ac in values]
    actual_values = [ac.value for ac in values]

    # if the data has missing values, make sure to explicitly store them as nan. Otherwise the plot function will
    # display interpolated values. To use help function, a temporary df is needed.
    df_ac = pd.DataFrame({'date': actual_dates, 'actuals': actual_values})
    df_ac = _fill_missing_values_for_plot(granularity=result.input.actuals.granularity, df=df_ac)

    plot_few_observations = []
    if plot_change_points:

        change_points = result.ts_characteristics.change_points or []
        few_observations = [copy.deepcopy(x) for x in change_points if x.change_point_type.startswith('FEW_OBS')]

        plot_few_observations = _calculate_few_observation_borders(actual_dates, few_observations)

        level_shifts = [x for x in change_points if x.change_point_type == 'LEVEL_SHIFT']
        df_ac = _add_level_shifts(df_ac, level_shifts)

    if plot_outliers:
        outliers = result.ts_characteristics.outliers or []
        df_ac = _add_outliers(df_ac, outliers, result.changed_values)

    if plot_replaced_missings:
        df_ac = _add_replaced_missings(df_ac, result.changed_values)

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

        if plot_covariates and len(model.covariates) > 0:
            df_ac = _add_covariates(df_ac, model.covariates, result.input.covariates)

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        fig.suptitle(f'Backtesting of {actuals.name} - Iteration: {iteration}', fontsize=16)
        ax.set_title(f'using {model_name} (Rank {model_rank})')

        # plot
        ax.plot(df_ac.date, df_ac.actuals, color=prog_color.loc[0, "darkblue"],  label='Time Series')
        ax.plot(backtesting_dates, backtesting_fc, color=prog_color.loc[0, "cyan"], label='Forecast')

        if 'replaced_missing' in df_ac.columns:
            ax.fill_between(df_ac.date, min(df_ac.actuals), max(df_ac.actuals),
                            where=df_ac.replaced_missing.notnull(),
                            color=prog_color.loc[2, 'red'], alpha=0.30, label='Missings')

        if 'original_outlier' in df_ac.columns:
            ax.plot(df_ac.date, df_ac.original_outlier, 'o-',
                    color=prog_color.loc[0, 'red'], label='Original Outlier')
            ax.plot(df_ac.date, df_ac.replace_outlier, 'o-',
                    color=prog_color.loc[0, 'green'], label='Replacement Values')
            ax.plot(df_ac.date, df_ac.outlier_connection, '-',
                    color=prog_color.loc[4, 'red'], zorder=1)

        if 'level_shift' in df_ac.columns:
            ax.plot(df_ac.date, df_ac.level_shift, color=prog_color.loc[0, 'gold'], label='Levels')

        for idx, time_frame in enumerate(plot_few_observations):
            ax.fill_between(df_ac.date, min(df_ac.actuals), max(df_ac.actuals),
                            where=(df_ac.date >= time_frame[0]) & (df_ac.date < time_frame[1]),
                            color=prog_color.loc[1, 'greyblue'], alpha=0.30, label='Few Observations' if idx == 0 else None)

        if plot_prediction_intervals and None not in backtesting_lower and None not in backtesting_upper:
            ax.fill_between(backtesting_dates, backtesting_lower, backtesting_upper,
                            color=prog_color.loc[2, "cyan"], alpha=0.30, label='Prediction Interval')

        covariate_column = [col for col in df_ac if col.startswith('covariate_lag')]
        if len(covariate_column) > 0:
            _add_covariates_to_plot(ax, covariate_column, df_ac)
        else:
            # legend
            ax.legend(loc=legend_position['loc'], bbox_to_anchor=(1, 1))

        # style
        ax.set_frame_on(False)

        # margin
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.show()
