import math
import time
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

import futureexpert.checkin as checkin
from futureexpert import (DataDefinition,
                          ExpertClient,
                          FileSpecification,
                          ForecastingConfig,
                          MethodSelectionConfig,
                          PreprocessingConfig,
                          ReconciliationConfig,
                          ReconciliationMethod,
                          ReportConfig,
                          TsCreationConfig)
from futureexpert.forecast_consistency import (MakeForecastConsistentConfiguration,
                                               MakeForecastConsistentDataSelection,
                                               ReconciliationConfig,
                                               ReconciliationMethod)

load_dotenv()


def test_expert_client___given_save_hierarchy___returns_all_hierarchy() -> None:
    # Arrange
    client = ExpertClient()
    raw_data_source = Path("tests") / 'unit_tests/test_data/demo_demand_planning_data.csv'
    data_definition = DataDefinition(date_column=checkin.DateColumn(name='DATE', format='%d.%m.%Y', name_new='Date'),
                                     value_columns=[checkin.ValueColumn(
                                         name='DEMAND', name_new='Demand')],
                                     group_columns=[checkin.GroupColumn(name="CUSTOMER", name_new='Customer'),
                                                    checkin.GroupColumn(name="MATERIAL", name_new="Material"),
                                                    checkin.GroupColumn(name="REGION", name_new="Region")])

    config_ts_creation = TsCreationConfig(time_granularity='monthly',
                                          start_date="2007-10-01",
                                          end_date="2024-06-01",
                                          value_columns_to_save=[
                                              'Demand'],
                                          grouping_level=[
                                              'Region', 'Customer'],
                                          save_hierarchy=True,
                                          missing_value_handler='setToZero')
    # Act
    check_in_version_id = client.check_in_time_series(raw_data_source=raw_data_source,
                                                      file_specification=FileSpecification(delimiter=';', decimal='.'),
                                                      data_definition=data_definition,
                                                      config_ts_creation=config_ts_creation)

    # Assert
    check_in_result = client.get_time_series(version_id=check_in_version_id)
    hierarchy_levels = list(set([tuple(ts.grouping.keys()) for ts in check_in_result.time_series]))
    assert len(check_in_result.time_series) == 9
    assert len(hierarchy_levels) == 3
    assert () in hierarchy_levels
    assert ('Region',) in hierarchy_levels
    assert ('Region', 'Customer') in hierarchy_levels or ('Customer', 'Region') in hierarchy_levels


def test_expert_client___given_different_methods_per_hierachical_level___uses_correct_methods() -> None:
    # Arrange
    fc_methods = ['MA(3)']
    fc_methods_per_level = {2: ['FoundationModel']}
    num_levels = 3
    expected_fc_methods_per_level = {
        level: fc_methods_per_level.get(level, fc_methods)
        for level in range(num_levels)
    }
    client = ExpertClient()
    raw_data_source = Path("tests") / 'unit_tests/test_data/demo_demand_planning_data.csv'
    data_definition = DataDefinition(date_column=checkin.DateColumn(name='DATE', format='%d.%m.%Y', name_new='Date'),
                                     value_columns=[checkin.ValueColumn(
                                         name='DEMAND', name_new='Demand')],
                                     group_columns=[checkin.GroupColumn(name="CUSTOMER", name_new='Customer'),
                                                    checkin.GroupColumn(name="MATERIAL", name_new="Material"),
                                                    checkin.GroupColumn(name="REGION", name_new="Region")])

    config_ts_creation = TsCreationConfig(time_granularity='monthly',
                                          start_date="2007-10-01",
                                          end_date="2024-06-01",
                                          value_columns_to_save=[
                                              'Demand'],
                                          grouping_level=[
                                              'Region', 'Customer'],
                                          save_hierarchy=True,
                                          missing_value_handler='setToZero')

    check_in_version_id = client.check_in_time_series(raw_data_source=raw_data_source,
                                                      file_specification=FileSpecification(delimiter=';', decimal='.'),
                                                      data_definition=data_definition,
                                                      config_ts_creation=config_ts_creation)
    fc_report_config = ReportConfig(title='Unit Test: Method Selection per hierarchical Level',
                                    preprocessing=PreprocessingConfig(),
                                    forecasting=ForecastingConfig(fc_horizon=1,
                                                                  use_ensemble=False),
                                    method_selection=MethodSelectionConfig(
                                        number_iterations=1,
                                        forecasting_methods=fc_methods,
                                        forecasting_methods_per_hierarchy_level=fc_methods_per_level
                                    ),
                                    max_ts_len=72)
    # Act
    forecast_identifier = client.start_forecast(version=check_in_version_id, config=fc_report_config)
    while not (client.get_report_status(id=forecast_identifier)).is_finished:
        time.sleep(20)

    # Assert
    results = client.get_fc_results(id=forecast_identifier, include_backtesting=True, include_k_best_models=10)
    for result in results.forecast_results:
        hierarchical_depth = len(result.input.actuals.name.split('-')) - 1
        model_names = [model.model_name for model in result.models]
        assert model_names == expected_fc_methods_per_level[hierarchical_depth]


def test_expert_client___given_forecast_minimums___enforces_minimums() -> None:
    # Arrange
    client = ExpertClient()
    minimum_sale = 110
    target_series_name = 'Sales-Europe-Italy'
    project_root = Path(__file__).parent.parent.parent

    csv_path = project_root / 'use_cases' / 'sales_forecasting' / 'demo_sales_data.csv'
    df = pd.read_csv(csv_path, sep=';', decimal='.')

    actuals_version_id = client.check_in_time_series(
        raw_data_source=df,
        data_definition=DataDefinition(
            date_column=checkin.DateColumn(name='Date', format='%d.%m.%Y'),
            value_columns=[checkin.ValueColumn(name='Sales')],
            group_columns=[
                checkin.GroupColumn(name='Region'),
                checkin.GroupColumn(name='Country')
            ],
            remove_columns=[3]
        ),
        config_ts_creation=TsCreationConfig(
            time_granularity='monthly',
            start_date='2016-03-01',
            value_columns_to_save=['Sales'],
            grouping_level=['Region', 'Country'],
            save_hierarchy=True,
            filter=[
                checkin.FilterSettings(type='exclusion', variable='Region', items=['Global']),
                checkin.FilterSettings(type='exclusion', variable='Country', items=['Global'])
            ],
            missing_value_handler='keepNaN'
        )
    )

    min_sales_df = pd.DataFrame({
        'Date': ['01.06.2024'],
        'Region': ['Europe'],
        'Country': ['Italy'],
        'MinSales': [minimum_sale]
    })

    min_sales_version_id = client.check_in_time_series(
        raw_data_source=min_sales_df,
        data_definition=DataDefinition(
            date_column=checkin.DateColumn(name='Date', format='%d.%m.%Y'),
            value_columns=[checkin.ValueColumn(name='MinSales')],
            group_columns=[
                checkin.GroupColumn(name='Region'),
                checkin.GroupColumn(name='Country')
            ]
        ),
        config_ts_creation=TsCreationConfig(
            time_granularity='monthly',
            start_date='2016-03-01',
            value_columns_to_save=['MinSales'],
            grouping_level=['Region', 'Country'],
            save_hierarchy=False,
            filter=[
                checkin.FilterSettings(type='exclusion', variable='Region', items=['Global']),
                checkin.FilterSettings(type='exclusion', variable='Country', items=['Global'])
            ],
            missing_value_handler='keepNaN'
        )
    )

    fc_methods = ['LinearRegression', 'Naive', 'MA(granularity)', 'MA(3)', 'MA(season lag)', 'FoundationModel']
    fc_methods_per_level = {2: ['FoundationModel']}

    fc_report_config = ReportConfig(
        title='Monthly Sales Forecast on Multiple Hierarchical Levels',
        preprocessing=PreprocessingConfig(
            detect_outliers=True,
            replace_outliers=True,
            detect_changepoints=True
        ),
        forecasting=ForecastingConfig(
            fc_horizon=12,
            lower_bound=0,
            use_ensemble=False,
            confidence_level=0.90
        ),
        method_selection=MethodSelectionConfig(
            number_iterations=12,
            refit=True,
            step_weights={1: 1., 2: 1., 3: 1., 4: 0.5, 5: 0.5, 6: 0.5},
            forecasting_methods=fc_methods,
            forecasting_methods_per_hierarchy_level=fc_methods_per_level
        ),
        max_ts_len=72
    )

    forecast_identifier = client.start_forecast(version=actuals_version_id, config=fc_report_config)
    print(f"Forecast report ID: {forecast_identifier}")

    # Wait for forecast completion
    while not (current_status := client.get_report_status(id=forecast_identifier)).is_finished:
        time.sleep(20)

    assert current_status.results.error == 0, \
        'At least one forecast failed. All forecasts must succeed to complete this test.'

    hcfc_config = ReconciliationConfig(
        method=ReconciliationMethod.TOP_DOWN_AVERAGE_FORECAST_PROPORTION,
        fallback_methods=[],
        round_forecast_to_package_size=True,
        enforce_forecast_minimum_constraint=True
    )

    reconciliation_config = MakeForecastConsistentConfiguration(
        data_selection=MakeForecastConsistentDataSelection(
            version=actuals_version_id,
            fc_report_id=forecast_identifier.report_id,
            forecast_minimum_version=min_sales_version_id
        ),
        report_note='Test min forecast with package size rounding',
        reconciliation=hcfc_config
    )

    # Act
    consistent_report = client.start_making_forecast_consistent(reconciliation_config)
    print(f"HCFC report ID: {consistent_report}")

    while not (hcfc_status := client.get_report_status(id=consistent_report)).is_finished:
        time.sleep(20)

    # Assert
    assert hcfc_status.results.error == 0, \
        f'HCFC reconciliation failed with {hcfc_status.results.error} errors'

    results = client.get_fc_results(id=consistent_report, include_backtesting=True, include_k_best_models=10)

    italy_forecast = None
    for forecast_result in results.forecast_results:
        if forecast_result.input.actuals.name == target_series_name:
            italy_forecast = forecast_result
            break

    assert italy_forecast is not None, \
        f"Could not find forecast for '{target_series_name}' in results. " \
        f"Available series: {[fr.input.actuals.name for fr in results.forecast_results]}"

    reconciled_model = italy_forecast.models[0]
    first_forecast_value = reconciled_model.forecasts[0].point_forecast_value

    assert first_forecast_value >= minimum_sale, \
        f"Forecast value {first_forecast_value} is below minimum {minimum_sale}"

    assert first_forecast_value.is_integer(), \
        f"Forecast value {first_forecast_value} is not an integer (package size rounding should enforce this)"

    assert reconciled_model.model_name == 'FoundationModel - top down average forecast proportion', \
        f"Expected reconciled model, got: {reconciled_model.model_name}"
