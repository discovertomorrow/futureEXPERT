import logging

from futureexpert.__about__ import __version__
from futureexpert.associator import (AssociatorConfig,
                                     ClusteringConfiguration,
                                     DataSelection,
                                     TrendDetectionConfiguration)
from futureexpert.checkin import DataDefinition, FileSpecification, FilterSettings, TsCreationConfig
from futureexpert.expert_client import ErrorReason, ExpertClient
from futureexpert.forecast import ForecastingConfig, MethodSelectionConfig, PreprocessingConfig, ReportConfig
from futureexpert.forecast_consistency import (MakeForecastConsistentConfiguration,
                                               MakeForecastConsistentDataSelection,
                                               ReconciliationConfig,
                                               ReconciliationMethod)
from futureexpert.matcher import ActualsCovsConfiguration, LagSelectionConfig, MatcherConfig
from futureexpert.shared_models import MAX_TS_LEN_CONFIG, CovariateRef

__all__ = [
    'DataDefinition',
    'ErrorReason',
    'ExpertClient',
    'FileSpecification',
    'FilterSettings',
    'ForecastingConfig',
    'LagSelectionConfig',
    'MatcherConfig',
    'MethodSelectionConfig',
    'PreprocessingConfig',
    'ReportConfig',
    'TsCreationConfig',
    'MAX_TS_LEN_CONFIG',
    'ActualsCovsConfiguration',
    'CovariateRef',
    'AssociatorConfig',
    'DataSelection',
    'ClusteringConfiguration',
    'TrendDetectionConfiguration',
    'MakeForecastConsistentDataSelection',
    'ReconciliationConfig',
    'ReconciliationMethod',
    'MakeForecastConsistentConfiguration'
]


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx").propagate = False
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore").propagate = False
