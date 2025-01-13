import logging

from futureexpert.__about__ import __version__
from futureexpert.checkin import DataDefinition, FileSpecification, FilterSettings, TsCreationConfig
from futureexpert.expert_client import ExpertClient
from futureexpert.forecast import ForecastingConfig, MethodSelectionConfig, PreprocessingConfig, ReportConfig
from futureexpert.matcher import ActualsCovsConfiguration, LagSelectionConfig, MatcherConfig
from futureexpert.shared_models import MAX_TS_LEN_CONFIG, CovariateRef

__all__ = [
    'DataDefinition',
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
    'CovariateRef'
]


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx").propagate = False
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore").propagate = False
