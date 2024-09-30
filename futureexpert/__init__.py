import logging

from futureexpert.__about__ import __version__
from futureexpert.checkin import (DataDefinition,
                                  DateColumn,
                                  FileSpecification,
                                  FilterSettings,
                                  GroupColumn,
                                  NewValue,
                                  TsCreationConfig,
                                  ValueColumn)
from futureexpert.expert_client import ExpertClient
from futureexpert.forecast import (MAX_TS_LEN_CONFIG,
                                   ForecastingConfig,
                                   MethodSelectionConfig,
                                   PreprocessingConfig,
                                   ReportConfig)
from futureexpert.matcher import ActualsCovsConfiguration, MatcherConfig
from futureexpert.shared_models import CovariateRef

__all__ = [
    'DataDefinition',
    'DateColumn',
    'ExpertClient',
    'FileSpecification',
    'FilterSettings',
    'ForecastingConfig',
    'GroupColumn',
    'MatcherConfig',
    'MethodSelectionConfig',
    'NewValue',
    'PreprocessingConfig',
    'ReportConfig',
    'TsCreationConfig',
    'ValueColumn',
    'MAX_TS_LEN_CONFIG',
    'ActualsCovsConfiguration',
    'CovariateRef'
]


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx").propagate = False
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore").propagate = False
