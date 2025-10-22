from typing import Optional

from futureexpert.shared_models import MAX_TS_LEN_CONFIG


def snake_to_camel(snake_string: str) -> str:
    """Converts snake case to lower camel case.

    Parameters
    ----------
    snake_string
        string im snake case format.
    """
    title_string = snake_string.title().replace('_', '')
    return title_string[:1].lower() + title_string[1:]


def calculate_max_ts_len(max_ts_len: Optional[int], granularity: str) -> Optional[int]:
    """Calculates the max_ts_len.

    Parameters
    ----------
    max_ts_len
        At most the number of most recent observations is used.
    granularity
        Granularity of the time series.
    """
    max_allowed_len = MAX_TS_LEN_CONFIG.get(granularity, None)
    assert max_allowed_len, 'For the given granularity no max_ts_len configuration exists.'

    if max_ts_len is None:
        return max_allowed_len
    if max_ts_len > max_allowed_len:
        raise ValueError(
            f'Given max_ts_len {max_ts_len} is not allowed for granularity {granularity}. ' +
            'Check the environment variable MAX_TS_LEN_CONFIG for allowed configuration.')
    return max_ts_len
