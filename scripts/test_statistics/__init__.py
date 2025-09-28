from .lwps import LWPSStatistic
from .fdi import FDIStatistic

AVAILABLE_STATISTICS = {
    'lwps': LWPSStatistic,
    'fdi': FDIStatistic,
}

def create_statistic(name: str, config: dict):
    if name not in AVAILABLE_STATISTICS:
        raise ValueError(f'unknown statistic: {name}')
    return AVAILABLE_STATISTICS[name](config)
