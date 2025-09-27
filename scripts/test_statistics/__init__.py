from .lwps import LWPSStatistic

AVAILABLE_STATISTICS = {
    'lwps': LWPSStatistic,
}

def create_statistic(name: str, config: dict):
    if name not in AVAILABLE_STATISTICS:
        raise ValueError(f'unknown statistic: {name}')
    return AVAILABLE_STATISTICS[name](config)
