from .fdi import FDIStatistic
from .ifs import IFSStatistic
from .lwps import LWPSStatistic
from .ocf import OCFStatistic
from .pfe import PFEStatistic

AVAILABLE_STATISTICS = {
    'lwps': LWPSStatistic,
    'fdi': FDIStatistic,
    'ifs': IFSStatistic,
    'pfe': PFEStatistic,
    'ocf': OCFStatistic,
}


def create_statistic(name: str, config: dict):
    if name not in AVAILABLE_STATISTICS:
        raise ValueError(f'unknown statistic: {name}')
    return AVAILABLE_STATISTICS[name](config)
