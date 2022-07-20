"""functions related to heart rate analysis """

from ast import Return
from pandas import Series
from typing import List
from power import normalised_power
import math


def aerobic_decoupling(Power: Series, HR: Series) -> float:
    """calculataes a

    Args:
        Power (Series): power
        HR (Series): heart rate

    Returns:
        float: aerobic decoupling, a percentage
    """

    mid_point = len(HR) / 2
    np_start, np_end = normalised_power(Power[:mid_point]), normalised_power(
        Power[mid_point:]
    )
    hr_start, hr_end = HR[:mid_point].mean(), HR[mid_point:].mean()

    return round(100 * (1 - ((np_end / hr_end) / (np_start / hr_start))), 2)


def strava_suffer_score(HR: Series, zones: List) -> int:

    """calculates the strava suffer score

     Args:
        HR (Series): heart rate
        zones (list): 4 heart rate zone cut offs, allowing the definition z1, z2, z3, z4, z5

    Raises:
        ValueError: assumes 4 zone cut offs are provided, if not a valueerror is raised

    Returns:
        int: suffer score from strava
    """

    if len(zones) != 4:
        raise ValueError(f"suffer score expects 4 zone cut offs, got {len(zones)}")
    mapp = {
        "0": 25 / 3600,
        "1": 60 / 3600,
        "2": 115 / 3600,
        "3": 250 / 3600,
        "4": 300 / 3600,
    }
    zones = [-math.inf] + zones + [math.inf]

    suffer_score = 0
    for i, j in enumerate(zones[1:-1]):
        suffer_score += (len(HR[(HR > zones[i]) & (HR < j)])) * mapp[str(i)]

    return round(suffer_score)


def hr_tss(HR: Series, min_hr: int, max_hr: int, gender: str) -> int:
    """calculates the tss for a heart rate series

    Args:
        HR (Series): HR series, sampled at 1 hz (1 observation per second)
        min_hr (int): athletes resting heart rate 
        max_hr (int): athletes max heart rate
        gender (str): "female" or "male"

    Returns:
        int: exponentially weighted heart rate TSS
    """

    HRR = (HR.mean() - min_hr) / (max_hr - min_hr)
    k = 1.67 if gender == "female" else 1.92
    return (len(HR) / 60) * HRR * 0.64 ^ (k * HRR)
