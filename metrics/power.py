"""a group of functions to calculate power metrics on a single power time series"""

from typing import List, Dict, Tuple
from pandas import Series
import numpy as np
from sklearn.linear_model import LinearRegression


def normalised_power(X: Series, window: int = 30) -> int:
    """Calculates the normalised power ® for the input, X.

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        window (int, optional): Window size over which smoothing is performed. Defaults to 30.

    Returns:
        float: Normalised power in watts
    """
    return round(np.sqrt(np.sqrt(np.mean(X.rolling(window).mean() ** 4))))

def xPower():
    pass # http://perfprostudio.com/webhelp/studio/scr/BikeScore.htm

def relative_intensity():
    pass

def quadrant_analysis():
    # https://trainingwithdata.com/golden-cheetah-the-ultimate-guide/15/ 
    # split force in to 4 groups and get cadence in each 
    pass 

def force_from_power(power: Series, cadence: Series) -> Series:
    return power / cadence

def kilojoules(X: Series) -> float:
    """Calculates the Kilohoules required to produce the watts

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)

    Returns:
        float: Kilojoules required to produce the watts
    """
    return round((X / 1000).sum(), 2)


def intensity_factor(X: Series, ftp: int) -> float:
    """Calculates the intensity factor®. NP/FTP.

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        ftp (int): Funtional threshold power

    Returns:
        float: Intensity factor
    """

    return round((normalised_power(X) / ftp), 2)


def training_stress_score(X: Series, ftp: int) -> int:
    """Calculates the training stress score® (TSS)

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        ftp (int): Funtional threshold power

    Returns:
        int: Training stress score
    """

    return round((len(X) * normalised_power(X) * intensity_factor(X)) / (ftp * 36))


def average_power(X: Series) -> float:
    """calcumates the mean

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)

    Returns:
        float: The average power in watts
    """
    return round(X.mean(), 2)


def variability_index(X: Series) -> float:
    """Calculates the variability index. Normalised power / average power.

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)

    Returns:
        float: Variability index
    """
    return round(normalised_power(X) / average_power(X), 2)


def power_curve(
    X: Series,
    intervals: List = [
        1,
        5,
        10,
        30,
        60,
        120,
        300,
        600,
        1800,
        *[3600 * i for i in range(1, 25)],
    ],
) -> Dict:

    """Calculates the maximum average power for the input durations.

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        intervals (List, optional): A list of time intervals in seconds

    Returns:
        Dict: a dictionary containing key, value pairs of 'time (secs)' : 'power (watts)'
    """

    di = {}
    for i in [i for i in intervals if i < len(X)]:
        di[i] = X.rolling(i).mean().max()

    return di


def critical_power_W(
    X: Series, intervals: List = [180, 300, 480, 1200], model: str = "three"
) -> Tuple[float, float]:

    """Calculates CP and W' based on Power = W’/Time + CP.


    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        intervals (List, optional): A list of time intervals in seconds
        model (str): 'three' or 'four', controls whether 3 or 4 time intervals are used in the CP calculation.

    Returns:
        Tuple: CP, W'
    """

    if model not in ["three", "four"]:
        raise ValueError(f"Model must be either three or four. Got {model}")

    if model == "three":
        intervals = intervals[:-1]

    power_duration_dict = power_curve(X, intervals)

    times = np.array([1 / t for t in power_duration_dict.keys()]).reshape(-1, 1)
    powers = np.array([p for p in power_duration_dict.values()])

    model = LinearRegression().fit(times, powers)

    return round(model.intercept_, 2), round(model.coef_[0] / 1000, 2)


def endurance_index(X):

    CP, W = critical_power_W(X)
    return round(W / CP, 2)


def vo2_max_estimate(w5: float, weight: float) -> float:
    return round(10.8 * (w5 / weight) + 7, 2)
