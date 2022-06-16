"""a group of functions to calculate power metrics on a single power time series"""

from typing import List, Dict, Tuple
from pandas import Series
import pandas as pd
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


def xPower(X: Series, window: int = 25):
    """Calculates the xpower ® for the input, X. xPower uses a 25 second exponential average, rather than the linear decay used in normalised power.

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        window (int, optional): Window size over which smoothing is performed. Defaults to 25.

    Returns:
        float: xpower in watts
    """
    return round(np.sqrt(np.sqrt(np.mean(pd.Series.ewm(X, span=window).mean() ** 4))))

def intensity_factor(X: Series, ftp: int) -> float:
    """Calculates the intensity factor®. NP/FTP.

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        ftp (int): Funtional threshold power

    Returns:
        float: Intensity factor
    """

    return round((normalised_power(X) / ftp), 2)


def relative_intensity(X: Series, ftp: int) -> float:
    """Calculates the relative intesity. xpower/FTP.

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        ftp (int): Funtional threshold power

    Returns:
        float: relative intensity
    """

    return xPower(X)/ ftp


def quadrant_analysis(force: Series, cadence: Series) -> Dict:
    """Returns the a quadrant cadence and force analysis. That is, the mean cadence for each quartile of force observed in a series.

    Args:
        force (Series): Input series, force sampled at 1Hz (1 sec)
        cadence (Series):  Input series, cadence in rpm sampled at 1Hz (1 sec)

    Returns: 
        Dict: key value pairs of quadrants of force and mean cadene
    """
    
    df = pd.concat([force, cadence], axis=1)
    cut_offs = [c for c in df.quantile([.25, .5, .75]).iloc[:, 0]]
    pairs = [(-math.inf, cut_offs[0]), (cut_offs[0], cut_offs[1]), (cut_offs[1], cut_offs[2]), (cut_offs[2], math.inf)]
    keys = ['low_force', 'mid_low_force', 'mid_high_force', 'high_force']
    di= {}
    for i, r in enumerate(pairs):
        di[keys[i]] = df[(df[0] > r[0]) & (df[0] < r[1])][1].mean()

    return di


def force_from_power(power: Series, cadence: Series) -> Series:
    """calculates force given power and speed

    Args:
        power (Series): Input series, power in watts sampled at 1Hz (1 sec)
        cadence (Series): Input series, cadence in rpm sampled at 1Hz (1 sec)

    Returns:
        Series: Force time series
    """
    return power / cadence


def kilojoules(X: Series) -> float:
    """Calculates the Kilohoules required to produce the watts

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)

    Returns:
        float: Kilojoules required to produce the watts
    """
    return round((X / 1000).sum(), 2)


def training_stress_score(X: Series, ftp: int) -> int:
    """Calculates the training stress score® (TSS)

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        ftp (int): Funtional threshold power

    Returns:
        int: Training stress score
    """

    return round((len(X) * normalised_power(X) * intensity_factor(X, ftp)) / (ftp * 36))


def bike_score(X: Series, ftp: int) -> int:
    """Calculates Skiba's bike score

    Args:
        X (Series): Input series, power in watts sampled at 1Hz (1 sec)
        ftp (int): Funtional threshold power

    Returns:
        int: bike score
    """

    return round((len(X) * xPower(X) * relative_intensity(X, ftp)) / (ftp * 36))

def average_power(X: Series) -> float:
    """Calculates the mean power

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



def endurance_index(X: Series) -> float:
    """Calculates the endurance index, W'/CP

    Args:
        X (Series):  Input series, power in watts sampled at 1Hz (1 sec)

    Returns:
        float: endurance index
    """
    CP, W = critical_power_W(X)
    return round(W / CP, 2)


def vo2_max_estimate(w5: float, weight: float) -> float:
    """Estimates vo2 max from 5 min power, acsm model

    Args:
        w5 (float): _description_
        weight (float): _description_

    Returns:
        float: co2 max estimate
    """
    return round(10.8 * (w5 / weight) + 7, 2)




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

def predict_power_given_time(time: int, CP: int, W: float) -> int:
    """Models the theoretical power a rider can produce for time t, given a CP and W'

    Args:
        time (int): time in seconds
        CP (int): CP
        W (float): W'

    Returns:
        int: power in watts
    """
    return round((W * 1000 / time) + CP)


def predict_tte_given_power(power: int, CP: int, W: float) -> int: 
    """Models the theoretical time a rider can ride at a specific power, given a CP and W'

    Args:
        power (int): _description_
        CP (int): _description_
        W (float): _description_

    Returns:
        int: _description_
    """
    return round((W * 1000)/(power - CP))


def AWC(power : int, time: int, CP: int) -> int:
    """calculates the AWC for a given maximal effort

    Args:
        power (int): power in watts
        time (int): time in seconds
        CP (int): critical power in watts

    Returns:
        int: anaerobic work capacity, joules
    """
    return round((power - CP) * time)

def tau(power : int, time: int, CP: int) -> int:
    """calculates the tau (i.e. AWC used) for a given maximal effort

    Args:
        power (int): power in watts (should be less than maximum for this interval)
        time (int): time in seconds
        CP (int): critical power in watts

    Returns:
        int: anaerobic work capacity, joules
    """
    return round((power - CP) * time)


def get_wbal_recovery_balance(recov_dur: int, recovery_watts: int, awc: int, w_exp: int, CP: int) -> float:
    """calculates the recovered awc, based on recovery duration, watts, awc and w_exp

    Args:
        recov_dur (int): time spent recovering
        recovery_watts (int): average watts when recovering 
        awc (int): anaerobic work capacity, joules 
        w_exp (int): tau, the w' expended 

    Returns:
        float: w balance in joules
    """
    tw = 546*10**(-0.1 * (CP - recovery_watts)) + 316
    w_bal = awc - (w_exp) * 10 **(-(recov_dur/tw))
    return w_bal 


def percentage_awc_remaining(w_bal: float, awc: int ) -> float:
    """calculate the percentage of total awc used 

    Args:
        w_bal (float): current w' balance
        awc (int): anaerobic work capacity

    Returns:
        int: Proportion remaining (0,1)
    """
    return round( 1- (w_bal / awc), 2) 

def anaerobic_cap_remaining_watts(perc_remain: float, PMAX: int, CP: int) -> float:
    """Calculates anaerobic capacity remaining 

    Args:
        perc_remain (float): proportion remaining (0,1)
        CP (int): critical power in watts
        PMAX (int): max 1s power in watts

    Returns:
        float: Remaining capacity in watts 
    """
    return round(PMAX -((PMAX - CP ) * perc_remain), 2)




