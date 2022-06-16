"""functions used in the anaerobic power model """


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
    return round((W * 1000) / (power - CP))


def AWC(power: int, time: int, CP: int) -> int:
    """calculates the AWC for a given maximal effort

    Args:
        power (int): power in watts
        time (int): time in seconds
        CP (int): critical power in watts

    Returns:
        int: anaerobic work capacity, joules
    """
    return round((power - CP) * time)


def tau(power: int, time: int, CP: int) -> int:
    """calculates the tau (i.e. AWC used) for a given maximal effort

    Args:
        power (int): power in watts (should be less than maximum for this interval)
        time (int): time in seconds
        CP (int): critical power in watts

    Returns:
        int: anaerobic work capacity, joules
    """
    return round((power - CP) * time)


def get_wbal_recovery_balance(
    recov_dur: int, recovery_watts: int, awc: int, w_exp: int, CP: int
) -> float:
    """calculates the recovered awc, based on recovery duration, watts, awc and w_exp

    Args:
        recov_dur (int): time spent recovering
        recovery_watts (int): average watts when recovering
        awc (int): anaerobic work capacity, joules
        w_exp (int): tau, the w' expended

    Returns:
        float: w balance in joules
    """
    tw = 546 * 10 ** (-0.1 * (CP - recovery_watts)) + 316
    w_bal = awc - (w_exp) * 10 ** (-(recov_dur / tw))
    return w_bal


def percentage_awc_remaining(w_bal: float, awc: int) -> float:
    """calculate the percentage of total awc used

    Args:
        w_bal (float): current w' balance
        awc (int): anaerobic work capacity

    Returns:
        int: Proportion remaining (0,1)
    """
    return round(1 - (w_bal / awc), 2)


def anaerobic_cap_remaining_watts(perc_remain: float, PMAX: int, CP: int) -> float:
    """Calculates anaerobic capacity remaining

    Args:
        perc_remain (float): proportion remaining (0,1)
        CP (int): critical power in watts
        PMAX (int): max 1s power in watts

    Returns:
        float: Remaining capacity in watts
    """
    return round(PMAX - ((PMAX - CP) * perc_remain), 2)
