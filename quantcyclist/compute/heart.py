"""functions related to heart rate analysis """

from pandas import Series
from power import normalised_power
def aerobic_decoupling(Power: Series, HR: Series):
    mid_point = len(HR)/2 
    np_start, np_end = normalised_power(Power[:mid_point]), normalised_power(Power[mid_point:])
    hr_start, hr_end = HR[:mid_point].mean(), HR[mid_point:].mean() 

    return round(100 * (1 - ((np_end/hr_end) / (np_start/hr_start))), 2)
