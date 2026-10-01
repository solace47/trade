import pandas as pd

from audit_tail_formula_confirmed_success import mature


def test_missing_current_or_boundary_observation_dates_are_not_closed_negatives():
    frame=pd.DataFrame({'date':['2024-06-27']*5,'next_date':['2024-06-28',None,'2024-07-01','2024-06-27','2024-06-26']})
    spec={'training_start':'2024-01-01','training_end':'2024-07-01'}
    assert mature(frame,spec).tolist()==[True,False,False,False,False]
