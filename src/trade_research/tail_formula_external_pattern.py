"""Literal priority branches from the pinned external rule, without its narratives."""
import numpy as np


BRANCHES = ['early_fade', 'late_spike', 'weak', 'flat', 'p4', 'p5',
            'p5_high', 'p4_side', 'above', 'weak_other']


def classify(d):
    close, first, wap = d.last_close, d.first_close, d.vwap
    direction = (close - d.tail_first) / d.tail_first * 100
    volume_ratio = ((d.tail_volume / 20) / (d.half_volume / 115)).where(d.half_volume.gt(0), 1.)
    above = d.tail_above / 20
    conditions = [
        (d.morning_high > first * 1.01) & (close < first * .99),
        (close > wap * 1.02) & (direction > 1) & (volume_ratio > 1.5),
        (above < .3) & (close < first * .98),
        ((d.day_high - d.day_low) / d.day_low * 100 < 2) & ((close - first).abs() / first < .01),
        (d.afternoon_high > wap * 1.01) & (close >= wap * .995) & (above >= .5) & (direction > -.3),
        (close / first - 1 > 0) & (close / first - 1 < .03) & (above >= .7)
            & (volume_ratio >= 1) & (volume_ratio <= 2) & (direction >= -.1),
        (close / first - 1 > .02) & (close / first - 1 < .06) & (above >= .6)
            & (volume_ratio >= .9) & ((close - d.tail_high).abs() / d.tail_high < .003),
        (above >= .4) & (close > first * .995) & (close >= wap * .998) & (direction > -.2),
        above >= .5,
    ]
    branch = np.select([x.fillna(False).to_numpy(bool) for x in conditions], range(9), default=9)
    return branch, np.isin(branch, [4, 5, 6, 7, 8])
