import itertools

import numpy as np

from trade_research.tail_formula_top_pairs_raw import rank_geometry, rank_weights
from trade_research.tail_formula_pairwise import derivatives, leaf_step


def test_tie_discounts_equal_exhaustive_random_permutations():
    for scores, k in [(np.array([2., 2., 1., 1., 1.]), 3), (np.zeros(5), 10),
                      (np.array([3., 2., 1., 1., 1.]), 2)]:
        groups = [np.flatnonzero(scores == value) for value in sorted(set(scores), reverse=True)]
        permutations = itertools.product(*(list(itertools.permutations(g)) for g in groups))
        realizations = []
        for parts in permutations:
            order = np.concatenate(parts)
            discount = np.zeros(len(scores))
            for position, item in enumerate(order, 1):
                discount[item] = 1 / np.log2(position + 1) if position <= k else 0
            realizations.append(discount)
        all_discounts = np.asarray(realizations)
        mean, within = rank_geometry(scores, np.zeros(len(scores), dtype=int), k)
        for i, j in itertools.permutations(range(len(scores)), 2):
            expected = np.mean(np.abs(all_discounts[:, i] - all_discounts[:, j]))
            actual = within[i] if scores[i] == scores[j] else abs(mean[i] - mean[j])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)


def test_weights_are_date_local_offset_invariant_and_initially_uniform():
    np.testing.assert_array_equal(
        rank_weights(np.zeros(6), np.zeros(6, dtype=int),
                     np.repeat([0, 1, 2], 3), np.tile([3, 4, 5], 3), 2),
        np.full(9, 1. / 9))
    days = np.repeat([0, 1], 4)
    positive = np.array([0, 0, 1, 1, 4, 4, 5, 5])
    negative = np.array([2, 3, 2, 3, 6, 7, 6, 7])
    np.testing.assert_allclose(rank_weights(np.zeros(8), days, positive, negative, 2), .25)
    score = np.array([.8, .8, .1, -.2, 2., 1., 0., -.1])
    before = rank_weights(score, days, positive, negative, 2)
    changed = score.copy(); changed[4:] = [10, -20, 30, -40]
    np.testing.assert_array_equal(rank_weights(changed, days, positive, negative, 2)[:4], before[:4])
    changed = score + np.where(days == 0, 8., -4.)
    np.testing.assert_array_equal(rank_weights(changed, days, positive, negative, 2), before)
    for day in [0, 1]:
        np.testing.assert_allclose(before[days[positive] == day].sum(), 1., atol=1e-15)


def test_frozen_rank_weights_have_correct_gradients_and_leaf_curvature():
    score = np.array([.7, .7, -.3, .2])
    positive, negative = np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])
    weight = rank_weights(score, np.zeros(4, dtype=int), positive, negative, 2)
    gradient, curvature, _ = derivatives(score, positive, negative, weight)
    def loss(s):
        return np.sum(weight * np.logaddexp(0, s[negative] - s[positive]))
    epsilon = 1e-5
    for i in range(4):
        direction = np.eye(4)[i]
        derivative = (loss(score + epsilon * direction) - loss(score - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(derivative, -gradient[i], atol=1e-10)
    leaves = np.array([0, 1, 0, 1])
    _, numerator, denominator = leaf_step(0, leaves, gradient, curvature, positive, negative)
    direction = (leaves == 0).astype(float)
    slope = (loss(score + epsilon * direction) - loss(score - epsilon * direction)) / (2 * epsilon)
    second = (loss(score + epsilon * direction) - 2 * loss(score) + loss(score - epsilon * direction)) / epsilon**2
    np.testing.assert_allclose(slope, -numerator, atol=1e-10)
    np.testing.assert_allclose(second, denominator, atol=3e-6)
