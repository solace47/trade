"""Bounded conjunction learning on conservative, date-weighted opportunities."""
import numpy as np


def masks(x, conditions):
    selected = np.ones(len(x), dtype=bool)
    for feature, operator, threshold in conditions:
        selected &= x[:, feature] <= threshold if operator == 0 else x[:, feature] > threshold
    return selected


def atom_list(x, quantiles):
    cuts = np.quantile(x, quantiles, axis=0, method='lower')
    atoms = []
    for feature in range(x.shape[1]):
        for threshold in sorted(set(map(int, cuts[:, feature]))):
            for operator in [0, 1]:
                atom = (feature, operator, threshold)
                flag = masks(x, [atom])
                if flag.any() and not flag.all():
                    atoms.append(atom)
    return atoms


def statistics(selected, date_ids, half_ids, known, success, minimum_days, minimum_known):
    n = len(half_ids)
    count = np.bincount(date_ids[selected], minlength=n)
    good = np.bincount(date_ids[selected & success], minlength=n)
    identified = np.bincount(date_ids[selected & known], minlength=n)
    stats = []
    for half in [0, 1]:
        active = (half_ids == half) & (count > 0)
        days = int(active.sum())
        known_rows = int(identified[active].sum())
        if days < minimum_days or known_rows < minimum_known:
            return None
        stats.append(dict(days=days, rows=int(count[active].sum()), known=known_rows,
                          success=int(good[active].sum()),
                          lower=float(np.mean(good[active] / count[active]))))
    return stats


def ranking(record):
    return (-record['minimum_integer'], -record['mean_integer'],
            sum(x['rows'] for x in record['halves']), record['conditions'])


def learn(x, date_ids, half_ids, known, success, protocol, progress=None):
    atoms = atom_list(x, protocol['atomic_training_quantiles'])
    flags = [masks(x, [atom]) for atom in atoms]
    trace, beam, best, seen = [], [], None, set()
    for depth in range(1, protocol['max_conditions'] + 1):
        parents = [((), np.ones(len(x), dtype=bool))] if depth == 1 else [
            (r['conditions'], masks(x, r['conditions'])) for r in beam]
        layer = []
        for parent, parent_flag in parents:
            used = {a[0] for a in parent}
            for atom, flag in zip(atoms, flags):
                if atom[0] in used:
                    continue
                conditions = tuple(sorted((*parent, atom)))
                if conditions in seen:
                    continue
                seen.add(conditions)
                stats = statistics(parent_flag & flag, date_ids, half_ids, known, success,
                    protocol['minimum_signal_days_each_training_half'],
                    protocol['minimum_known_opportunity_rows_each_training_half'])
                if stats is None:
                    trace.append(dict(depth=depth, conditions=conditions, eligible=False))
                    continue
                values = [h['lower'] for h in stats]
                scale = protocol['objective_integer_scale']
                record = dict(depth=depth, conditions=conditions, eligible=True, halves=stats,
                    minimum_integer=int(np.floor(min(values)*scale + .5)),
                    mean_integer=int(np.floor(np.mean(values)*scale + .5)))
                trace.append(record)
                layer.append(record)
        layer.sort(key=ranking)
        beam = layer[:protocol['beam_width']]
        if beam and (best is None or ranking(beam[0]) < ranking(best)):
            best = beam[0]
        if progress:
            progress(dict(depth=depth, evaluated=len(seen), eligible=len(layer),
                          best_minimum=best['minimum_integer']/protocol['objective_integer_scale'] if best else None))
        if not beam:
            break
    # The exact unrounded lower bound determines the predeclared 0.5 gate.
    chosen = best if best and min(h['lower'] for h in best['halves']) > .5 else None
    return atoms, trace, best, chosen
