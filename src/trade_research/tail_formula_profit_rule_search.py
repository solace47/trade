"""Profit-directed bounded conjunctions; unknown returns stay unknown."""
import numpy as np

from .tail_formula_rule_search import masks


def statistics(selected, date_ids, half_ids, known, success, reference_known, reference, robust=False):
    n = len(half_ids)
    count = np.bincount(date_ids[selected], minlength=n)
    identified = np.bincount(date_ids[selected & known], minlength=n)
    good = np.bincount(date_ids[selected & success], minlength=n)
    price_mask = selected & reference_known
    prices = np.bincount(date_ids[price_mask], minlength=n)
    totals = np.bincount(date_ids[price_mask], weights=reference[price_mask], minlength=n)
    result = []
    for half in [0, 1]:
        signal_days = (half_ids == half) & (count > 0)
        price_days = signal_days & (prices > 0)
        result.append(dict(days=int(signal_days.sum()), rows=int(count[signal_days].sum()),
            known=int(identified[signal_days].sum()), success=int(good[signal_days].sum()),
            lower=float(np.mean(good[signal_days]/count[signal_days])) if signal_days.any() else None,
            reference_days=int(price_days.sum()), reference_rows=int(prices[price_days].sum()),
            reference=float(np.mean(totals[price_days]/prices[price_days])) if price_days.any() else None))
        if robust:
            values = totals[price_days]/prices[price_days]
            result[-1].update(reference_median=float(np.median(values)) if len(values) else None,
                reference_clipped=float(np.mean(np.clip(values,-.03,.03))) if len(values) else None,
                positive_reference_fraction=float(np.mean(values>0)) if len(values) else None)
    return result


def supported(halves, protocol):
    return all(h['days'] >= protocol['minimum_signal_days_each_training_half']
        and h['known'] >= protocol['minimum_known_opportunity_rows_each_training_half']
        and h['reference_days'] >= protocol['minimum_reference_days_each_training_half']
        and h['reference_rows'] >= protocol['minimum_known_reference_rows_each_training_half']
        for h in halves[:protocol.get('training_segments',2)])


def qualifies(halves):
    return all(h['reference'] is not None and h['reference'] > 0 and h['lower'] > .5
        and ('reference_median' not in h or (h['reference_median']>0 and h['reference_clipped']>0
             and h['positive_reference_fraction']>.5)) for h in halves)


def ranking(record):
    if 'robust_ranks' in record:
        return (*(-n for n in record['robust_ranks']),sum(h['rows'] for h in record['halves']),record['conditions'])
    return (-record['minimum_integer'], -record['mean_integer'],
            sum(h['rows'] for h in record['halves']), record['conditions'])


def learn(x, atoms, date_ids, half_ids, known, success, reference_known, reference, protocol, progress=None):
    flags = [masks(x, [atom]) for atom in atoms]
    trace, beam, seen, best, chosen = [], [], set(), None, None
    for depth in range(1, protocol['max_conditions'] + 1):
        parents = [((), np.ones(len(x), dtype=bool))] if depth == 1 else [
            (r['conditions'], masks(x, r['conditions'])) for r in beam]
        layer = []
        for parent, parent_flag in parents:
            for atom, flag in zip(atoms, flags):
                if atom[0] in {a[0] for a in parent}:
                    continue
                conditions = tuple(sorted((*parent, atom)))
                if conditions in seen:
                    continue
                seen.add(conditions)
                halves = statistics(parent_flag & flag, date_ids, half_ids, known, success, reference_known, reference,
                                    robust=protocol.get('robust_objective',False))
                record = dict(depth=depth, conditions=conditions, eligible=supported(halves, protocol), halves=halves)
                trace.append(record)
                if not record['eligible']:
                    continue
                active = halves[:protocol.get('training_segments',2)]
                values = [h['reference'] for h in active]
                scale = protocol['objective_integer_scale']
                record.update(minimum_integer=int(np.floor(min(values)*scale+.5)),
                              mean_integer=int(np.floor(np.mean(values)*scale+.5)))
                if protocol.get('robust_objective',False):
                    median = [h['reference_median'] for h in active]
                    clipped = [h['reference_clipped'] for h in active]
                    record['robust_ranks'] = [int(np.floor(v*scale+.5)) for v in
                        [min(median),np.mean(median),min(clipped),np.mean(clipped)]]
                layer.append(record)
                if qualifies(active) and (chosen is None or ranking(record) < ranking(chosen)):
                    chosen = record
        layer.sort(key=ranking)
        beam = layer[:protocol['beam_width']]
        if beam and (best is None or ranking(beam[0]) < ranking(best)):
            best = beam[0]
        if progress:
            progress(dict(depth=depth, candidates=len(trace), eligible=len(layer),
                          best_reference=best['minimum_integer']/protocol['objective_integer_scale'] if best else None))
        if not beam:
            break
    return trace, best, chosen
