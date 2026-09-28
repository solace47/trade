"""Extend audited Q1 labels only for the frozen union's missing dates."""
import argparse
import json

from trade_research import tail_formula_path_relative_q1 as study
from trade_research import tail_formula_q1_candidates as previous
from trade_research import tail_formula_q1_candidate_observations as observations
from trade_research import tail_formula_q1_candidate_boundary as boundary
from trade_research.corporate_cash import sha


def setup():
    study.checked_models()
    coverage = json.loads((study.ROOT/'label_coverage_verification.json').read_text())
    assert coverage['passed'] and not coverage['all_needed_labels_cached']
    assert coverage['selection_verification_sha256'] == sha(study.ROOT/'selection_verification.json')
    observations.study = study; observations.ROOT = study.ROOT
    observations.LABELS = study.ROOT/'labels'; observations.CATALOG = observations.LABELS/'catalog'
    observations.OLD = previous.ROOT/'labels'
    boundary.study = study; boundary.ROOT = study.ROOT/'before1000'; boundary.OLD = observations.LABELS


def run(stage):
    setup()
    if stage in ['prepare','reuse_evidence','raw']:
        return getattr(observations,stage)()
    if stage in ['windows','labels30']:
        raw, labels = observations.setup_legacy()
        return raw.windows() if stage == 'windows' else labels.labels()
    if stage in ['verify_windows','verify_labels30']:
        import verify_tail_formula_q1_observations as verifier
        return verifier.main('windows' if stage == 'verify_windows' else 'labels30')
    if stage in ['build29','labels29']:
        return boundary.build() if stage == 'build29' else boundary.labels()
    import verify_tail_formula_q1_boundary as verifier
    verifier.coordinator = study
    return verifier.observations() if stage == 'verify29' else verifier.labels()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','reuse_evidence','raw','windows','verify_windows',
        'labels30','verify_labels30','build29','verify29','labels29','verify_labels29'])
    print(json.dumps(run(p.parse_args().stage),ensure_ascii=False,indent=2))
