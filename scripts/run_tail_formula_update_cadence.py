"""Run the predeclared new monthly models in isolated stage processes."""
import argparse
import json
from pathlib import Path
import subprocess


def run(months):
    assert months and len(months) == len(set(months))
    assert all(m in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12] for m in months)
    for month in months:
        root = Path('data/research/tail_formula_update_cadence/models') / f'm{month:02d}'
        root.mkdir(parents=True, exist_ok=True)
        for stage in ['model', 'verify_model', 'scores', 'verify_scores']:
            path = root / (stage + '_command.log')
            with path.open('w') as log:
                subprocess.run(['.venv/bin/python', '-m', 'trade_research.tail_formula_update_cadence',
                                stage, '--month', str(month)], stdout=log, check=True)
            r = json.loads(path.read_text())
            print(json.dumps(dict(month=month, stage=stage,
                **{k: r[k] for k in ['passed', 'rows', 'days', 'node_checks', 'max_score_difference',
                                     'new_2025H2_score_groups_read'] if k in r})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--months', type=int, nargs='+', required=True)
    run(parser.parse_args().months)
