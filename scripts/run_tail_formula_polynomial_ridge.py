"""Run each frozen polynomial arm/fold in isolated subprocess stages."""
import argparse
import json
from pathlib import Path
import subprocess


def run(fold):
    for arm in ['linear', 'quadratic']:
        root = Path('data/research') / f'tail_formula_polynomial_ridge_{arm}_{fold}'
        root.mkdir(parents=True, exist_ok=True)
        for stage in ['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify']:
            log = root / (stage + '_command.log')
            with log.open('w') as out:
                subprocess.run(['.venv/bin/python', '-m', 'trade_research.tail_formula_polynomial_ridge',
                                stage, '--arm', arm, '--fold', fold], stdout=out, check=True)
            record = json.loads(log.read_text())
            print(json.dumps(dict(arm=arm, fold=fold, stage=stage,
                **{k: record[k] for k in ['passed', 'rows', 'days', 'terms', 'selected', 'max_score_difference',
                                          'max_normal_equation_residual'] if k in record})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', choices=['2024', 'recent'], required=True)
    run(parser.parse_args().fold)
