"""Direct NR7 selection; reuse existing verified annual/comparison machinery."""
import argparse
import inspect
import json
from pathlib import Path

from trade_research import tail_formula_narrow_breakout as study
import run_tail_formula_volume_memory as common


def bind():
    common.study = study
    common.EXECUTION = Path('config') / (study.STEM + '_evaluation_protocol.json')


def finish():
    bind()
    # The declared direct-shape gate requires strictly lower bad3 in both
    # years. Adapt just this comparator before computing any new pair.
    source = inspect.getsource(common.finish)
    needle = "main('memory'+y,y)['bad3']<=main('control'+y,y)['bad3']"
    assert source.count(needle) == 1
    namespace = dict(common.__dict__)
    exec(compile(source.replace(needle,needle.replace('<=','<')), '<fixed-nr7-strict-bad3-gate>', 'exec'),namespace)
    return namespace['finish']()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','freeze','analyze','finish']); a=p.parse_args()
    if a.stage in ['prepare','freeze']: result=getattr(study,a.stage)()
    elif a.stage=='finish': result=finish()
    else:
        bind(); result=common.analyze()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
