"""Float32 sensitivity of the exported encoding and rule sum, not a client test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research.tail_formula_additive import leaf_indices
from trade_research.corporate_cash import save_json,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--features-root',type=Path,required=True)
    args=ap.parse_args();root=args.root
    model=json.loads((root/'model_report.json').read_text())
    assert json.loads((root/'model_verification.json').read_text())['model_report_sha256']==sha(root/'model_report.json')
    f=pd.read_parquet(args.features_root/'features.parquet');f=f.loc[f.formula_input_valid].copy()
    raw=f[model['feature_names']].to_numpy(dtype=float)
    exact=np.floor(np.clip(100*raw+10000+.000001,0,999999)).astype('int32')
    rounded=np.floor(np.clip(np.float32(100)*raw.astype('float32')+np.float32(10000)+np.float32(.000001),np.float32(0),np.float32(999999))).astype('int32')
    score64=np.full(len(f),model['bias']);score32=np.full(len(f),model['bias'],dtype='float32')
    for tree in model['trees']:
        values=np.array(tree['value'])*model['learning_rate']
        score64+=values[leaf_indices(exact,tree)]
        score32+=values.astype('float32')[leaf_indices(rounded,tree)]
    cut=model['thresholds'][3]['threshold'];changed=(score64>cut)!=(score32>np.float32(cut))
    rows=[]
    for half in sorted(f.half.unique()):
        mask=f.half.eq(half).to_numpy()
        rows.append(dict(half=half,rows=int(mask.sum()),double_selected=int((score64[mask]>cut).sum()),
            float32_selected=int((score32[mask]>np.float32(cut)).sum()),changed_flags=int(changed[mask].sum())))
    r=dict(model_report_sha256=sha(root/'model_report.json'),feature_sha256=sha(args.features_root/'features.parquet'),
        rows=len(f),changed_encoded_scalars=int((exact!=rounded).sum()),changed_flags=int(changed.sum()),
        maximum_score_difference=float(abs(score64-score32).max()),by_half=rows,
        scope='仅模拟已有特征转float32后的整数编码和逐项加法，不模拟原始特征计算、客户端解析或编译；不能当作原生公式一致证明。',
        native_client_verified=False,outcomes_read=False,new_2026_prices_read=False)
    save_json(root/'numeric_precision_probe.json',r);print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
