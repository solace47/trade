"""Fixed, already audited market sentiment expressions and native helpers."""

import numpy as np

from . import tail_formula_baseline as prior

META, CONTROL = prior.META, prior.CONTROL
NEW_EXPRESSIONS = {'EMREPAIR':'IF(EMREADY,100*EMUP/EMN,DRAWNULL)',
                   'EMFADE':'IF(EMREADY,100*EMDN/EMN,DRAWNULL)'}
EXPRESSIONS = {**CONTROL, **NEW_EXPRESSIONS}
DAILY_HELPER = 'AGE1:REF(BARSCOUNT(C),1);\n'
HELPER = '''EMQ20:=VALUEWHEN(TIME=1420,C);
EMQ49:=VALUEWHEN(TIME=1449,C);
EMPC:=ROUND(DYNAINFO(3)*100);
EMP20:=ROUND(EMQ20*100);
EMP49:=ROUND(EMQ49*100);
EMB0:=BARSLAST(DATE<>REF(DATE,1))+1;
EMV:=VALUEWHEN(TIME=1449,SUM(V,EMB0));
EMA:=VALUEWHEN(TIME=1449,SUM(AMOUNT,EMB0));
EMAGE:="YJEMAGE.AGE1#DAY";
EMMB:=FINANCE(3)=1 AND NOT(NAMELIKE('ST')) AND NOT(NAMELIKE('*ST'));
EMCLK:=VALUEWHEN(TIME=1420,DATE)=DATE AND VALUEWHEN(TIME=1449,DATE)=DATE;
EMGOOD:=EMCLK AND EMPC>0 AND EMP20>0 AND EMP49>0 AND EMQ20-EMQ20=0 AND EMQ49-EMQ49=0
 AND ABS(EMQ20*100-EMP20)<=0.01 AND ABS(EMQ49*100-EMP49)<=0.01;
EMOK:=EMMB AND EMAGE>=60 AND EMV>0 AND EMA>0 AND EMGOOD;
N:IF(EMOK,1,0);
UP:IF(EMOK AND EMP20<EMPC AND EMP49>EMPC,1,0);
DN:IF(EMOK AND EMP20>EMPC AND EMP49<EMPC,1,0);
BAD:IF(EMMB AND EMAGE>=60 AND EMV>0 AND (EMA<=0 OR NOT(EMGOOD)),1,0);
'''
EXTRA_HEADER = '''EMN:=INSUM('沪深Ａ股','YJEM60',1,0);
EMUP:=INSUM('沪深Ａ股','YJEM60',2,0);
EMDN:=INSUM('沪深Ａ股','YJEM60',3,0);
EMBAD:=INSUM('沪深Ａ股','YJEM60',4,0);
EMREADY:=EMN>=2000 AND EMBAD=0 AND EMUP>=0 AND EMDN>=0 AND EMUP+EMDN<=EMN;
'''
HEADER = prior.HEADER + EXTRA_HEADER
NATIVE_GATE = 'EMREADY'


def transition_atoms(p20, p49, pc):
    """Strict sign crossings; flat prices are neither repair nor fade."""
    a, b, c = [np.asarray(x, float) for x in [p20, p49, pc]]
    assert a.shape == b.shape == c.shape
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (a > 0) & (b > 0) & (c > 0)
    cents = [np.floor(x * 100 + .5) for x in [a, b, c]]
    good &= np.logical_and.reduce([abs(x * 100 - y) <= .01 for x, y in zip([a, b, c], cents)])
    x, y, z = cents
    return np.column_stack([np.where(good, (x < z) & (y > z), np.nan),
                            np.where(good, (x > z) & (y < z), np.nan)])

