"""Apply independent event statistics checks to the disclosed-purchase study."""
import json

from trade_research.first_insider_short import ROOT, GROUPS
from verify_touch_sequence_analysis import check


if __name__=='__main__':
    print(json.dumps(check(ROOT=ROOT,GROUPS=GROUPS,event_column='event'),ensure_ascii=False,indent=2))
