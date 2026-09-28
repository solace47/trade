import unittest
import numpy as np
from trade_research.tail_formula_path_order_labels import event_positions

class OpportunityOrderTests(unittest.TestCase):
    def test_risk_before_or_on_completion_rejects_but_later_risk_does_not(self):
        positive=np.ones((3,5),dtype=bool);risk=np.zeros((3,5),dtype=bool)
        risk[0,1]=True;risk[1,2]=True;risk[2,3]=True
        first,adverse,label=event_positions(positive,risk)
        np.testing.assert_array_equal(first,[2,2,2]);np.testing.assert_array_equal(adverse,[1,2,3])
        np.testing.assert_array_equal(label,[0,0,1])

    def test_gap_and_no_event_do_not_create_success(self):
        positive=np.array([[1,1,0,1,1],[0,0,0,0,0],[0,0,1,1,1]],dtype=bool)
        first,risk,label=event_positions(positive,np.zeros_like(positive))
        np.testing.assert_array_equal(first,[5,5,4]);np.testing.assert_array_equal(risk,[5,5,5])
        np.testing.assert_array_equal(label,[0,0,1])

if __name__=='__main__':
    unittest.main()
