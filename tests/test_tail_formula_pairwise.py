import unittest
import numpy as np
import pandas as pd

from trade_research.tail_formula_pairwise import make_pairs, derivatives, leaf_step


class PairwiseOpportunityTests(unittest.TestCase):
    def test_fixed_pairs_cover_every_member_and_do_not_cross_dates(self):
        f=pd.DataFrame(dict(date=['2024-01-02']*5+['2024-01-03']*2,opportunity15=[1,0,0,0,0,1,1]))
        pairs,days=make_pairs(f)
        self.assertEqual(len(pairs),32)
        self.assertEqual(days.pairs.tolist(),[32,0])
        self.assertAlmostEqual(pairs.weight.sum(),1)
        for _,part in pairs.groupby('turn'):
            self.assertEqual(set(part.positive)|set(part.negative),set(range(5)))
        self.assertTrue(f.loc[pairs.positive,'date'].reset_index(drop=True).equals(f.loc[pairs.negative,'date'].reset_index(drop=True)))
        pd.testing.assert_frame_equal(pairs,make_pairs(f)[0])

    def test_score_gradient_matches_finite_differences_and_date_offset_cancels(self):
        score=np.array([.3,-.4,.9,-.1]);pos=np.array([0,2]);neg=np.array([1,3]);w=np.array([1.,.5])
        g,_,loss=derivatives(score,pos,neg,w)
        for i in range(len(score)):
            d=np.zeros(len(score));d[i]=1e-6
            numeric=(derivatives(score+d,pos,neg,w)[2]-derivatives(score-d,pos,neg,w)[2])/2e-6
            self.assertAlmostEqual(g[i],-numeric,places=9)
        shifted=score+np.array([8.,8.,-5.,-5.])
        np.testing.assert_allclose(g,derivatives(shifted,pos,neg,w)[0],rtol=0,atol=1e-14)
        self.assertAlmostEqual(loss,derivatives(shifted,pos,neg,w)[2],places=13)

    def test_same_leaf_pairs_have_zero_directional_curvature(self):
        score=np.array([.2,-.2]);pos=np.array([0]);neg=np.array([1]);w=np.array([1.])
        g,h,_=derivatives(score,pos,neg,w)
        self.assertEqual(leaf_step(0,np.array([0,0]),g,h,pos,neg),(0.,0.,0.))

    def test_leaf_curvature_matches_scalar_loss_second_derivative(self):
        score=np.array([.2,-.2,.1]);pos=np.array([0,0]);neg=np.array([1,2]);w=np.array([.4,.6])
        leaves=np.array([0,1,0]);g,h,loss=derivatives(score,pos,neg,w)
        _,num,curv=leaf_step(0,leaves,g,h,pos,neg)
        d=(leaves==0)*1e-4
        a=derivatives(score+d,pos,neg,w)[2];b=derivatives(score-d,pos,neg,w)[2]
        self.assertAlmostEqual(curv,(a-2*loss+b)/1e-8,places=7)
        self.assertAlmostEqual(num,-(a-b)/2e-4,places=8)


if __name__=='__main__':
    unittest.main()
