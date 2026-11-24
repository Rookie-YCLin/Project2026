from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Python"))
from allocation import METHODS, allocate, index, simplex, weights


class AllocationTests(unittest.TestCase):
    def test_allocation_against_enumeration(self):
        rng=np.random.default_rng(1024)
        s=np.sort(rng.uniform(0,25,(100,4)),axis=-1)[:,::-1]
        s[:5]=[[0,0,0,0],[8,0,0,0],[8,4,4,2],[8,1,1e-10,0],[9,1,.2,.01]]
        bits=rng.integers(0,2,len(s))
        for method in METHODS[1:]:
            for q in (1.25,4.,13.75):
                x,d=allocate(s,bits,q,method)
                self.assertTrue(np.all(x>=0) and np.all(d["weights"]>=0))
                np.testing.assert_allclose(d["weights"].sum(axis=1),1,atol=1e-14)
                np.testing.assert_allclose(x.sum(axis=1),d["T"],atol=1e-12)
                np.testing.assert_array_equal(index(x.sum(axis=1),q)%2,bits)
                for m,(orig,b) in enumerate(zip(s,bits,strict=True)):
                    k=np.arange(b,100,2)
                    feasible=np.ones(k.size,dtype=bool) if method=="S1" else np.all(
                        orig+(k*q-orig.sum())[:,None]*d["weights"][m]>=-1e-12,axis=1)
                    candidates=k[feasible]
                    best=min(candidates,key=lambda k:(abs(k*q-orig.sum()),-k))
                    self.assertEqual(d["k"][m],best)
                if method in ("S2","S3","S4","S5"):
                    j=int(method[1])-2
                    np.testing.assert_array_equal(x[:,np.arange(4)!=j],s[:,np.arange(4)!=j])

    def test_projection_kkt_and_zero(self):
        s=np.array([[8.,4.,2.,0.],[9.,9.,.03,0],[2.,1.,0.,0.]])
        T=np.array([5.,1.,0.])
        x=simplex(s,T)
        np.testing.assert_allclose(x.sum(axis=1),T)
        self.assertTrue(np.all(x>=0))
        for a,b in zip(s[:2],x[:2],strict=True):
            active=b>0
            tau=(b-a)[active][0]
            np.testing.assert_allclose((b-a)[active],tau,atol=1e-13)
            self.assertTrue(np.all(a[~active]+tau<=0))
        for m in ("S6","S7"):
            np.testing.assert_array_equal(weights(np.zeros((1,4)),m),np.full((1,4),.25))

    def test_tie_and_unsorted_reconstruction(self):
        s=np.array([[5.,3.,1.,1.]])
        x,d=allocate(s,np.array([1]),5.,"S5")
        self.assertEqual(d["T_unconstrained"][0],15.)
        self.assertTrue(d["rank_reversal"][0])
        rng=np.random.default_rng(218)
        u,_=np.linalg.qr(rng.normal(size=(4,4)))
        v,_=np.linalg.qr(rng.normal(size=(4,4)))
        A=(u*s[0])@v.T
        B=(u*x[0])@v.T
        self.assertAlmostEqual(np.linalg.norm(B-A,"fro")**2,np.sum((x-s)**2))
        self.assertAlmostEqual(np.linalg.svd(B,compute_uv=False).sum(),d["T"][0])
        self.assertFalse(np.allclose(B,(u*np.sort(x[0])[::-1])@v.T))


if __name__=="__main__":
    unittest.main()
