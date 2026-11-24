"""人工矩陣與資料排列測試，可用 unittest 執行。"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from image_pipeline import ll_channels, matrices, permutations
from numerics import angles, distribution, perturb, spectral, subspace_errors, svd, validate_svd


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([np.diag([4, 3, 2, 1]), [[5,1,-2,3],[2,4,0,1],[1,2,7,4],[0,0,1,3]],
                           np.zeros((4,4)), np.outer([1,2,3,4],[4,3,2,1]),
                           np.diag([5,3,3,1]), np.diag([5,3+1e-8,3,1])], dtype=float)
        self.F = svd(self.A)

    def test_identity_and_diagnostics(self):
        validate_svd(self.A, self.F)
        result, diag, _ = perturb(self.A, self.A, self.F, self.F)
        self.assertTrue(np.all(result["D_F"] == 0))
        self.assertTrue(np.all(result["D_sigma"] == 0))
        self.assertTrue(np.all(result["nuclear_abs"] == 0))
        self.assertTrue(diag["original_zero_matrix"][2])
        self.assertTrue(diag["original_near_rank_deficient"][3])
        self.assertTrue(diag["original_repeated_2"][4])
        self.assertFalse(diag["original_repeated_2"][5])
        self.assertTrue(diag["original_near_repeated_2"][5])
        self.assertTrue(np.isnan(result["gap_norm_1"][2]))

    def test_sign_and_right_vector(self):
        u, _, _ = self.F
        np.testing.assert_array_equal(angles(u,u), angles(u,-u))
        A = self.A[0:1]
        B = A @ np.roll(np.eye(4),1,axis=1)
        m, _, _ = perturb(A,B,svd(A),svd(B))
        self.assertGreater(m["D_F"][0],0)
        self.assertEqual(m["D_sigma"][0],0)
        self.assertTrue(all(m[f"theta_v_{j}"][0] == 90 for j in range(1,5)))
        self.assertTrue(all(m[f"theta_u_{j}"][0] == 0 for j in range(1,5)))

    def test_subspace_nonuniqueness(self):
        A = self.A[4:5]
        U,s,V = svd(A)
        R = np.eye(4)
        R[1:3,1:3] = [[0.6,-0.8],[0.8,0.6]]
        U2,V2 = U@R,V@R
        validate_svd(A,(U2,s,V2))
        self.assertGreater(angles(U,U2).max(),50)
        direct, d, _, _, counts = subspace_errors((U,s,V),(U2,s,V2),1e-6)
        self.assertLess(d,1e-15)
        self.assertEqual(counts[1],1)
        self.assertFalse(direct[0,1])

    def test_distributions(self):
        result = distribution(np.array([0,10,40,70,np.nan]))
        np.testing.assert_allclose(result, [4,30,25,np.std([0,10,40,70]),0,70,3,7.5,47.5,61,65.5])
        np.testing.assert_array_equal(spectral(np.array([[6,4,3,0]]))[1], [[2,1,1,3]])

    def test_spatial_layout_with_odd_edges(self):
        img = np.arange(17*19*3,dtype=np.uint8).reshape(17,19,3)
        ll,yuv = ll_channels(img)
        self.assertEqual(ll.shape,(3,9,10))
        self.assertTrue(np.all(yuv[-1] == 0) and np.all(yuv[:,-1] == 0))
        shuffled = permutations(4, 1)
        A = matrices(ll,shuffled)
        self.assertEqual(A.shape,(3,4,4,4))
        self.assertEqual(A.dtype,np.float64)


if __name__ == "__main__":
    unittest.main()
