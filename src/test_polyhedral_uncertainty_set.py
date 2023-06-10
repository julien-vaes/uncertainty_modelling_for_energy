import unittest
import numpy as np

from sklearn.preprocessing import StandardScaler
from polyhedral_uncertainty_set import get_model_linear_constraints

class TestModelLinearConstraints(unittest.TestCase):

    def setUp(self):
        # Create some sample data
        # set dimensions
        n_rows = 10
        n_w = 4
        n_z = 8
        
        # generate random numpy arrays for A_w, A_z, and b
        self.A_w = np.random.rand(n_rows, n_w)
        self.A_z = np.random.rand(n_rows, n_z)
        self.b = np.random.rand(n_rows)

    # Test the function 'get_model_linear_constraints', make sure that the matrices returned are the same as the one in input
    def test_import_model_linear_constraints(self):

        my_A_w, my_A_z, my_b = get_model_linear_constraints(self.A_w, self.A_z, self.b)

        self.assertTrue(np.allclose(my_A_w, self.A_w))
        self.assertTrue(np.allclose(my_A_z, self.A_z))
        self.assertTrue(np.allclose(my_b, self.b))

if __name__ == '__main__':
    unittest.main()

