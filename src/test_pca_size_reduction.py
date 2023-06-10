import unittest
import numpy as np

# import pca_size_reduction

from sklearn.preprocessing import StandardScaler
from pca_size_reduction import get_linear_transformations_between_original_and_standardized_data
from pca_size_reduction import StepSizeReductionViaPCA

class TestLinearTransformations(unittest.TestCase):
    def setUp(self):
        # Create some sample data
        self.data = {
                'attr1': np.array([[1, 2, 3, 4, 5], [5, 5, 5, 5, 5], [0, 10, 100, 10, 0]]),
                'attr2': np.array([[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1]]),
                'attr3': np.array([[57, 50], [12, 54], [94, 3], [22, 99], [1, 96], [85, 28], [32, 49], [66, 73], [83, 96], [75, 7]])
                }

        # Get the linear transformations
        self.get_std_data_func, self.get_orig_data_func, self.orig_to_std, self.std_to_orig = get_linear_transformations_between_original_and_standardized_data(self.data)

    def test_standardization(self):
        # Test that the standardized data matches the expected values
        expected_std_data ={
                'attr1': np.array([[-0.46291005, -1.1111678 , -0.72908521, -0.88900089,  0.70710678],
                                [ 1.38873015, -0.20203051, -0.68489823, -0.50800051,  0.70710678],
                                [-0.9258201 ,  1.31319831,  1.41398344,  1.3970014 , -1.41421356]]),
                'attr2': np.array([[-1., -1., -1., -1.,  0.,  1.,  1.,  1.,  1.],
                                [ 1.,  1.,  1.,  1.,  0., -1., -1., -1., -1.]]),
                'attr3': np.array([[ 0.13557116, -0.16276393],
                                [-1.28319678, -0.04439016],
                                [ 1.30211368, -1.55365567],
                                [-0.96791501,  1.2873147 ],
                                [-1.63000672,  1.19853438],
                                [ 1.0183601 , -0.81381964],
                                [-0.65263325, -0.19235737],
                                [ 0.41932475,  0.51788522],
                                [ 0.95530374,  1.19853438],
                                [ 0.70307833, -1.43528191]])
                }
        std_data = self.get_std_data_func(self.data)
        for attr in expected_std_data:
            self.assertTrue(np.allclose(expected_std_data[attr], std_data[attr]))

    def test_standardization_linear(self):
        # Test that the standardized data obtained with the linear transformation corresponds to the data obtained with the build in function StandardScaler

        # Get the standardized data obtained with the linear transformation
        std_data = self.get_std_data_func(self.data)
        for attr in self.data:

            # Get the standardized data obtained with the function 'StandardScaler' 
            my_scaler = StandardScaler()
            my_y = my_scaler.fit_transform(self.data[attr])

            self.assertTrue(np.allclose(my_y, std_data[attr]))

    def test_recovery(self):
        # Test that the recovered data matches the original data
        expected_data = self.data
        recovered_data = self.get_orig_data_func(self.get_std_data_func(self.data))
        for attr in expected_data:
            self.assertTrue(np.allclose(expected_data[attr], recovered_data[attr]))

    def test_linear_transformations(self):
        # Test that the linear transformations functions are correct and matches the hard-coded one with the matrix A and vector b
        for attr in self.data:
            A, b = self.orig_to_std[attr]
            x = self.data[attr]
            y_expected = np.dot(x,A) + b
            y_actual = self.get_std_data_func({attr: x})[attr]
            self.assertTrue(np.allclose(y_expected, y_actual))

            A_inv, b_inv = self.std_to_orig[attr]
            y = self.get_std_data_func({attr: x})[attr]
            x_expected = np.dot(y,A_inv) + b_inv
            x_actual = self.get_orig_data_func({attr: y})[attr]
            self.assertTrue(np.allclose(x_expected, x_actual))

class TestStepSizeReductionViaPCA(unittest.TestCase):
    
    def setUp(self):
        # create a 2-dimension array with 5 data points each
        self.old_data = {
            'old_attr1': np.array([[1,2,3,4,5],[6,7,8,9,10]]),
            'old_attr2': np.array([[11,12,13,14,15],[16,17,18,19,20]])
        }
        # set up some test parameters
        attributes_to_merge_together = {
            'new_attr': ['old_attr1', 'old_attr2']
        }
        explained_variance_threshold = {
            'new_attr': 1.0e-4
        }
        n_directions_threshold = {
            'new_attr': 2
        }
        self.step_size_reduction_via_pca = StepSizeReductionViaPCA(
            a_step_index = 0,
            a_attributes_to_merge_together = attributes_to_merge_together,
            a_explained_variance_threshold = explained_variance_threshold,
            a_n_directions_threshold = n_directions_threshold
        )
        # store some expected results
        self.expected_attributes_to_merge = {
            'new_attr': ['old_attr1', 'old_attr2']
        }
        self.expected_explained_variance_threshold = {
            'new_attr': 1.0e-4
        }
        self.expected_n_directions_threshold = {
            'new_attr': 2
        }
    
    def test_attributes_to_merge_together(self):
        for k in self.expected_attributes_to_merge:
            self.assertTrue((self.step_size_reduction_via_pca.get_attributes_to_merge_together()[k] == self.expected_attributes_to_merge[k]).all())
    
    def test_explained_variance_threshold(self):
        self.assertEqual(
            self.step_size_reduction_via_pca.explained_variance_threshold,
            self.expected_explained_variance_threshold,
            msg="Explained variance threshold is not as expected"
        )
    
    def test_n_directions_threshold(self):
        self.assertEqual(
            self.step_size_reduction_via_pca.n_directions_threshold,
            self.expected_n_directions_threshold,
            msg="Number of directions threshold is not as expected"
        )
    
    def test_stage_data_before_size_reduction(self):
        self.assertIsNone(
            self.step_size_reduction_via_pca.stage_data_before_size_reduction,
            msg="Stage data before size reduction is not None"
        )
        self.step_size_reduction_via_pca.stage_data_before_size_reduction = self.old_data
        self.assertTrue(np.allclose(self.step_size_reduction_via_pca.stage_data_before_size_reduction['old_attr1'], self.old_data['old_attr1']))
    
    def test_dimension_before_size_reduction(self):
        self.assertIsNone(
            self.step_size_reduction_via_pca.dimension_before_size_reduction,
            msg="Dimension before size reduction is not None"
        )
        self.step_size_reduction_via_pca.stage_data_before_size_reduction = self.old_data
        self.step_size_reduction_via_pca.dimension_before_size_reduction = len(self.old_data['old_attr1'])
        self.assertEqual(
            self.step_size_reduction_via_pca.dimension_before_size_reduction,
            2,
            msg="Dimension before size reduction is not as expected"
        )
        
    def test_init(self):
        # Test that an error is raised if no 'explained variance threshold' is provided for an attribute
        with self.assertRaises(ValueError):
            StepSizeReductionViaPCA(self.step_size_reduction_via_pca.step_index, self.step_size_reduction_via_pca.attributes_to_merge_together, {}, self.step_size_reduction_via_pca.n_directions_threshold)
        
        # Test that an error is raised if no 'number of directions threshold' is provided for an attribute
        with self.assertRaises(ValueError):
            StepSizeReductionViaPCA(self.step_size_reduction_via_pca.step_index, self.step_size_reduction_via_pca.attributes_to_merge_together, self.step_size_reduction_via_pca.explained_variance_threshold, {})
        
        # Test that the class variables are initialized correctly
        step = StepSizeReductionViaPCA(self.step_size_reduction_via_pca.step_index, self.step_size_reduction_via_pca.attributes_to_merge_together, self.step_size_reduction_via_pca.explained_variance_threshold, self.step_size_reduction_via_pca.n_directions_threshold)
        self.assertEqual(step.step_index, self.step_size_reduction_via_pca.step_index)
        for k in step.attributes_to_merge_together:
            self.assertTrue((step.attributes_to_merge_together[k] == self.step_size_reduction_via_pca.attributes_to_merge_together[k]).all())
        self.assertDictEqual(step.explained_variance_threshold, {"new_attr": 1.0e-4})
        self.assertDictEqual(step.n_directions_threshold, {"new_attr": 2})
        self.assertIsNone(step.n_directions_retained)
        self.assertIsNone(step.pca)
        self.assertIsNone(step.get_trunc_pca_basis_from_original_data)
        self.assertIsNone(step.get_original_data_from_trunc_pca_basis)
        self.assertIsNone(step.linear_transformation_input_to_trunc_pca_basis)
        self.assertIsNone(step.linear_transformation_trunc_pca_to_input_basis)
        self.assertIsNone(step.stage_data_before_size_reduction)
        self.assertIsNone(step.dimension_before_size_reduction)
        self.assertIsNone(step.dimension_before_size_reduction_sum)
        self.assertIsNone(step.stage_data_after_size_reduction)
        self.assertIsNone(step.dimension_after_size_reduction)
        
if __name__ == '__main__':
    unittest.main()
