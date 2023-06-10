# pca_size_reduction.py 

A module that implements a size reduction of the data with PCA.

## Input:

The size reduction via PCA takes as inputs the following attributes:

- `a_original_data`: a original data in the form of a dictionary where the keys are the attributes and the values are Numpy arrays containing the data (each row corresponds to a data point) 
- `a_attributes_to_merge_together`: a vector with the details on the attributes to merge together for each step of the size reduction via PCA. Each value is a dictionary, where the key is the new attribute generated and the values are the attributes that are merged together to create this new attribute.
- `a_explained_variance_threshold`: a vector with the details on the explained variance thresholds when creating the new attributes for each step of the size reduction via PCA. Each value is a dictionary, where the key is the new attribute generated and the value is the explained variance threshold.
- `a_n_directions_threshold`: a vector with the details on the maximum number of PCA components to keep when creating the new attributes for each step of the size reduction via PCA. Each value is a dictionary, where the key is the new attribute generated and the value is the  maximum number of PCA components to keep.

## Functions:

- `get_linear_transformations_between_original_and_standardized_data(a_data)`: Computes the linear transformations to standardize and unstandardize the data.

## Classes:

- `StageData`: Stores data, stage index, and a function to project data back to the original data dimension and scale.

	The class has three methods:
	
	- `init(self, a_stage, a_data, a_get_data_in_original_basis)`: is the constructor method that initializes the instance with the provided stage, data, and function to get the data in the original basis.

		- `a_stage` (int): is the stage corresponding to the data
		- `a_data` (dict): is the data in the standardized basis. It is a dictionary where each key represents a feature and each value is an array of standardized data points.
		- `a_get_data_in_original_basis`: is a function that returns the data in the original basis.
	
	- `get_data(self)`: returns the data stored in the instance.
	
	- `get_stage(self)`: returns the stage stored in the instance.

- `StepSizeReductionViaPCA`:
	
	Implements a step of size reduction via PCA.
	It has the following attributes:
    - `step_index`: the step index.
    - `attributes_to_merge_together`: a dictionary where the keys correspond to the new attributes and the values are the attributes to merge together sorted alphabetically.
    - `explained_variance_threshold`: a dictionary where the keys correspond to the new attributes and the values are the explained variance thresholds for each new attribute.
    - `n_directions_threshold`: a dictionary where the keys correspond to the new attributes and the values are the maximum number of directions to retain in the PCA basis for each new attribute.
    - `n_directions_retained`: the number of directions retained in the PCA basis for each new attribute.
    - `pca`: a dictionary that contains the PCA for each attribute.
    - `get_trunc_pca_basis_from_original_data()`: a function to go from the original basis to the truncated PCA basis.
    - `get_original_data_from_trunc_pca_basis()`: a function to go from the truncated PCA basis to the original basis.
    - `linear_transformation_input_to_trunc_pca_basis()`: a function to map from the input to the truncated PCA basis.
    - `linear_transformation_trunc_pca_to_input_basis()`: a function to map from the truncated PCA basis to the input basis.
    - `stage_data_before_size_reduction`: data before the size reduction via PCA.
    - `dimension_before_size_reduction`: the original dimension.
    - `dimension_before_size_reduction_sum`: the sum of the original dimensions.
    - `stage_data_after_size_reduction`: data after the size reduction via PCA.
    - `dimension_after_size_reduction`: the new dimension.

- `SizeReductionViaPCA`: Implements the size reduction via PCA. It has the following attributes:
    - `original_data`: the original data.
    - `n_steps`: the number of steps to execute for the data size reduction.
    - `attributes_to_merge_together`: a list of dictionaries where each dictionary has the new attributes and the attributes to merge together for each step.
    - `attributes`: a list of the attributes at each stage.
    - `explained_variance_threshold`: a list of dictionaries where each dictionary has the new attributes and the explained variance threshold for each step.
    - `n_directions_threshold`: a list of dictionaries where each dictionary has the new attributes and the maximum number of PCA directions to keep for each step.
    - `steps`: a list of StepSizeReductionViaPCA objects, each corresponding to a step of size reduction via PCA.
    - `stage_data`: a list of StageData objects, each corresponding to a stage of size reduction via PCA.

## Examples:
- Example usage can be found in the file `main.py`.

## Notes:
- This module requires the following packages to be installed: `numpy`, `pandas`, and `sklearn`.
