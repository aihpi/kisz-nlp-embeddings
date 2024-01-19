import pytest

import numpy as np
from src.plotting import Plotter

class TestPlotter:
    @pytest.fixture(scope='class')
    def example_vectors(self):
        return [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    
    def test_empty_vectors(self):
        with pytest.raises(ValueError, match=r"The list of vectors is empty"):
            plotter = Plotter()
            plotter.euclid_dist([])

    def test_single_vector(self):
        with pytest.raises(ValueError, match=r"The list must contain at least 2 vectors"):
            plotter = Plotter()
            plotter.euclid_dist([np.array([1, 2])])

    def test_non_numpy_vector(self):
        with pytest.raises(TypeError, match=r".* not a numpy array .*"):
            plotter = Plotter()
            plotter.euclid_dist([np.array([1, 2]), [3, 4]])