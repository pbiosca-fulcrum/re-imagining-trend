"""
Minimal example test file to show how you might test a small part of the project.
Run these with:  pytest  or python -m unittest discover
"""

import unittest
import numpy as np

from src.utils.utilities import rank_normalization

class TestUtilities(unittest.TestCase):

    def test_rank_normalization(self):
        series = np.array([1, 2, 5, 10, 9, 3], dtype=float)
        # Some basic check
        # convert to a pd.Series for rank_normalization usage
        import pandas as pd
        s = pd.Series(series)
        r = rank_normalization(s)
        self.assertEqual(len(r), len(series))
        self.assertAlmostEqual(r.min(), -1.0, places=5)
        self.assertAlmostEqual(r.max(), 1.0, places=5)

if __name__ == "__main__":
    unittest.main()
