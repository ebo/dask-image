#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

from numba import jit, double, float64, int64, typeof
import numpy as np

from dask_image.ndmorph import _numba_watershed as nw


def test_watershed_seeded():
    """Test that passing just the number of seeds to watershed works."""
    image = np.zeros((5, 6))
    image[:, 3:] = 1
    compact = nw.watershed(image, 2, compactness=0.01)

    expected = np.array([[1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2]], dtype=np.int64)

    np.testing.assert_equal(compact, expected)

def test_incorrect_markers_shape():
    with pytest.raises(ValueError):
        image = np.ones((5, 6))
        markers = np.ones((5, 7))
        output = nw.watershed(image, markers)

def test_compact_watershed():
    from skimage.morphology import watershed
    image = np.zeros((5, 6))
    image[:, 3:] = 1
    seeds = np.zeros((5, 6), dtype=int)
    seeds[2, 0] = 1
    seeds[2, 3] = 2
    compact = nw.watershed(image, seeds, compactness=0.01)
    expected = np.array([[1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2]], dtype=int)
    np.testing.assert_equal(compact, expected)
    normal = watershed(image, seeds)
    expected = np.ones(image.shape, dtype=int)
    expected[2, 3:] = 2
    np.testing.assert_equal(normal, expected)

