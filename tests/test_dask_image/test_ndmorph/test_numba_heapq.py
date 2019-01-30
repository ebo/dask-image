#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

from numba import jit, double, float64, int64, typeof # int32, int8,

from dask_image.ndmorph import _numba_heapq as nh


# numba cannot guess the type/fingerprint of the heap.
def test_heap_initialization_err():
    hp = []

    with pytest.raises(ValueError):
        ret = nh.heappushpop(hp, (1,2,3,4))

# One solution is to seed the heap with an item
def test_heap_initialization():
    hp = [(0,0,0,0)]

    nh.heappush(hp, (1,2,3,4))

    ret = nh.heappushpop(hp, (5,6,7,8))
    assert (0,0,0,0) == ret

    ret = nh.heappop(hp)
    assert (1,2,3,4) == ret

# FIXME: for full coverage we need tests for
#    heapreplace, heapify, nlargest, nsmallest, and merge

