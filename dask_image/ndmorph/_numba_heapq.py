# -*- coding: utf-8 -*-

# the following implemenetation is a trivial port to utilize numba in
# segmentation code which uses heapq

from numba import jit, double, float64, int32, int8, void

__all__ = ['heappush', 'heappop', 'heapify', 'heapreplace', 'merge',
           'nlargest', 'nsmallest', 'heappushpop']

@jit(nopython=True)
def heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)

@jit(nopython=True)
def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
    else:
        returnitem = lastelt
    return returnitem

@jit(nopython=True)
def heapreplace(heap, item):
    """Pop and return the current smallest value, and add the new item.

    This is more efficient than heappop() followed by heappush(), and can be
    more appropriate when using a fixed-size heap.  Note that the value
    returned may be larger than item!  That constrains reasonable uses of
    this routine unless written as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)
    """
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup(heap, 0)
    return returnitem

@jit(nopython=True)
def heappushpop(heap, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heap[0] < item:
        item, heap[0] = heap[0], item
        _siftup(heap, 0)
    return item

@jit(nopython=True)
def heapify(x):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n//2)):
        _siftup(x, i)

@jit(nopython=True)
def _heappushpop_max(heap, item):
    """Maxheap version of a heappush followed by a heappop."""
    if heap and item < heap[0]:
        item, heap[0] = heap[0], item
        _siftup_max(heap, 0)
    return item

@jit(nopython=True)
def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

@jit(nopython=True)
def nlargest(n, iterable):
    """Find the n largest elements in a dataset.

    Equivalent to:  sorted(iterable, reverse=True)[:n]
    """
    from itertools import islice, count, tee, chain
    if n < 0:
        return []
    it = iter(iterable)
    result = list(islice(it, n))
    if not result:
        return result
    heapify(result)
    _heappushpop = heappushpop
    for elem in it:
        _heappushpop(result, elem)
    result.sort(reverse=True)
    return result

@jit(nopython=True)
def nsmallest(n, iterable):
    """Find the n smallest elements in a dataset.

    Equivalent to:  sorted(iterable)[:n]
    """
    from itertools import islice, count, tee, chain
    if n < 0:
        return []
    it = iter(iterable)
    result = list(islice(it, n))
    if not result:
        return result
    _heapify_max(result)
    _heappushpop = _heappushpop_max
    for elem in it:
        _heappushpop(result, elem)
    result.sort()
    return result

# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
@jit(nopython=True)
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

# The child indices of heap index pos are already heaps, and we want to make
# a heap at index pos too.  We do this by bubbling the smaller child of
# pos up (and so on with that child's children, etc) until hitting a leaf,
# then using _siftdown to move the oddball originally at index pos into place.
#
# We *could* break out of the loop as soon as we find a pos where newitem <=
# both its children, but turns out that's not a good idea, and despite that
# many books write the algorithm that way.  During a heap pop, the last array
# element is sifted in, and that tends to be large, so that comparing it
# against values starting from the root usually doesn't pay (= usually doesn't
# get us out of the loop early).  See Knuth, Volume 3, where this is
# explained and quantified in an exercise.
#
# Cutting the # of comparisons is important, since these routines have no
# way to extract "the priority" from an array element, so that intelligence
# is likely to be hiding in custom comparison methods, or in array elements
# storing (priority, record) tuples.  Comparisons are thus potentially
# expensive.
#
# On random arrays of length 1000, making this change cut the number of
# comparisons made by heapify() a little, and those made by exhaustive
# heappop() a lot, in accord with theory.  Here are typical results from 3
# runs (3 just to demonstrate how small the variance is):
#
# Compares needed by heapify     Compares needed by 1000 heappops
# --------------------------     --------------------------------
# 1837 cut to 1663               14996 cut to 8680
# 1855 cut to 1659               14966 cut to 8678
# 1847 cut to 1660               15024 cut to 8703
#
# Building the heap by using heappush() 1000 times instead required
# 2198, 2148, and 2219 compares:  heapify() is more efficient, when
# you can use it.
#
# The total compares needed by list.sort() on the same lists were 8627,
# 8627, and 8632 (this should be compared to the sum of heapify() and
# heappop() compares):  list.sort() is (unsurprisingly!) more efficient
# for sorting.

@jit(nopython=True)
def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)

@jit(nopython=True)
def _siftdown_max(heap, startpos, pos):
    'Maxheap variant of _siftdown'
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

@jit(nopython=True)
def _siftup_max(heap, pos):
    'Maxheap variant of _siftup'
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the larger child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of larger child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos] < heap[childpos]:
            childpos = rightpos
        # Move the larger child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown_max(heap, startpos, pos)

# If available, use C implementation
#try:
#    from _heapq import *
#except ImportError:
#    pass

@jit(nopython=True)
def merge(*iterables):
    '''Merge multiple sorted inputs into a single sorted output.

    Similar to sorted(itertools.chain(*iterables)) but returns a generator,
    does not pull the data into memory all at once, and assumes that each of
    the input streams is already sorted (smallest to largest).

    >>> list(merge([1,3,5,7], [0,2,4,8], [5,10,15,20], [], [25]))
    [0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]

    '''
    _heappop, _heapreplace, _StopIteration = heappop, heapreplace, StopIteration
    _len = len

    h = []
    h_append = h.append
    for itnum, it in enumerate(map(iter, iterables)):
        try:
            next = it.__next__
            h_append([next(), itnum, next])
        except _StopIteration:
            pass
    heapify(h)

    while _len(h) > 1:
        try:
            while True:
                v, itnum, next = s = h[0]
                yield v
                s[0] = next()               # raises StopIteration when exhausted
                _heapreplace(h, s)          # restore heap condition
        except _StopIteration:
            _heappop(h)                     # remove empty iterator
    if h:
        # fast case when only a single iterator remains
        v, itnum, next = h[0]
        yield v
        yield from next.__self__

# Extend the implementations of nsmallest and nlargest to use a key= argument
_nsmallest = nsmallest
@jit(nopython=True)
def nsmallest(n, iterable, key=None):
    """Find the n smallest elements in a dataset.

    Equivalent to:  sorted(iterable, key=key)[:n]
    """
    from itertools import islice, count, tee, chain
    # Short-cut for n==1 is to use min() when len(iterable)>0
    if n == 1:
        it = iter(iterable)
        head = list(islice(it, 1))
        if not head:
            return []
        if key is None:
            return [min(chain(head, it))]
        return [min(chain(head, it), key=key)]

    # When n>=size, it's faster to use sorted()
    try:
        size = len(iterable)
    except (TypeError, AttributeError):
        pass
    else:
        if n >= size:
            return sorted(iterable, key=key)[:n]

    # When key is none, use simpler decoration
    if key is None:
        it = zip(iterable, count())                         # decorate
        result = _nsmallest(n, it)
        return [r[0] for r in result]                       # undecorate

    # General case, slowest method
    in1, in2 = tee(iterable)
    it = zip(map(key, in1), count(), in2)                   # decorate
    result = _nsmallest(n, it)
    return [r[2] for r in result]                           # undecorate

_nlargest = nlargest
@jit(nopython=True)
def nlargest(n, iterable, key=None):
    """Find the n largest elements in a dataset.

    Equivalent to:  sorted(iterable, key=key, reverse=True)[:n]
    """

    from itertools import islice, count, tee, chain
    # Short-cut for n==1 is to use max() when len(iterable)>0
    if n == 1:
        it = iter(iterable)
        head = list(islice(it, 1))
        if not head:
            return []
        if key is None:
            return [max(chain(head, it))]
        return [max(chain(head, it), key=key)]

    # When n>=size, it's faster to use sorted()
    try:
        size = len(iterable)
    except (TypeError, AttributeError):
        pass
    else:
        if n >= size:
            return sorted(iterable, key=key, reverse=True)[:n]

    # When key is none, use simpler decoration
    if key is None:
        it = zip(iterable, count(0,-1))                     # decorate
        result = _nlargest(n, it)
        return [r[0] for r in result]                       # undecorate

    # General case, slowest method
    in1, in2 = tee(iterable)
    it = zip(map(key, in1), count(0,-1), in2)               # decorate
    result = _nlargest(n, it)
    return [r[2] for r in result]                           # undecorate

#if __name__ == "__main__":
@jit(nopython=True)
def test_it():
    # Simple sanity test

    index = 1
    elem_value = 123
    elem_age = 0
    elem_index = index
    elem_source = index
    #heappush(hp, elem.copy())
    elem = (elem_value,elem_age,elem_index,elem_source)



    
    heap = [elem]
    data = [(3,1,0,1), (5,1,0,1), (7,1,0,1), (9,1,0,1), (2,1,0,1), (4,1,0,1), (6,1,0,1), (8,1,0,1), (0,1,0,1)]
    for item in data:
        heappush(heap, item)
    sort = []
    while heap:
        sort.append(heappop(heap))
    print(sort)

    #import doctest
    #doctest.testmod()

#test_it()
