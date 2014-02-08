#!/usr/bin/env python
import math, os

""" range min query problem RMQ[2,7] = 3
find the index of min value of an array range[i,j]
http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=lowestCommonAncestor

Range -> 1..n -> make it into a heap with node val = the min in the range.

Given a list, build a heap with heap[i] indicates a segment [0..i] range on top of the list
heap[0] = list[0:sz], heap[1] = list[0:mid], heap[2]=list[mid:sz], etc.

make a segment tree, each tree node covers an idx range of list, i.e, a list segment.
val of node is the min of its list range under heap.
As we segment idx, idx are 1..n, segment tree is complete bin tree. min val sift up to root.
root covers [0..len-1], left covers [0..mid], right covers [mid+1, len-1], until to the leaf.
the value of each node is the min of the range. value in root is the min of entire array.
To query, recursively query left children, right children, and choose the min of them.

the idx of each segment node in the heap[] is determined by the range it covers.
root a 0, left half(left child), at root*2, right half(right child) at idx*2+1, recursively build.

To build the heap whose idx maps a segment in the list;
in DP fn, we need heap idx, range start and end. fill bottom case and start using DP while building it.
top down to bottom, then recursively build up by ret value picked when recursive down at the leaf

more use case: reduce LCA problem to RMQ problem
"""

class Node(object):
    def __init__(self, heapidx, range_start, range_end, minval=0, minvalidx=0):
        super(Node, self).__init__()
        self.heapidx = heapidx
        self.range_start = range_start  # the start idx of orig array this seg node covers
        self.range_end = range_end  # the end idx of orig array this seg node covers
        self.minval = minval        # the min val under this segment tree range
        self.minvalidx = minvalidx

    def toString(self):
        return '[ ' + str(self.heapidx) + ' ' + str(self.range_start) + '-' + str(self.range_end) + ':' + str(self.minval) + ' ]'

    def setMin(self, minval, minvalidx):
        self.minval = minval
        self.minvalidx = minvalidx

class Heap(object):
    def __init__(self, size):
        super(Heap, self).__init__()
        self.size = size
        self.l = []   # a list of Nodes
        self.l = [Node(-1,-1,-1)]*size

    def set(self, idx, n):
        self.l[idx] = n

    def swap(self, i, j):
        self.l[i], self.l[j] = self.l[j], self.l[i]

    def toString(self):
        strv = ' '
        for i in self.l:
            strv += self.l.toString()
            strv += ' '
        print strv
        return strv

    def dumpRangeHeap(self):
        for n in self.l:
            if n.heapidx >= 0:
                print n.toString()

    def head(self):
        return self.l[0]

    def parent(self, i):
        if i > 0:
            return (i-1)/2, self.l[(i-1)/2]
        else:
            return None, None
    def lchild(self, i): # index from 0, l=2x+1, r=2x+2
        if i*2+1 < self.size:
            return i*2+1, self.l[i*2+1]
        else:
            return None, None
    def rchild(self, i):
        if i*2+2 < self.size:
            return i*2+2, self.l[i*2+2]
        else:
            return None, None

class RMQ(object):
    def __init__(self, l=None):
        super(RMQ, self).__init__()
        self.l = l
        self.heap = None  # the range heap

    def setArray(self, l):
        self.l = l
        self.size = len(l)
        height = int(math.log(self.size, 2))+1
        self.heap = Heap(pow(2, 2*height))  # the last row has n elements, total nodes=2^logn+1

    ''' top down to bottom, then recursively build up by ret value picked when recursive down at the leaf
        the rmq heap array len is bigger than origin array as only the leaf node of rmq array=origin array.
        recursive build api bin div origin array seg[beg, end] and assign the min to rmq array at idx.
        here we build a heap tree, each node's idx maps to a range in the list.
        To build the heap whose idx maps a segment in the list;
        in DP fn, we need heap idx, range start and end. fill bottom case and start using DP while building it.
    '''
    def buildSegmentTree(self, heapidx, range_start, range_end):
        # always create a node with the passed in idx when building tree bottom up.
        curnode = Node(heapidx, range_start, range_end)
        self.heap.set(heapidx, curnode)

        if range_start == range_end:
            # top-down reach the bottom, each leaf node covers one list item
            curnode.setMin(self.l[range_start], range_start)
            return self.l[range_start], range_start
        else:
            # internal node, min of the lchild and rchild.
            mid = (range_start+range_end)/2
            # top-down to current node's lchild and rchild
            lminv, lminidx = self.buildSegmentTree(heapidx*2+1, range_start, mid)
            rminv, rminidx = self.buildSegmentTree(heapidx*2+2, mid+1, range_end)
            if(lminv <= rminv):
                curnode.setMin(lminv, lminidx)
                return lminv, lminidx
            else:
                curnode.setMin(rminv, rminidx)
                return rminv, rminidx

    ''' if interval [i..j] covered by one seg tree node, ret seg tree node.
        if interval acros seg tree node covers, split, recursive search lrchild with narrowed range.
    '''
    def rangeMin(self, node, i, j):  # search this node, means search lrchild of the node.
        self.heap.dumpRangeHeap()
        print 'range min', i, j, self.l

        ''' this segment node non-related'''
        if i > node.range_end or j < node.range_start:
            return -1, -1

        ''' this node's cover within i,j range, no need to explore its children
            at node [3,4], if search ij is 3,7, can ret; if search ij is 4-7, cont on rchild
        '''
        if i <= node.range_start and j >= node.range_end:
            return node.minval, node.minvalidx

        ''' otherwise, recursively search left/right children, bubble up min'''
        lidx, lnode = self.heap.lchild(node.heapidx)
        ridx, rnode = self.heap.rchild(node.heapidx)
        if lidx and lidx >= 0:  # if lchild exist
            lminv, lminidx = self.rangeMin(lnode, i, j)
        if ridx and ridx >= 0:
            rminv, rminidx = self.rangeMin(rnode, i, j)

        if lminv == -1:
            return rminv, rminidx
        elif rminv == -1:
            return lminv, lminidx
        else:
            return min(lminv, rminv), lminidx if lminv <= rminv else rminidx

    def test(self):
        l= [2,4,3,1,6,7,8,9,1,7]
        self.setArray(l)
        # build segment tree from root, idx 0, size [0..len(l)]
        self.buildSegmentTree(0, 0, len(l)-1)
        minv, minvidx = self.rangeMin(self.heap.head(), 2,7)
        print 'min 2-7 :', minv, minvidx
        minv, minvidx = self.rangeMin(self.heap.head(), 4,7)
        print 'min 4-7 :', minv, minvidx


"""
Josephus_problem, remove Kth item from a circle of n items.
the main point is remap the index after kth removed, and restart another round of recursion.
   f(n,k) = (f(n-1,k) + k) mod n, where f(1,k) = 0
"""
def Josephus(n, k):
    if n is 1:
        return 1
    else:
        return (Josephus(n-1, k)+k-1) % n) + 1

def Josephus(n, k):
    r = 0   # index of of k
    i = 2

    while i <= n:
        r = (r+k) % i
        i += 1
    return r+1


if __name__ == '__main__':
    rmq = RMQ()
    rmq.test()
