#!/usr/bin/env python

import sys

class PQNode:
    def __init__(self, k, v=None):
        self.index = 0
        self.key = k
        self.value = v

    # comparable IF, using key
    def compare(self, other):
        if self.key < other.key:
            return -1
        elif self.key == other.key:
            return 0
        else:
            return 1

''' PriQ is a just min heap '''
class PriQueue(object):
    # no cls variable

    # no.1, constructor
    def __init__(self, size=100, reverse=False):
        __maxsize = size
        self.reverse = reverse
        self.__q = []

    def toString(self):
        print self.__q

    def swap(self, i, j):
        self.__q[i], self.__q[j] = self.__q[j], self.__q[i]

    def siftup(self, root, end):
        child = end
        while child >= root :
            parent = (child - 1) / 2
            # swap when parent is bigger, min heap
            if parent >= root and self.__q[parent].key > self.__q[child].key:
                self.swap(parent, child)
                child = parent
            else:
                return

    def siftdown(self, start, end):
        root = start
        while root <= end:
            lchild = root * 2 + 1
            rchild = root * 2 + 2
            maxkey = self.__q[root].key
            maxnode = root
            if lchild <= end and maxkey > self.__q[lchild].key:
                maxkey = self.__q[lchild].key
                maxnode = lchild
            if rchild <= end and maxkey > self.__q[rchild].key:
                maxkey = self.__q[rchild].key
                maxnode = rchild
            if root != maxnode:
                self.swap(root, maxnode)
                root = maxnode
            else:
                return

    def heapify(self):
        ''' nlgn, start from mid, which is the level just above leaf, level-- back to root'''
        m = len(self.__q) / 2
        while m >= 0:  # =0, incl 0 into the logic
            self.siftdown(m, len(self.__q) - 1)
            m = m - 1

    ''' O(2n) linear, reccursive divide to two subtree'''
    def bubble_down(self, l, i):
        minchild = self.minChild(l, i)
        self.swap(i, minchild)
        self.bubble_down(l, minchild)

    def makeHeap(self):
        for i in xrange(len(l)-1, -1, -1):
            self.bubble_down(l, i)

    ''' insert into heap just append to the end, and softup'''
    def insert(self, PQNode):
        self.__q.append(PQNode)
        size = len(self.__q)
        self.siftup(0, len(self.__q) - 1)
        #print self.__q


    ''' extract extreme from head, swap end to head, and siftdown '''
    def extract(self):
        if not len(self.__q):
            return None
        max = self.__q[0]
        self.swap(0, len(self.__q) - 1)
        self.__q.pop()
        self.siftdown(0, len(self.__q) - 1)
        return max

    def findNodeByValue(self, val):
        # how do we find a node in the heap ?
        # have to go sequential ?
        for i in xrange(len(self.__q)):
            if __q[i].val == val:
                return __q[i]
        return None

    ''' found the kth small ele in a min-heap is greater than x in O(k)
        always recursively divide in parallel in lchild and rchild
    '''
    def compareHeap(self, l, i, cnt, x):
        if cnt < 0 or i>len(l):
            return cnt
        if l[i] < x:
            cnt = self.compareHeap(l, 2*i, cnt-1, x)
            cnt = self.compareHeap(l, 2*i+1, cnt, x)

    def test(self):
        self.insert(PQNode(4))
        self.insert(PQNode(6))
        self.insert(PQNode(3))
        self.insert(PQNode(13))
        self.insert(PQNode(9))
        self.insert(PQNode(2))
        self.insert(PQNode(5))
        self.insert(PQNode(7))

        self.heapify()

        #self.toString()
        n = self.extract()
        while n:
            print n.key
            n = self.extract()

""" this class impl a min heap
"""
class PQ(object):
    def __init__(self, size, initval=10000):
        super(PQ, self).__init__()
        self.size = size
        self._q = [initval]*size

    def toString(self):
        print self._q

    def swap(self, i, j):
        self._q[i], self._q[j] = self._q[j], self._q[i]

    def siftup(self, idx):
        while idx >= 0:
            parent = (idx-1)/2  # [0,1,2 ... 1,3,4...]
            if parent >= 0 and self._q[parent] < self._q[idx]:
                self.swap(parent, idx)
                idx = parent
            else:
                return

    def siftdown(self, idx):
        tmpidx = idx

        while tmpidx < self.size:
            lc = 2*tmpidx+1
            rc = 2*(tmpidx+1)
            max = self._q[tmpidx]
            maxidx = tmpidx

            if lc < self.size and self._q[lc] >= max:
                max = self._q[lc]
                maxidx = lc
            if rc < self.size and self._q[rc] > max:
                max = self._q[rc]
                maxidx = rc
            if maxidx != tmpidx:
                self.swap(tmpidx, maxidx)
                tmpidx = maxidx
            else:
                return  # nothing to do, done

    def extract(self):
        self.swap(0, self.size-1)
        self.siftdown(0)

    def insert(self, val):
        if val < self._q[0]:
            self._q[0] = val
            self.siftdown(0)

    def test(self):
        for i in xrange(10):
            self.insert(2*(i+1))

        for i in xrange(5,1, -1):
            self.insert(2*i-1)

""" 
Heap base, impl common func for lchild, rchild, parent, head, etc.
To impl increase-key/decrease-key, need a map of value to its idx in the heap.
    on Swap(i, j): map[value[i]] = j; map[value[j]] = i;
    on Insert(key, value): map.Add(value, heapSize) in the beginning;
    on ExtractMin: map.Remove(extractedValue)
    on UpdateKey(value, newKey): index = map[value]; keys[index] = newKey; BubbleUp(index) in case of DecreaseKey or BubbleDown/Heapify(index) in case of IncreaseKey to restore min-heap-property.
"""
class Heap(object):
    def __init__(self, size, initval):
        super(Heap, self).__init__()
        self.size = size
        self.l = [initval]*size
        self.idxMap = {}

    # swap is called when siftup and siftdown, every change of node in heap go thru swap.
    def swap(self, i, j):
        self.l[i], self.l[j] = self.l[j], self.l[i]
        self.idxMap[self.l[i]] = j
        self.idxMap[self.l[j]] = i

    def toString(self):
        print self.l

    def head(self):
        return self.l[0]
    def parent(self, i):
        if i > 0:
            return (i-1)/2, self.l[(i-1)/2]
        else:
            return None, None
    def lchild(self, i):
        if i*2+1 < self.size:
            return i*2+1, self.l[i*2+1]
        else:
            return None, None
    def rchild(self, i):
        if i*2+2 < self.size:
            return i*2+2, self.l[i*2+2]
        else:
            return None, None

    def insert(self, v):
        self.l.append(v)
        self.size = len(self.l)
        idx = self.size
        self.idxMap[v] = idx
        self.siftup(idx)

    def update(self, oldv, newv):
        ''' update the value of old value to the new value '''
        idx = self.idxMap[oldv]
        self.l[idx] = newv
        self.idxMap[newv] = idx
        self.siftup(idx)   
        # self.siftdown(idx)   # in case of decrease key.

    def extractMin(self):
        val = self.l[0]
        self.idxMap[val] = None

    def siftdown(self, idx, end):   # sift down header
        while idx <= end:
            maxval = self.l[idx]   # maxval heap, parent > all children
            maxidx = idx
            lidx, lval = self.lchild(idx)
            ridx, rval = self.rchild(idx)

            if lval is not None and lval > maxval:
                maxidx, maxval = lidx, lval
            if rval is not None and rval > maxval:
                maxidx, maxval = ridx, rval
            if  maxidx == idx:
                break
            else:
                self.swap(idx, maxidx)
                idx = maxidx

    def siftup(self, idx):
        while idx >= 0:
            parent = (idx-1)/2  # [0,1,2 ... 1,3,4...]
            if parent >= 0 and self.l[parent] < self.l[idx]:
                self.swap(parent, idx)
                idx = parent
            else:
                return


""" this class impl a max heap of size K.
    The head is the max of all eles in the heap
    max k-heap to maintain smallest k eles
    Init max heap with sys.maxint so the max can be pop up.
"""
class MaxHeap(Heap):
    def __init__(self, size):
        super(MaxHeap, self).__init__(size, sys.maxint)

    def siftdown(self, idx):   # sift down header
        while idx < self.size:  # idx less than size
            max = self.l[idx]   # max heap, parent > all children
            maxidx = idx
            lidx, lval = self.lchild(idx)
            ridx, rval = self.rchild(idx)

            if lval is not None and lval > max:
                maxidx, max = lidx, lval
            if rval is not None and rval > max:
                maxidx, max = ridx, rval
            if  maxidx == idx:
                break
            else:
                self.swap(idx, maxidx)
                idx = maxidx

    def extract(self, val):
        if self.l[0] > val:
            self.l[0] = val
            self.siftdown(0)

    def test(self):
        l = [3,4,5,15,19,20,25]
        step = 1
        for e in l:
            self.extract(e)
        self.toString()
        print '-------'

        max = self.head()
        step = 2
        for i in xrange(len(l)):
            if l[i] > self.head():
                break;
            for j in xrange(i+1, len(l)):
                if l[i] + l[j] < self.head():
                    self.extract(l[i]+l[j])
                    j += 1
                    continue
                else:
                    break
        self.toString()

        step = 3
        for i in xrange(len(l)):
            if l[i] > self.head():
                break
            for j in xrange(i+1, len(l)):
                if l[i] + l[j] > self.head():
                    break
                for k in xrange(j+1, len(l)):
                    if l[i]+l[j]+l[k] < self.head():
                        self.extract(l[i]+l[j]+l[k])
                        continue
                    else:
                        break
        self.toString()

""" this class impl a min heap of size K.
    The head is the min of all eles in the heap
    min k-heap to maintain largest k eles
    Init min heap with -sys.maxint so the min can be pop up.
"""
class MinHeap(Heap):
    def __init__(self, size):
        super(MinHeap, self).__init__(size, -sys.maxint)

    def siftdown(self, idx):   # sift down header
        while idx < self.size:
            min = self.l[idx]
            minidx = idx
            lidx, lval = self.lchild(idx)
            ridx, rval = self.rchild(idx)

            if lval is not None and lval < min:
                minidx, min = lidx, lval
            if rval is not None and rval < min:
                minidx, min = ridx, rval
            if  minidx == idx:
                break
            else:
                self.swap(idx, minidx)
                idx = minidx

    def extract(self, val):
        if self.l[0] < val:
            self.l[0] = val
            self.siftdown(0)

    def test(self):
        l = [3,4,5,15,19,20,25]
        step = 1
        for e in l:
            self.extract(e)
        self.toString()
        print '-------'


"""
  leetcode sliding window : http://leetcode.com/2011/01/sliding-window-maximum.html
  for k ele in sliding window, it is a max heap. when we slide window, add new ele
  from right window edge into heap. How do we remove the left edge ele in the heap ?
  As heap does not support find ele, the key here is, you can leave it in the heap and
  lazy remove all of slided out node when they pops to the top of max heap after the max at top is 
  removed when slided out.
  Each ele in heap has both val and idx, idx is used to know whether ele inside window.
"""
def max_sliding_win(l, w):
    q = MaxHeap(w)
    q.insert([l[i], i]) for i in l[:w]
    maxW[] = 0

    for right_edge in xrange(w, len(l)):
        maxnode = q.top()
        maxW[right_edge - w] = maxnode[0]
        # if max node is gonna be slided out, clear all top node
        while maxnode[1] <= right_edge - w:  # not within right-d
            q.pop()
            maxnode = q.top()
        q.push(l[right_edge], right_edge)

""" use a deq to maintain deq with largest in front, and desc order.
    when right-edge get a new node, from back of deq, all smaller can be poped out.
    this way, we maintain a desc deq with max at head.
    each node in and out deq only once, hence O(n)
"""
def max_sliding_win_deq(l, w):
    q = deq()
    for i in xrange(w):
        while (!Q.empty() && A[i] >= A[Q.back()])
        Q.pop_back();
    Q.push_back(i);
  
    for i in xrange(w, len(l)):
        B[i-w] = A[Q.front()];
        while (!Q.empty() && A[i] >= A[Q.back()])
            Q.pop_back();
        while (!Q.empty() && Q.front() <= i-w)
            Q.pop_front();
        Q.push_back(i);
    B[n-w] = A[Q.front()];


if __name__ == '__main__':
    q = PriQueue(10)
    q.test()
    maxh = MaxHeap(7)
    maxh.test()
    minh = MinHeap(7)
    minh.test()
