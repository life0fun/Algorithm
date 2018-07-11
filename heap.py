class MaxHeap():
  class Node():
    def __init__(self, val, i, j):  # fix beg and end for ary traverse.
      self.val = val
      self.ij = (i,j)   # the i,j index in the two array
    ''' recursive in-order iterate the tree this node is the root '''
    def inorder(self):
      if self.lchild:
        self.lchild.inorder()
      self.toString()
      if self.rchild:
        self.rchild.inorder()

  def __init__(self):
      self.heap = []
  def toString(self):
      for n in self.heap:
          print n.val, n.ij
      print ' --- heap end ---- '
  def head(self):
      return self.heap[0]
  def lchild(self, idx):
      lc = 2*idx+1
      if lc < len(self.heap):
          return lc, self.heap[lc].val
      return None, None
  def rchild(self, idx):
      rc = 2*idx+2
      if rc < len(self.heap):
          return rc, self.heap[rc].val
      return None, None
  def parent(self, idx):
      if idx > 0:
          return (idx-1)/2, self.heap[(idx-1)/2].val
      return None, None
  def swap(self,i, j):
      self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
  def siftup(self, idx):
      print self.toString()
      while True:
          pidx, pv = self.parent(idx)
          print 'parent ', idx, pidx, pv, self.heap[idx].val
          if pv and pv < self.heap[idx].val:
              self.swap(pidx, idx)
              idx = pidx
          else:
              return
  def siftdown(self, idx):
      while idx < len(self.heap):
          lc, lcv = self.lchild(idx)
          rc, rcv = self.rchild(idx)
          if lcv and rcv:
              maxcv = max(lcv, rcv)
              maxc = lc if maxcv == lcv else rc
          else:
              maxcv = lcv if lcv else rcv
              maxc = lc if lc else rc

          if self.heap[idx].val < maxcv:
              self.swap(idx, maxc)
              idx = maxc
          else:
              return  # done when loop break
  def insert(self, val, i, j):
      print 'inserting ', val, i, j
      node = MaxHeap.Node(val, i, j)
      self.heap.append(node)
      self.siftup(len(self.heap)-1)
  def extract(self):
      head = self.heap[0]
      print 'extract ', head.val, head.ij
      self.swap(0, len(self.heap)-1)
      self.heap.pop()      # remove the end, the extracted hearder, first
      self.siftdown(0)

""" track each key's offset in heap ary thru key2idx, keep santiy during swap """
from collections import defaultdict
class MinHeap(object):
  def __init__(self, mxSize=10):
    self.arr = []
    self.size = 0
    self.mxSize = mxSize
    self.key2idx = defaultdict()   # map a key to its idx in the arry
  def get(self, idx):
    if idx >= 0 and idx < self.size:
      return self.arr[idx][0]
    return None
  # given a value, found its offset/idx in the array
  def getByVal(self, val):
    idx = self.key2idx[val]
    return self.get(idx)
  def lchild(self, idx):
    if idx*2+1 < self.size:
      return idx*2+1
    return None
  def rchild(self, idx):
    if idx*2+2 < self.size:
      return idx*2+2
    return None
  def parent(self,idx):
    if idx > 0:
      return (idx-1)/2
    return None
  def swap(self, src, dst):  # update idx also
    srcv = self.get(src)
    dstv = self.get(dst)
    self.arr[src], self.arr[dst] = [dstv], [srcv]
    self.key2idx[srcv] = dst
    self.key2idx[dstv] = src
  def insert(self, key, val):
    entry = [key, val]
    self.arr.append(entry)
    self.key2idx[key] = self.size
    self.size += 1
    return self.siftup(self.size-1)
  def siftup(self, idx, end=self.size):
    parent = self.parent(idx)
    if parent >= 0 and self.get(parent)[0] > self.get(idx)[0]:
      self.swap(parent, idx)
      return self.siftup(parent)
    return parent
  def siftdown(self, idx, end=self.size):
    l,r = self.lchild(idx), self.rchild(idx)
    minidx = idx
    if l and self.get(l)[0] < self.get(minidx)[0]:
      minidx = l
    if r and self.get(r)[0] < self.get(minidx)[0]:
      minidx = r
    if minidx != idx:
      self.swap(idx, minidx)
      return self.siftdown(minidx)
    return minidx
  def heapify(self):
    mid = (len(self.arr)-1)/2
    while mid >= 0:
      self.siftdown(mid)
      mid -= 1
  def extractMin(self):
    self.swap(0, self.size-1)
    self.size -= 1
    self.siftdown(0)
  def decreaseKey(self, v, newv):
    idx = self.key2idx[v]
    self.arr[idx] = [newv]
    self.siftup(idx)
def testMinHeap():
    h = MinHeap()
    h.insert(5)
    h.insert(4)
    h.insert(9)
    h.insert(3)
    h.decreaseKey(9, 2)
    print h.arr[0][0], h.arr[1][0], h.arr[2][0]


"" bst with lo as the bst search key, augment max to the max of ed of all nodes under root.
    recursive dfs with insert and search. carry parent down during recursvie.
"""
class IntervalTree(object):  TreeMap key is start, and use floorKey and ceilingKey.
  def __init__(self, lo=None, hi=None):
    self.lo = lo
    self.hi = hi
    self.max = max(lo, hi)
    self.left = self.rite = None
  def overlap(self, intv):
    [lo,hi] = intv
    return lo > self.hi or hi < self.lo ? False : True
  def toString(self):
    val = "[ " + str(self.lo) + ":" + str(self.hi) + ":" + str(self.max)
    if self.left:     val += " / :" + self.left.toString()
    if self.rite:     val += " \ :" + self.rite.toString()
    val += " ] "
    return val
  def insert(self, intv):  # lo as BST key
    [lo,hi] = intv
    if not self.lo:  # cur tree, self node is empty, add.
      self.lo = lo
      self.hi = hi
      self.max = max(self.lo, self.hi)
      self.left = self.rite = None
      return self
    #update new max along the path
    if self.max < hi:
      self.max = hi
    if lo < self.lo:
      if not self.left:
        self.left = IntervalTree(lo, hi)
      else:
        self.left = self.left.insert(intv)
      return self.left
    else:
      if not self.rite:
        self.rite = IntervalTree(lo, hi)
      else:
        self.rite = self.rite.insert(intv)
      return self.rite
  def search(self, intv):
    [lo,hi] = intv
    if self.overlap(intv):
      return self
    if self.left and lo <= self.left.max:
      return self.left.search(intv)
    elif self.rite:
      return self.rite.search(intv)
    else:
      return None   
  def riteMin(self):  # ret the min from self's rite, either rite, or leftmost of rite.
    if not self.rite:
      return self
    node = self.rite
    while node.left:
      node = node.left
    return node
  # branch out left/rite to find all nodes that overlap with the intv.
  def dfs(self, intv, result):
    [lo,hi] = intv
    if self.overlap(intv):
      result.add(self)   # update result upon base condition.
    if self.left and lo < self.left.max:
      self.left.dfs(intv, result)
    if self.rite and lo > self.lo or hi > self.riteMin().lo:
      self.rite.dfs(intv, result)
    return result
  def delete(self, intv):
    [lo,hi] = intv
    if lo == self.lo and hi == self.hi:
      if not self.left:
        return self.rite
      elif not self.rite:
        return self.left
      else:
        # find in order successor, leftmost of rite branch.
        node = self.rite
        while node.left:
            node = node.left
        self.lo = node.lo
        self.hi = node.hi
        # recursive delete in order successor
        self.rite = self.rite.delete([node.lo, node.hi])
        self.max = max(self.lo, self.hi, self.left.max)
        if self.rite:
            self.max = max(self.max, self.rite.max)
        return self
    if lo <= self.lo:
        self.left = self.left.delete(intv)
    else:
        self.rite = self.rite.delete(intv)
    # update max after deletion
    self.max = max(self.lo, self.hi)
    if self.left:
        self.max = max(self.left.max, self.max)
    if self.rite:
        self.max = max(self.rite.max, self.max)
    return self
  def inorder(self):
    result = set()
    pre, cur = None, self
    if not cur:
      return cur
    while cur:
      if not cur.left:
        result.add(cur)
        pre,cur = cur, cur.rite
      else:
        node = cur.left
        while node.rite and node.rite != cur:
          node = node.rite
        if not node.rite:
          node.rite = cur
          cur = cur.left    # descend to left, recur
        else:
          result.add(cur)
          pre,cur = cur, cur.rite
          node.rite = None
    return result

''' sort start, loop check stk.top, '''
def maxOverlappingIntervals(arr):
    edarr = sorted(arr, key=lambda x:x[1])
    root = IntervalTree(edarr[0])
    mx = edarr[0][2]
    for i in xrange(1, len(arr)):
        out = []
        root.search(edarr[i], out)
        s = 0
        for oi in xrange(len(out)):
            s += out[oi][2]
        s += edarr[i][2]
        mx = max(mx,s)
        root.insert(edarr[i])
    return mx
print maxIntervals([[1, 6, 100],[2, 3, 200],[5, 7, 400]])

""" merge or toggle interval """
class IntervalArray(object):
  def __init__(self):
      self.arr = []
      self.size = 0
  # bisect ret i as insertion point. arr[i-1] < val <= arr[i]
  # when find high end, idx-1 is the first ele smaller than searching val.
  def bisect(self, arr, val):
    lo,hi = 0, len(self.arr)-1
    if val > arr[-1]:      return False, len(arr)
    while lo != hi:
      md = (lo+hi)/2
      if arr[md] < val:    lo = md + 1 # lift lo only when absolutely big
      else:                hi = md   # drag down hi to mid when equal to find begining.
    if arr[lo] == val:     return True, lo
    else:                  return False, lo
  def overlap(self, st1, ed1, st2, ed2):
    return st2 > ed1 or ed2 < st1 ? False : True
  # which slot in the arr this new interval shall be inserted
  def findStartSlot(self, st, ed):
    """ pre-ed < start < next-ed, bisect insert pos is next """
    endvals = map(lambda x: x[1], self.arr)
    found, idx = self.bisect(endvals, st)  # insert position
    return idx
  # the last slot in the arr that this new interval may overlap
  def findEndSlot(self, st, ed):
    """ pre-start < ed < next-start, bisect ret insert pos, pre-start +1, so left shift"""
    startvals = map(lambda x: x[0], self.arr)
    found, idx = self.bisect(startvals, ed)        
    return idx-1  # bisect ret i arr[i] >= ed. ret i-1
  def merge(self, st1, ed1, st2, ed2):
    return [min(st1,st2), max(ed1,ed2)]
  def insertMerge(self, sted):
    [st,ed] = sted
    if len(self.arr) == 0:
      self.arr.append([st,ed])
      return 0
    stslot = self.findStartSlot(st,ed)
    edslot = self.findEndSlot(st,ed)
    print "insert ", sted, stslot, edslot
    if stslot >= len(self.arr):
      self.arr.insert(stslot, [st, ed])
    elif stslot > edslot:  # find ed slot is ed slot-1
      self.arr.insert(stslot, [st, ed])
    else:
      minst = min(self.arr[stslot][0], st)
      maxed = max(self.arr[edslot][1], ed)
      for i in xrange(stslot+1, edslot+1):
          del self.arr[stslot+1]
      self.arr[stslot][0] = minst
      self.arr[stslot][1] = maxed
    return stslot
  def insertToggle(self, sted):
    [st,ed] = sted
    if len(self.arr) == 0:
      self.arr.append([st,ed])
      return 0
    stslot = self.findStartSlot(st,ed)
    edslot = self.findEndSlot(st,ed)
    if stslot >= len(self.arr):
      self.arr.append([st,ed])
    elif stslot > edslot:
      self.arr.insert(stslot, [st,ed])
    else:
      output = []
      for i in xrange(0, stslot):
        output.append(self.arr[i])
      output.append([min(st, self.arr[stslot][0]), max(st, self.arr[stslot][0])])
      # toggle to create [ed->st]
      for i in xrange(stslot+1, edslot+1):
        output.append([self.arr[i-1][1], self.arr[i][0]])
      output.append([min(ed, self.arr[edslot][1]), max(ed, self.arr[edslot][1])])
      for i in xrange(edslot+1, len(self.arr)):
        output.append(self.arr[i])
      self.arr = output
  def insertToggleIter(self, sted):
    [st,ed] = sted
    output = []
    if len(self.arr) == 0:
      self.arr.append(sted)
      return 0
    if ed < self.arr[0][0]:
      self.arr.insert(0, sted)
      return 0
    if st > self.arr[len(self.arr)-1][1]:
      self.arr.append(sted)
      return len(self.arr)-1
    for i in xrange(len(self.arr)):
      if not self.overlap(self.arr[i][0], self.arr[i][1], st, ed):
          output.append(self.arr[i])
          continue
      if self.arr[i][0] < st:
          output.append([self.arr[i][0], st])
      if self.arr[i][0] > st:
          prest = st
          if i > 0:
              prest = max(prest, self.arr[i-1][1])
          output.append([prest, self.arr[i][0]])
      if self.arr[i][0] < ed and self.arr[i][1] > ed:
          output.append([ed, self.arr[i][1]])
      if i < len(self.arr) - 1 and self.arr[i][1] < ed and self.arr[i+1][0] > ed:
          output.append([self.arr[i][1], ed])
    self.arr = output
    return i
  def dump(self):
    for i in xrange(len(self.arr)):
        print self.arr[i][0], " -> ", self.arr[i][1]


""" 
node contains only 3 pointers, left < eq < rite, always descend down along eq pointer.
Trie of Suffix Wrapped Words, #apple', 'e#apple', 'le#apple', 'ple#apple', 'pple#apple, apple#apple
"""
class Trie:
  def __init__(self, val):
    self.val = val
    self.next = defaultdict(TernaryTree)  // new HashMap<String>();
    self.leaf = False
''' Ternary tree is repr-ed by its root, recurisve at root for ops '''
class TernaryTree(object):
  def __init__(self, key=None):
    self.key = key
    self.cnt = 1  # when leaf is true, the cnt, freq of the node
    self.leaf = False
    self.left = self.rite = self.eq = None
  ''' recursive down to subtree root and new word header '''
  def insert(self, word):
    hd = word[0]
    if not self.key:  # at first, root does not have key.
      self.key = hd
    if hd == self.key:
      if len(word) == 1: self.leaf = True  return self
      else:  # more keys, advance and descend
        if not self.eq:  # new eq point to new node 
          self.eq = TernaryTree(word[1])  # eq point to next char.
        self.eq.insert(word[1:])
    elif hd < self.key:
      if not self.left:
        self.left = TernaryTree(hd)  # left tree root is hd
      self.left.insert(word)
    else:
      if not self.rite:
        self.rite = TernaryTree(hd)  # rite tree root is hd
      self.rite.insert(word)
  def search(self, word):   # return parent where eq originated
    if not len(word):
      return False, self
    hd = word[0]
    if hd == self.key:
      if len(word) == 1:  return True, self # when leaf flag is set
      elif self.eq:       return self.eq.search(word[1:])
      else:               return False, self
    elif hd < self.key:
      if self.left:       return self.left.search(word)
      else:               return False, self
    elif hd > self.key:
      if self.rite:       return self.rite.search(word)
      else:               return False, self
  def all(self, prefix):
    result = set()
    found, node = self.search(prefix)
    if not found:        return result
    if node.leaf:        result.add(prefix)
    if node.eq:          node.eq.dfs(prefix, result)
    return result
  def dfs(self, prefix, result):
    if self.leaf:       result.add(prefix + self.key)
    if self.left:       self.left.dfs(prefix, result)
    if self.eq:         self.eq.dfs(prefix+self.key, result)
    if self.rite:       self.rite.dfs(prefix, result)
    return result

def test():
  t = TernaryTree()
  t.insert("af")
  t.insert("ab")
  t.insert("abd")
  t.insert("abe")
  t.insert("standford")
  t.insert("stace")
  t.insert("cpq")
  t.insert("cpq")
  found, node = t.search("b") 
  print found, node.key
  print t.all("abc")
  print t.all("ab")
  print t.all("af")
  print t.all("sta")
  print t.all("c")



""" 2d tree, key=[x,y] """
class KdTree(object):
  def __init__(self, xy, level=0):
      self.xy = xy
      self.level = level  # each node at a level.
      self.left = self.rite = None
  def __repr__(self):
      return "[" + str(self.xy) + "] L" + str(self.level)
  def leaf(self):
      if self.left or self.rite:
          return False
      return True
  def find(self,xy):
      keyidx = self.level % 2
      if self.xy[keyidx] == xy[keyidx]:
          return True, self
      elif xy[keyidx] < self.xy[keyidx]:
          if self.left:
              return self.left.find(xy)
      else:
          if self.rite:
              return self.rite.find(xy)
      return False, self
  def insert(self, xy):
      found, parent = self.find(xy)
      keyidx = parent.level % 2
      if xy[keyidx] > parent.xy[keyidx]:
          parent.rite = KdTree(xy, parent.level+1)
          return parent.rite
      else:
          parent.left = KdTree(xy, parent.level+1)
          return parent.left
  # root.search(q,null,sys.maxint)
  def search(self, q, p, w):
    if self.left:
      if distance(q,self.xy) < w:
        return self.xy
      else:
        return p
    else:
      keyidx = self.level % 2
      if w == sys.maxint:   # w is infinity
        if q[keyidx] < self.xy[keyidx]:
          p = self.left.search(q,p,w)
          w = distance(p,q)
          if q[keyidx] + w > self.xy[keyidx]:
            p = self.rite.search(q, p, w)
        else:
          p = self.rite.search(q,p,w)
          w = distance(p,q)
          if q[keyidx] + w > self.xy[keyidx]:
            p = self.left.search(q, p, w)
      else:  # w is not infinity
        if q[keyidx] - w < self.xy[keyidx]:
          p = self.left.search(q, p, w)
          w = distance(p,q);
          if q[keyidx] + w > self.xy[keyidx]:
            p = self.rite.search(q, p, w)
      return p            
def test():
    kd = KdTree([35, 90])
    kd.insert([70, 80])
    kd.insert([10, 75])
    kd.insert([80, 40])
    kd.insert([50, 90])
    kd.insert([70, 30])
    kd.insert([90, 60])
    kd.insert([50, 25])
    kd.insert([25, 10])
    kd.insert([20, 50])
    kd.insert([60, 10])

    print kd, kd.rite, kd.left
    print kd.find([80,40])[1].left


""" Binary Indexed Tree(BIT, or Fenwick), Each node in BI[] stores sum of a range.
the key is idx in BI, its parent is by removing the last set bit. 
idx = idx - (idx & (-idx)), (n&-n),ritemost set bit. n&n-1, clr ritemost set bit.
http://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/
* Example: given an array a[0]...a[7], we use a array BIT[9] to
* represent a tree, where index [2] is the parent of [1] and [3], [6]
* is the parent of [5] and [7], [4] is the parent of [2] and [6], and
* [8] is the parent of [4]. I.e.,
* 
* BIT[] as a binary tree:
*            ______________*
*            ______*
*            __*     __*
*            *   *   *   *
* indices: 0 1 2 3 4 5 6 7 8
* BIT[i] = ([i] is a left child) ? the partial sum from its left most
* descendant to itself : the partial sum from its parent (exclusive) to
* itself. (check the range of "__").
* 
* Eg. BIT[1]=a[0], BIT[2]=a[1]+BIT[1]=a[1]+a[0], BIT[3]=a[2],
* BIT[4]=a[3]+BIT[3]+BIT[2]=a[3]+a[2]+a[1]+a[0],
* BIT[6]=a[5]+BIT[5]=a[5]+a[4],
* BIT[8]=a[7]+BIT[7]+BIT[6]+BIT[4]=a[7]+a[6]+...+a[0], ...
* 
* Thus, to update a[1]=BIT[2], we shall update BIT[2], BIT[4], BIT[8],
* i.e., for current [i], the next update [j] is j=i+(i&-i) //double the
* last 1-bit from [i].
* Similarly, to get the partial sum up to a[6]=BIT[7], we shall get the
* sum of BIT[7], BIT[6], BIT[4], i.e., for current [i], the next
* summand [j] is j=i-(i&-i) // delete the last 1-bit from [i].
* To obtain the original value of a[7] (corresponding to index [8] of
* BIT), we have to subtract BIT[7], BIT[6], BIT[4] from BIT[8], i.e.,
* starting from [idx-1], for current [i], the next subtrahend [j] is
* j=i-(i&-i), up to j==idx-(idx&-idx) exclusive. (However, a quicker
* way but using extra space is to store the original array.)
* 
"""
class BITree:
    def __init__(self, arr):
        self.arr = arr
        self.bi = [0]*len(arr)
    def parentIdx(self, idx):
        return idx - (idx & (-idx))
    def childidx(self, idx):
        return idx + (idx & (-idx)) 
    # ret the sum of arr from 0..idx
    def getSum(self, idx):
        s = 0
        idx += 1 # idx in BITree[] is 1 more than the idx in arr[]
        # traverse along parent to root and sum up all values
        while idx > 0:
            s += self.bi[idx]
            idx = self.parentIdx(idx)
        return s
    ''' after update bi tree node at idx, traverse along parents to root '''
    def update(self, idx, val):
        idx += 1
        while idx < len(self.arr):
            self.bi[idx] += val   # cumulate
            idx = self.childIdx(idx)
    def build(self):
        for i in xrange(len(self.arr)):
            self.update(i, self.arr[i])



'''
Huffman coding: sort freq into min heap, take two min, insert a new node with freq = sum(n1, n2) into heap.
http://en.nerdaholyc.com/huffman-coding-on-a-string/
'''
class HuffmanCode(object):
    def __init__(self):
        self.minfreqheap = MinHeap()
        self.minheap = {}

    def initValue(self):
        self.minheap.insert('b', 3, None, None)
        self.minheap.insert('e', 4, None, None)
        self.minheap.insert('p', 2, None, None)
        self.minheap.insert('s', 2, None, None)
        self.minheap.insert('o', 2, None, None)
        self.minheap.insert('r', 1, None, None)
        self.minheap.insert('!', 1, None, None)

        self.minheap.heapify()

        self.minheap.toString()

    def encode(self):
        self.initValue()
        while len(self.minheap.Q) >= 2:
            left = self.minheap.extract()
            right = self.minheap.extract()
            left.toString()
            right.toString()
            self.minheap.insert('x', left.val+right.val, left, right)

        print ' huffman code ...'
        self.minheap.Q[0].inorder()


"""
count num of 1s that appear on all numbers less than n. 13=6{1,10,11,12,13}
consider n=2015: check the option of xxx1, xx1z, x1zz, 1zzz, sum them up
(1)consider 5: higherNum = 201; curDigit = 5; lowerNum = 0; weight = 1;
    choice of xxx1: xxx from 000 to 201 => (higherNum+1)*weight
(2)consider 1: higherNum = 20; curDigit =1; lowerNum = 5; weight = 10
      choice of xx1z: (a)xx from 0 to 19, z from 0 to 9  => higherNum*weight
                      (b)xx is 20       , z from 0 to 5  => lowerNum + 1
(3)consider 0: higherNum = 2; curDigit = 0; lowerNum = 15; weight =100
      choice of x1zz: (a)x from 0 to 1,   z from 0 to 99 => higherNum*wiehgt 
                      (b)x is 2         , 21zz > 2015    => right part 0 counts.
(4)consider 2: higherNum = 0; curDigit = 2; lowerNum = 015; weight = 1000
      choice of 1zzz: (a)no x(x==0)     , z from 000 to 999 => (higherNum+1)*weight
xyz, when y=0, tot += x*10, y=1, tot += x*10+z, y>1: tot += x*10+10
"""
def countOnes(n):
  lowidth = 1
  cnt = 0
  while n/lowidth > 0:
    hi,cur,lo = n/(lowidth*10),(n/lowidth)%10, n%lowidth
    if cur > 1:
      cnt += (hi+1)*lowidth
    elif cur == 1:
      cnt += hi*lowidth + lo + 1
    else:  # cur = 0, high part, 0..hi-1. and 20xx < 21xx, so xx does not count.
      cnt += hi*lowidth   # and 
    lowidth *= 10
  return cnt
assert countOnes(13) == 6
assert countOnes(219) == 152  # 2*10+9
assert countOnes(229) == 153  # 3*10

def countones(n):
  ln = list(str(n))
  cnt = 0
  for i in xrange(len(ln)-1,-1,-1):
    d = int(ln[i])
    left = int("".join(ln[:i]))
    rite = 0
    if i < len(ln)-1:
      rite = int("".join(ln[i+1:]))
    tens = power(10, len(ln)-1-i)
    if d > 1:
      cnt += (left+1)*tens
    if d == 1:
      cnt += left*tens + rite + 1
    if d == 1:
      cnt += left*tens
  return cnt

""" recursion; when msb > 1, 219, break to (100-199)+(0-99, 200-299), 100+msb*recur(100-1)
msb=1, 123, 123%100+recur(100-1); so w + msb*recur(w-1) + recur(n%w)
when msb=1, n%w + 1 + recur(w-1) + recur(n%w)
"""
def countones(n):
  def expo(n):
    width=1
    while n > 10:
      n = n/10
      width *= 10
    return n, width
  tot = 0
  sig,w = expo(n)
  if sig == 1:
    tot += 1 + n%w
    tot += 1*countones(w-1) + countones(n%w)
  else:
    # when sig > 1,000~099,200~299=count(99); 100~199=100+count(99)
    tot += w
    tot += sig*countones(w-1) + countones(n%w)
  return tot


""" thief can at cave k on day i. For all possibilities, strategy needs to cover all.
To cover day i at cave k, if strategy can cover day i-1, k-1 or k+1, [i,k] is covered.
tab[i][k] = (tab[i][k] or tab[i-1][k-1] or tab[i-1][k+1])
traverse strategy, as tab[i][k] only rely on tab[n-1], O(n) space.
"""
def alibaba(ncaves, strategy):
  ''' assume day 0 at cave pos, enum all possible cave for every day'''
  def posperm(ncaves, pos, days):
    result = []  # result will be n*2^d possibles
    result.append([pos])    # first day, at cave pos.
    for i in xrange(1,days):
      tmp = []
      for e in result:
        lastpos = e[-1]
        if lastpos == ncaves-1 or lastpos == 0:
          if lastpos == 0:
            e.append(1)
          else:
            e.append(lastpos-1)
          tmp.append(e)
        else:
          ne = e[:]
          ne.append(lastpos+1)
          tmp.append(ne)
          e.append(lastpos-1)
          tmp.append(e)
      result = tmp
    return result
  # n days
  days = len(strategy)
  result = []
  for i in xrange(ncaves):
    oneresult = posperm(ncaves, i, days)
    result.extend(oneresult)
  # iterate all possible result, each result is a list contains one possible when thief day 0 at cave k.
  for e in result:
    found = False
    for i in xrange(days):
      if strategy[i] == e[i]:
          found = True
    # after traversing, strategy not found a hit, then fail
    if not found:
      print "fail to find hit from strategy....", i, strategy, e
  return result

''' thief will be caught at cave c on day d if strategy[d]=c or strategy[d-1] == c-1/c+1
iter each prediction at day d, strategy[d], for each cave c, thief can survive only when
strategy[d] not cave c, and pre day cave c+1/c-1 also can survive.
one rolling row, survive[i] means whether thief wont be caught at cave i, at cur day k.
'''
def alibaba(ncaves, strategy):
  kdays = len(strategy)
  survive = [True]*ncaves  # at cur day, survive[i] = whether thief wont be caught at slot i
  survive[strategy[0]] = False  # will be caught when strategy[i] indicate it.
  for d in xrange(1,kdays):
    cave = strategy[d]
    survive[cave] = False
    canSurvive,pre = False, False
    for i in xrange(ncaves):  # check each cave at day d
      if i == 0:
        leftcave = False
      else:
        leftcave = pre
      if i == ncaves-1:
        ritecave = False
      else:
        ritecave = survive[i+1]   # thief was at pos i+1 on pre day
      pre = survive[i]  # cache survive[i] as pre
      if strategy[d] != i and (leftcave or ritecave):
        survive[i] == True
        canSurvive = True
    if not canSurvive:
      return False
  return True




'''
skip list, a probabilistic alternative to balanced tree
https://kunigami.wordpress.com/2012/09/25/skip-lists-in-python/
skiplist can be a replacement for heap. heap is lgn insert and o1 delete.
however, you can not locate an item in heap, you can only extract min/max.
with skiplist, list is sorted, insert is lgn, min/max is o1, del is lgn.
'''
class SkipNode:
    def __init__(self, height=0, elem=None):   # [ele, next[ [l1], [l2], ... ]
        '''each node has an ele, and a list of next pointers'''
        self.elem = elem             
        self.next = [None] * height  # next is a list of header at each level pts to next node

class SkipList:
    def __init__(self):
        self.head = SkipNode()  # list is a head node with n next ptrs
    def nextNodeAtEachLevel(self, ele):   # find list of express stop node at each level whose greatest value smaller than ele
        """ find q, we begin from the top-most level, go thru the list down, until find node with largest ele le q
        then go level below, begin search from the node found at prev level, and search again for node with largest ele le q
        when found, go down again, repeat until to the bottom.
        keep the node found at each level right before go down the level.
        """
        levelNextNode = [None]*len(self.head.next)
        levelhd = self.head
        #  for each level top down, list travel/while until stop on boundary.
        for level in reversed(xrange(len(self.head.next))):  # for each level down: park next node just before node >= ele.(insertion ponit).
            while levelhd.next[level] != None and levelhd.next[level].ele < ele:  # advance skipNode till to node whose next[level] >= ele, then down level
                levelhd = levelhd.next[level]  # store skip node(ele, next[...])
            # when stop at this level, record stop pos. hd < cur < next.
            levelNextNode[level] = levelhd
        return levelNextNode
    def find(self, ele):
        nextNode = nextNodeAtEachLevel(self, ele)
        if len(nextNode) > 0:
            candidate = nextNode[0].next[0]  # the final bottom level.
            if candidate != None and candidate.ele == ele:
                return candidate
        return None
    ''' insert always done at bottom level, flip coin which other lists should be in, 1/2, 1/4, 1/8
    '''
    def insert(self, ele):
        node = SkipNode(self.randomHeight(), ele)
        # first, populate head next all level with None
        while len(self.head.next) < len(node.next):
            self.head.next.append(None)
        nextList = self.nextNodeAtEachLevel(ele)
        if self.find(ele, nextList) == None :
            # did not find ele, start insert
            for level in range(len(node.next)):
                node.next[level] = nextList[level].next[level]
                nextList[level].next[level] = node
    ''' lookup for ith ele, for every link, store the width(span)(the no of bottom layer links) of the link.
    '''
    def findIthEle(self, i):
        node = self.head
        for level in reversed(xrange(len(self.head.next))):
            while i >= node.width[level]:
                i -= node.width[level]
                node = node.next[level]
        return node.value


sh map entry is linked list. entry value is list of [v,ts] tuple.
"""
from collections import defaultdict
class HashTimeTable:
  class Entry:   # Map.Entry is a list of [v,ts] sorted by ts
    def __init__(self, k, v, ts):
        self.key = k
        self.next = None
        # value list is a sorted list with entry as tuple of [v,ts]
        self.valueList = [[v,ts]]   # volatile
    def bisect(self, arr, ts=None):
        if not ts:
            return True, len(arr)-1
        if ts > arr[-1][1]:
            return False, len(arr)
        l,r = 0, len(arr)-1
        while l != r:   # fix beg / end before traverse 
            m = l + (r-l)/2
            if ts > arr[m][1]:
                l = m+1
            else:
                r = m
        if arr[l][1] == ts:
            return True, l
        else:
            return False, l
    def getValue(self,k, ts=None):
        found, idx = self.bisect(self.valueList, ts)
        if found:
            return self.valueList[idx][0]
        return False
    def insert(self, k, v, ts):
        lock()   // single writer.
        found, idx = self.bisect(self.valueList, ts)
        if found:
            self.valueList[idx][0] = v
        else:
            self.valueList.insert(idx, [v,ts])
                  
  def __init__(self, size):
      self.size = size
      self.bucket = [None]*size
  def search(self, k, ts=None):
      h = hash(k) % self.size
      entry = self.bucket[h]
      while entry:
          if entry.key == k:
              return entry, entry.getValue(k,ts)
          entry = entry.next
      return None, None
  def insert(self, k, v, ts):
      h = hash(k) % self.size
      if not self.bucket[h]:
          self.bucket[h] = HashTimeTable.Entry(k,v,ts)
          return self.bucket[h]
      else:
          entry,value = self.search(k,ts)
          if entry:
              return entry.insert(k,v,ts)
          else:
              entry = HashTimeTable.Entry(k,v,ts)
              entry.next = self.bucket[h]
              self.bucket[h] = entry
              return entry
    
htab = HashTimeTable(1)
htab.insert("a", "va1", 1)
htab.insert("a", "va2", 2)
htab.insert("b", "vb1", 1)
htab.insert("b", "vb2", 2)
htab.insert("b", "vb3", 3)
print htab.search("a")
print htab.search("a", 1)
print htab.search("b", 3)
