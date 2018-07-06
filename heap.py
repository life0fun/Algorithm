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


