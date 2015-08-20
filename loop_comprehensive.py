
import sys
import math
from collections import defaultdict


"""
struct list *merge(list *left, list *right, int (*compare)(list *, list*)){
    list *new, **np;
    np = &new;
    while(left != null && right != null){
        if(compare(left,right) > 0)
            *np = right; right = ->next; np = &right;
        }
    *np = left != null ? left : right;
    return new;
}
list * mergesort(list* list){
    list * left, *right;
    split(list, &left, &right);
    return merge(mergesort(left), mergesort(right));
}
"""

"""
Linear scan an array: context to store max/min/len seen so far and upbeat on each scan.
think of using pre-scan to fill out aux tables for o(1) look-up. 
1. max distance.
    http://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
1. max conti sum: use max start/end/len to record max seen so far.
2. segment with start-end overlap, sort by start, update min with smallest x value.
3. max sum: a ary with ary[i] = sum(l[0:i]). so sum[l[i:j]) = ary[j] - ary[i]
4. when combination, if at each pos, could have 2 choice, f(n) = f(n-1) + f(n-2)
5. for bit operation, find one bit to partition the set; repeat the iteration.
6. for string s[..], build suffix ary A[..] sorted in O(n) space O(n), s[A[i]] = start-pos-of-substring.
7. For int[range-known], find even occur of ints in place: toggle a[a[i]] sign. even occur will be + and odd will be -.
8. max dist when l[j] > l[j], shadow LMin[] and RMax[], merge sort scan.
9. stock profit. max(ctx.maxprofit, l.i - ctx.min), ctx.min = min(ctx.min, l.i)
   update each p[i] with max profit of l[0..i] from ctx.minp, or ctx.peak from n->1.
10. first missing positve int, bucket sort: l.0 shall store 1, l.1 stores 2. for each pos, while l.pos!=pos+1 : swap(l.pos, l[l.pos-1])
"""

''' recursive, one line print int value bin string'''
int2bin = lambda n: n>0 and int2bin(n>>1)+str(n&1) or ''

bstr_nonneg = lambda n: n>0 and bstr_nonneg(n>>1).lstrip('0')+str(n&1) or '0'

# hash a string is a reduce procedure, supply seed/initval to be 1
# provide an init val, reduce on top of this init value.
# fnv_hash = reduce(lambda tot, cur: tot^cur, 'string to be hashed')   # ret reslt to tot, cur mov nxt

# bstr_sgn: all integers, signed
# zero -> '0'
# negative get a minus sign
bstr_sgn = lambda n: n<0 and '-'+binarystr(-n) or n and bstr_sgn(n>>1).lstrip('0')+str(n&1) or '0'

# bstr: all integers
# zero -> '0'
# negative represented as complements, 16-bit by default
# optional second argument specifies number of bits
bstr = lambda n, l=16: n<0 and binarystr((2L<<l)+n) or n and bstr(n>>1).lstrip('0')+str(n&1) or '0'

''' convert int to bin str output'''
def int2bin(x, width=32):
    return ''.join(str((x>>i)&1) for i in xrange(width-1,-1,-1))

def bin2hex(x):
    size = len(x)
    hex = '0123456789ABCDEF'
    res = ""
    for i in xrange(size-4, -1, -4):
        sub = x[i:i+4]
        print x, sub
        sum = int(sub[3]) + 2*int(sub[2]) + 4*int(sub[1]) + 8*int(sub[0])
        res = hex[sum] + res
    print res

def int2hex(x, width=16):
    hex = '0123456789ABCDEF'
    l = []
    res = ""
    while x > 0:
        res = hex[x%16] + res
        x = x/16
    print res

def bin2int(binstr):
    for i in xrange(len(binstr)):
        sum += sum*2+binstr[len(binstr)-1-i]
    return sum

'''
src.find(c) find the offset of c in the number system, which is the decimal val of c.
convert to src to decimal, then decimal to dest
v(abc) = (a*base+b)*base+c
num = Digits[val%base] + num, val /= base
'''
def convert(num, src, trg):
    n = 0
    srcbase = len(src)
    trgbase = len(trg)
    for c in num:
        n=n*srcbase+src.find(c)  # len(src) is the base of src
        print 'c:', c, 'num:', num, 'n:', n
    res = ""
    while n:
        res = trg[n%trgbase] + res  #len(trg) is the base of target
        n /= trgbase
    print 'convert', num, ' res:', res
    return res


"""
when building a list, describe the first typical element,
then cons it onto the nature recursion.
tail recursion: compiler helps to avoid stackoverflow
"""
def mergeSort(p, q, l):
    if not len(p) or not len(q):
        l.extend(p)
        l.extend(q)
        return
    if p[0] == q[0]:
        l.append(p[0])
        return mergeSort(p[1:], q[1:], l)
    elif p[0] > q[0]:
        l.append(q[0])
        return mergeSort(p, q[1:], l)
    else:
        l.append(p[0])
        return mergeSort(p[1:], q, l)

'''
merge sort with while i j loop
Note that for loop is for list iteration, not suitable here.
'''
def merge(p, q, o):
    i,j = 0, 0
    while i<len(p) and j<len(q):
        if p[i] < q[j]:
            o.append(p[i])
            i += 1
        else:
            o.append(q[j])
            j += 1

    if i != len(p):
        o.extend(p[i:])
    if j != len(q):
        o.extend(q[j:])

'''
in-place merge use while ij loop, example, [1 3 9] [2 4 5 11]
the key is, when l[i] > l[j], need to shift j subary until the one > l[i]

'''
def mergeInPlace(p,q):
    l = []
    l.extend(p)
    l.extend(q)
    i = 0
    j = len(p)  # j starts from q
    while i < len(l) and j < len(l):
        if i == j:
            j += 1
            continue
        if l[i] <= l[j]:
            i += 1
            continue
        else:
            tmp = l[i]
            substart = subend = j
            while l[subend] < l[i]:
                subend += 1
            # fill insert pos, i
            l[i] = l[substart]
            # subary cp of j ary where items < l[i]
            l[substart:subend-1] = l[substart+1:subend-1]
            # insert l[i] into the correct idx
            l[subend] = tmp
            continue
    return l

def testMergeSort():
    q = []
    l = [1,3,4,7,9]
    r = [2,3,7,12]
    #mergeSort(l,r,q)
    mergeInPlace(l,r)
    print 'merge: ',q

    q = []
    l = [3,3,3,3,4]
    r = [3,3,3,3,6]
    r = []
    mergeSort1(l,r,q)
    print q


''' this is more readable code.
    1. pick a pivot, put to end
    2. start j one idx less than pivot, which is end-1.
    3. park i,j, i to first > pivot(i++ if l.i <= pivot), j to first <pivot(j-- if l.j > pivot)
    4. swap i,j and continue until i<j breaks
    4. swap pivot to i, the boundary of divide 2 sub string.
    5. 2 substring, without pivot in, recursion.
'''
''' qsort3([5,8,5,4,5,1,5], 0, 6)
'''
def qsort3(l, beg, end):
    def swap(i,j):
        l[i], l[j] = l[j], l[i]

    # always check loop boundary first!
    if(beg >= end):   # single ele, done.
        return

    #swap((beg+end)/2, end)   # now pivot is at the end
    #pivot = l[end]
    #i = beg
    #j = end -1

    swap((beg+end)/2, beg)   # now pivot is at the beg
    pivot = l[beg]
    i = beg+1
    j = end

    # while loop, execute when i=j, as long as left <= right
    while i<=j:   # = is two ele, put thru logic
        while l[i] <= pivot and i < end:  # i points to first ele > pivot, left half <= pivot
            i += 1
        while l[j] > pivot and j > beg:   # j points to first ele < pivot, right half > pivot
            j -= 1

        if(i < j):     # if i=j, swap not effect, but want move i,j actually.
            swap(i,j)
            i += 1
            j -= 1
        else:
            break

    # i,j must be within [beg, i, j, end]
    swap(beg, j)  # use nature j as i been +1 from head

    qsort3(l, beg, i-1)   # left half <= pivot
    qsort3(l, j+1, end)     # right half > pivot

    print 'final:', l

""" insert into circular link list 
    1. pre < val < next, 2. val is max or min, 3. list has only 1 element.
"""
def insertCircle(l, val):
    if not l: return Node(val)
    if l.next == l: 
        l.next = Node(val)
        return min(l.val, val)
    pre = head
    cur = head.next
    minnode = None
    while True:
        if prev < val < cur:
            prev.next = val
            return
        # at the end, val is max or min
        elif prev > cur and val > pre or val < cur:   
            prev.next = val
            minnode = pre
            return
    return minnode


""" dfs clone """
def clone(root, cloned):
  nroot = Node(root)
  cloned[root] = nroot
  for c in root.children:
    if not cloned[c]:
      nroot.children.push(clone(c, cloned))
  return nroot

""" serde tree with arbi # of children """
def serdeTree(root):
    print root
    for c in root.children:
        serdeTree(c)
    print "$"

# 1 [2 [4 $] $] [3 [5 $] $] $
from collections import deque
def deserTree(l):
    s = deque()
    while nxt = l.readLine():
        if nxt is "$":
            s.pop()
        else:
            n = Node(nxt)
            if count(s) == 0:
                s.append(n)
            else:
                s[-1].children.append(n)
                s.append(n)

""" serde of bin tree with l/r child, no lchild and rchild, append $ $"""
def serdeBtree(root):
    if not root: print '$'
    print root
    serdeBtree(root.left)
    serdeBtree(root.rite)

def serdeBtree(root):
    print root
    if root.lchild:
        serdeBtree(root.lchild)
    if root.rchild:
        serdeBtree(root.rchild)
    print "$"

def serdeBtree(line, parent, leftchild):
    if line is '$'
        return
    node = Node(line)
    if leftchild: parent.left = node else parent.rite = node
    serdeBtree(nextline, node, True)
    serdeBtree(nextline, node, False)

""" serde binary search tree, pre-order node 
    only recur when current node is child of parent, if next line not my child
    recur will keep ret until to its bst parent. read next when creating new node.
"""
def serdePreBst(val, parent, leftchild, minv, maxv):
    # only recur when this node is the child of parent
    if minv < val < maxv:
        node = Node(val)
        if leftchild: parent.left = node else parent.rite = node
        # read next line, if next line not my child, recur wi
        nextv = nextline()
        if nextv: 
            serdePreBst(nextv, node, True, -sys.maxint-1, val)
            serdePreBst(nextv, node, False, val, sys.maxint)

""" merge or toggle interval """
class Interval(object):
    def __init__(self):
        self.arr = []
        self.size = 0
    # bisect ret prev_slot + 1 where arr[prev_slot] < target, or 0.
    def bisect(self, arr, val):
        lo,hi = 0, len(self.arr)-1
        while lo != hi:
            md = (lo+hi)/2
            if val > arr[md]:
                lo = md+1
            else:
                hi = md
        if arr[lo] == val:
            return True, lo
        elif val > arr[lo]:
            return False, lo+1
        else:
            return False, lo
    # which slot in the arr this new interval shall be inserted
    def findStartSlot(self, st, ed):
        """ pre-ed < start < next-ed, bisect insert pos is next """
        endvals = map(lambda x: x[1], self.arr)
        found, idx = self.bisect(endvals, st)  # insert position
        return idx
    # the last slot in the arr that this new interval may overlap
    def findEndSlot(self, st, ed):
        """ pre-st < ed < next-st, bisect ret insert pos, pre-st+1, so left shift"""
        startvals = map(lambda x: x[0], self.arr)
        found, idx = self.bisect(startvals, ed)        
        return idx-1
    def overlap(self, st1, ed1, st2, ed2):
        if st2 > ed1 or ed2 < st1:
            return False
        return True
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
        elif stslot > edslot:  # no overlap
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

def testInterval():
    intv = Interval()
    intv.insertMerge([4,6])
    intv.insertMerge([7,9])
    intv.insertMerge([12,14])
    intv.insertMerge([1,2])
    intv.insertMerge([3,17])
    intv.dump()
    
    intv = Interval()
    intv.insertToggle([4,6])
    intv.insertToggle([8,10])
    intv.insertToggle([12, 14])
    intv.insertToggle([5, 15])
    intv.dump()

""" bst with lo as the bst search key
    interval tree annotate with max. Augmented search tree with # of children for rank.
"""
class IntervalTree(object):
    def __init__(self, lo=None, hi=None):
        self.lo = lo
        self.hi = hi
        self.max = max(lo, hi)
        self.left = self.rite = None
    def overlap(self, intv):
        [lo,hi] = intv
        if lo > self.hi or self.lo > hi:
            return False
        return True
    def toString(self):
        val = "[ " + str(self.lo) + ":" + str(self.hi) + ":" + str(self.max)
        if self.left:
            val += " / :" + self.left.toString()
        if self.rite:
            val += " \ :" + self.rite.toString()
        val += " ] "
        return val
    def insert(self, intv):  # lo as BST key
        [lo,hi] = intv
        if not self.lo:  # empty tree
            self.lo = lo
            self.hi = hi
            self.max = max(self.lo, self.hi)
            self.left = self.rite = None
            return self
        if self.max < hi:
            self.max = hi
        if lo < self.lo:
            if not self.left:
                self.left = IntervalTree(lo, hi)
                return self.left
            else:
                self.left.insert(intv)
        else:
            if not self.rite:
                self.rite = IntervalTree(lo, hi)
                return self.rite
            else:
                self.rite.insert(intv)
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
    def dfs(self, intv, result):
        [lo,hi] = intv
        if self.overlap(intv):
            result.add(self)   # update result upon base condition.
        if self.left and lo < self.left.max:
            self.left.dfs(intv, result)
        if self.rite and hi > self.riteMin().lo:
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

def test():
    intvtree = IntervalTree()
    intvtree.insert([17,19])
    intvtree.insert([5,8])
    intvtree.insert([21,24])
    intvtree.insert([4,8])
    intvtree.insert([15,18])
    intvtree.insert([7,10])
    intvtree.insert([16,22])
    print intvtree.search([21,23]).toString()
    result = set()
    intvtree.dfs([16,22], result)
    for e in result:
        print e.toString()

    intvtree = intvtree.delete([17,19])
    print intvtree.lo, intvtree.hi, intvtree.max
    result = set()
    intvtree.dfs([9,24], result)
    for e in result:
        print e.toString()
    
""" segment tree, two version, tree and heap """
import math
import sys
class RMQ(object):
    class Node():
        def __init__(self, lo=0, hi=0, minval=0, minvalidx=0, heapidx=0):
            self.lo = lo
            self.hi = hi
            self.minval = minval
            self.minvalidx = minvalidx  # idx in origin val arr
            self.heapidx = heapidx      # idx in rmq heap tree.
            self.left = self.rite = None
    def __init__(self, arr=[]):
        self.arr = arr  # value arr
        self.size = len(self.arr)
        self.heap = [None]*pow(2, 2*int(math.log(self.size, 2))+1)
        self.root = self.build(0, self.size-1, 0)[0]
    def build(self, lo, hi, heapidx):
        if lo == hi:
            n = RMQ.Node(lo, hi, self.arr[lo], lo, heapidx)
            self.heap[heapidx] = n
            return [n, lo]
        mid = (lo+hi)/2
        left,lidx = self.build(lo, mid, 2*heapidx+1)
        rite,ridx = self.build(mid+1, hi, 2*heapidx+2)
        minval = min(left.minval, rite.minval)
        minidx = lidx if minval == left.minval else ridx
        n = RMQ.Node(lo, hi, minval, minidx, heapidx)
        n.left = left
        n.rite = rite
        self.heap[heapidx] = n
        return [n,minidx]
    def query(self, root, lo, hi):
        if lo > root.hi or hi < root.lo:
            return sys.maxint, -1
        if lo <= root.lo and hi >= root.hi:
            return root.minval, root.minvalidx
        lmin,rmin = root.minval, root.minval
        if root.left:
            lmin,lidx = self.query(root.left, lo, hi)
        if root.rite:
            rmin,ridx = self.query(root.rite, lo, hi)
        minidx = lidx if lmin < rmin else ridx
        return min(lmin,rmin),minidx
    def queryIdx(self, idx, lo, hi):
        node = self.heap[idx]
        if lo > node.hi or hi < node.lo:
            return sys.maxint, -1
        if lo <= node.lo and hi >= node.hi:
            return node.minval, node.minvalidx
        lmin,rmin = node.minval, node.minval
        if self.heap[2*idx+1]:
            lmin,lidx = self.queryIdx(2*idx+1, lo, hi)
        if self.heap[2*idx+2]:
            rmin,ridx = self.queryIdx(2*idx+2, lo, hi)
        minidx = lidx if lmin < rmin else ridx
        return min(lmin,rmin), minidx
def test():
    r = RMQ([4,7,3,5,12,9])
    print r.query(r.root,0,5)
    print r.queryIdx(0,0,5)


""" 2d tree """
class KdTree(object):
    def __init__(self, xy, level=0):
        self.xy = xy
        self.level = level
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
      if self.left():
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


'''
find the diameter(width) of a bin tree.
diameter is defined as the longest path between two leaf nodes.
diameter = max(diameter(ltree, rtree) = max( depth(ltree) + depth(rtree) )
    http://www.cs.duke.edu/courses/spring00/cps100/assign/trees/diameter.html
'''
def diameter(root):
    if not root:
        return [0, 0]  # first is diameter, second is ht

    [ldiameter, ldepth] = diameter(root.lchild)
    [rdiameter, rdepth] = diameter(root.rchild)

    return [max(ldepth+rdepth+1, max(ldiameter, rdiameter)), max(ldepth, rdepth)+1]

def diameter(root):
    if not root.lchild and not root.rchild:
        return 0, [root, root], [root] # first diameter, second diamter leaf, last ht path
    
    ldiameter, ldialeaves, lhtpath = diameter(root.lchild)
    rdiameter, rdialeaves, rhtpath = diameter(root.rchild)
    if max(ldiameter, rdiameter)) > max(ldepth+rdepth)+1:
        leaves = ldialeaves
    else:
        leaves = [last(lhtpath), last(rhtpat)]
    return max(ldepth+rdepth+1, max(ldiameter, rdiameter)), leaves, max(ldepth, rdepth)+1


'''
min vertex set, dfs, when process vertex late, i.e., recursion of 
all its children done and back to cur node, use greedy, if exist
children not in min vertex set, make cur node into min set. return
recursion back to bottom up.
'''
def min_vertex(root):
    if root.leaf        # for leaf node, greedy use parent to cover the edge.
        return false
    for c in root.children:
        s = min_vertex(c)
        if not s:
            root.selected = true
    return root.selected

# the same as Largest Indep Set.
def vertexCover(root):
    if root == None or root.left == None and root.rite == None:
        return 0
    return root.vc if root.vc

    incl = 1 + vertexCover(root.left) + vertexCover(root.rite)
    excl = 1 + vertexCover(root.left.left) + vertexCover(root.left.rite)
    excl += 1 + vertexCover(root.rite.left) + vertexCover(root.rite.rite)
    root.vc = min(incl, excl)
    return root.vc

def depth(root):
    if not root:
        return 0
    return 1 + max(depth(root.lchild), depth(root.rchild))


""" populate next right pointer to point sibling in each node
    we can do bfs, level by level. or do recursion.
"""
def populateSibling(root):
    if not root: return root
    if root.lchild:
        root.lchild.sibling = root.rchild
    if root.rchild:
        root.rchild.sibling = root.sibling ? root.sibling.lchild : None
    populateSibling(root.lchild)
    populateSibling(root.rchild)


def populateNext(root):
    while root:
        nextlevelhd = None
        nextlevelcur = None
        cur = root
        while cur:
            for c in cur.children:
                if not nextlevelhd:
                    nextlevelhd = c
                if not nextlevelcur:
                    nextlevelcur = c
                else:
                    nextlevelcur.next = c
                    nextlevelcur = c
            cur = cur.next
        root = nextlevelhd

''' insertion is a procedure of replacing null with new node !'''
def insertBST(root, node):
    if not root.left and node.val < root.val:
        root.left = node
    if not root.right and node.val > root.val:
        root.right = node
    if root.left and root.val > node.val:
        insertBST(root.left, node)
    if root.right and root.val < node.val:
        insertBST(root.right, node)

def common_parent(root, n1, n2):
    if not root:
        return None
    if root is n1 or root is n2:
        return root
    left = common_parent(root.left, n1, n2)
    rite = common_parent(root.rite, n1, n2)
    if left and rite:
        return root
    return left if left else rite


# convert bst to link list, isleft flag is true when root is left child of parent 
# test with BST2Linklist(root, False), so the left min is returned.
def BST2Linklist(root, is_left_child):
    # leaf, return node itsef
    if not root.left and not root.rite:
        return root
    if root.left:
        lefthead = BST2Linklist(root.left, True)
        if lefthead:
            lefthead.next = root
            root.pre = lefthead
    if root.rite:
        ritehead = BST2Linklist(root.rite, False)
        if ritehead:
            root.next = ritehead
            ritehead.pre = root
    if is_left_child:
        # ret max rite
        maxrite = root
        while maxrite.rite:
            maxrite = maxrite.rite
        return maxrite
    else:
        # ret left min
        minleft = root
        while minleft.left:
            minleft = minleft.left 
        return minleft


''' max of left subtree, or first parent whose right subtree has me'''
def findPrecessor(node):
    if node.left:
        return findLeftMax(node.left)
    while root.val > node.val:  # smaller preced must be in left tree
        root = root.left
    while findRightMin(root) != node:
        root = root.right
    return root

def delBST(node):
    parent = findParent(root, node)
    if not node.left and not node.right:
        parent.left = null
    # if node has left, max of left child
    if node.left:
        cur = findLeftMax(node)
        delBST(cur)
        parent.left = cur
        return
    if node.right:
        cur = findRightMin(node)
        delBST(cur)
        parent.left = cur
        return
def testIsBST():
    root = Node()
    isBST(root, -sys.maxint, sys.maxint)
def toBST(l):
    if not l:
        return None
    mid = len(l)/2
    midv = l[mid]
    root = Node(midv)
    root.prev = toBST(l[:mid])
    root.next = toBST(l[mid:])
    return root


'''
Given a huge sorted integer array A, and there are huge duplicates in the
array. Given an integer T, inert T to the array, if T already exisit, insert
it at the end of duplicates. return the insertion point.
    A[15] = {2,5,7,8,8,8,8,8,8,8,8,8,8,8,9}
    If T == 6, return 2;
    If T == 8' return 14
'''
def findInsertionPosition(l, val):
    beg = 0
    end = len(l) - 1
    # the most important part is, when reduce to bottom, beg==end,
    # how do we handle ? all other cases will be reduced to this extreme.
    while beg <= end:
        mid = (beg+end)/2  # mid is Math.floor
        if l[mid] > val:
            end = mid - 1   # if mid=beg, end=mid-1 < beg, loop breaks
        elif l[mid] == val:   # in equal case, we want to append to end, so recur with mid+1
            beg = mid + 1
        else:   # in less case, of course advance lower.
            beg = mid + 1
    # now begin contains insertion point
    print 'insertion point is : ', beg
    return beg
def testFindInsertionPosition():
    l = [2,5,7,8,8,8,8,8,8,8,8,8,8,8,9]
    findInsertionPosition(l, 6)
    findInsertionPosition(l, 8)

''' func to calc sqrt on machine without floating point calc.
    first, expand to prec width, then binary search
    for n [0..1], write n as a/b, sqrt(a)/sqrt(b)
'''
def sqrt(n, prec=2):
    # use newton, xn+1 = xn - (xn*xn - s)/2*xn
    rt = n/2
    while rt*rt - n > 0.1/pow(10,prec):
        rt = rt - (pow(rt, 2) - n)/2*rt
    return rt

def sqrt(n, prec=2):
    def intSqrt(n, prec=2):
        xn = n*pow(10, prec)*pow(10,prec)   # expand to 100^2
        beg = 2
        end = xn-1

        while beg <= end:
            mid = (beg+end)/2
            if mid*mid < xn:
                beg = mid+1
            else:
                end = mid-1

        rt = beg if math.fabs(beg*beg-xn) < math.fabs(end*end-xn) else end
        print 'intSqrt :', xn, ' = ', rt
        return rt

    if n >= 1:
        return intSqrt(n, prec)
    else:
        numerator, denominator  = int(n*100), 100
        numsqrt = intSqrt(numerator, prec)
        denomsqrt = intSqrt(denominator, prec)
        rt = 1.0*numsqrt / denomsqrt
        print 'sqrt : ', n, ' = ', rt
        return rt

''' find the # of increasing subseq of size k '''
def numOfIncrSeq(l, start, cursol, k):
    if k == 0:
        print cursol
        return
    if start + k > len(l):
        return
    if len(cursol) > 0:
        last = cursol[len(cursol)-1]
    else:
        last = -sys.maxint
    for i in xrange(start, len(l), 1):
        if l[i] > last:
            tmpsol = list(cursol)
            tmpsol.append(l[i])
            numOfIncrSeq(l, i+1, tmpsol, k-1)

''' pascal triangle, 
http://www.cforcoding.com/2012/01/interview-programming-problems-done.html
'''


''' recursively on sublist '''
def list2BST(head, tail):
    mid = len(tail-head)/2
    mergesort(head)
    mergesort(mid)
    mergeListInPlace(head, mid)  # ref to wip/pydev/sort.py for sort two list in-place.

    if head is tail:
        head.prev = head.next = None
        return head

    mid = len(tail-head)/2
    mid.next = list2BST(mid.next, tail)
    mid.prev = list2BST(head, mid.prev)
    return mid

def bst2minheap(l):
    heapify(l)          # if array, just heapify, O(n)
    bst2linklist(root)  # if tree, convert to list, a sorted list alrady a min heap

def nfactor(n):
    filter(lambda x: n%x==0, [x for x in xrange(2,100)])

def factors_of(n):
    f = 2
    for step in chain([0,1,2,2], cycle([4,2,4,2,4,6,2,6])):
        f += step
        if f*f > n:
            if n != 1:
                yield n
            break
        if n%f == 0:
            yield f
            while n%f == 0:
                n /= f

def is_prime(n):
    '''proves primality of n using Lucas test.'''
    x = n-1
    if n%2 == 0:
        return n == 2
    factors = list( factors_of(x))
    b = 2
    while b < n:
        if pow( b, x, n ) == 1:

            for q in factors:
                if pow( b, x/q, n ) == 1:
                    break
            else:
                return True
        b += 1
    return False

''' continue when the smaller one, which is the remain of mod, still not zeroed '''
def gcd(a, b):
    while b != 0:   # continue as long as there is remain in mod
        a, b = b, a % b
    return a

def lcm(a, b): return a * b / gcd(a, b)

''' ensure a >= b '''
def gcd_extension(a, b):
    if a % b == 0:
        return (0, 1)
    else:
        x, y = gcd_extension(b, a%b)
        print "gcd_extended x %d, y %d" %(y, x-(y*(a/b)))
        return (y, x-(y*(a/b)))

def primes():
    primes_cache, prime_jumps = [], defaultdict(list)
    #primes_cache, prime_jumps = [], dict()
    prime = 1
    for i in count():
        if i < len(primes_cache): prime = primes_cache[i]
        else:
            prime += 1
            while prime in prime_jumps:
                for skip in prime_jumps[prime]:
                    prime_jumps[prime + skip] += [skip]
                del prime_jumps[prime]
                prime += 1
            prime_jumps[prime + prime] += [prime]
            #prime_jumps.setdefault((prime + prime),[]).extend([prime])
            primes_cache.append(prime)
        yield prime

def factorize(n):
    for prime in primes():
        if prime > n: return
        exponent = 0
        while n % prime == 0:
            print 'prime factor %d n %d exp %d' %(prime, n, exponent)
            exponent, n = exponent + 1, n / prime
        if exponent != 0:
            print 'prime %d n %d exp %d' %(prime, n, exponent)
            yield prime, exponent
        else:
            print 'n %d : do nothing on prime %d' %(n, prime)

def phi(n):
    r = 1
    for p,e in factorize(n):
        print p, e
        r *= (p - 1) * (p ** (e - 1))
    print r

''' sieve to get prime, for each #, prune all its multiples in the list'''
def cons(a, l):
    l.insert(0, a)
    return l   # must ret a list so recursion can go on!

''' this is a cursion func that take a list and prune, return pruned list
    why list insert/append not return the new list, recursion is dependent on it.
'''
def sieve(l):
    if not len(l):
        return []  # at the end of recursion, ret empty list
    return cons(l[0], sieve(filter(lambda x:x%l[0] != 0, l)))
    #return sieve(filter(lambda x:x%l[0] != 0, l)).insert(0, l[0])


''' recursive call this on each 0xFF '''
def sortdigit(l, startoffset, endoffset):
    l = sorted(l, key=lambda k: int(k[startoffset:endoffset]) if len(k) > endoffset else 0)

def radixsort(l):
    strl = map(lambda x: str(x), l)
    for i in xrange(5): # should be for each 2 bytes(0xFF)
        strl = sorted(strl, key=lambda k: int(k[len(k)-1-i]) if len(k)>i else 0)
        print strl

''' http://stackoverflow.com/questions/5212037/find-the-kth-largest-sum-in-two-arrays
    The idea is, take a pair(i,j), we need to consider both [(i+1, j),(i,j+1)] as next val
    in turn, when considering (i-1,j), we need to consider (i-2,j)(i-1,j-1), recursively.
    By this way, we have isolate subproblem (i,j) that can be recursively checked.
'''
class MaxHeap():
    class Node():
        def __init__(self, val, i, j):
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
        self.Q = []
    def toString(self):
        for n in self.Q:
            print n.val, n.ij
        print ' --- heap end ---- '
    def head(self):
        return self.Q[0]
    def lchild(self, idx):
        lc = 2*idx+1
        if lc < len(self.Q):
            return lc, self.Q[lc].val
        return None, None
    def rchild(self, idx):
        rc = 2*idx+2
        if rc < len(self.Q):
            return rc, self.Q[rc].val
        return None, None
    def parent(self, idx):
        if idx > 0:
            return (idx-1)/2, self.Q[(idx-1)/2].val
        return None, None
    def swap(self,i, j):
        self.Q[i], self.Q[j] = self.Q[j], self.Q[i]
    def siftup(self, idx):
        print self.toString()
        while True:
            pidx, pv = self.parent(idx)
            print 'parent ', idx, pidx, pv, self.Q[idx].val
            if pv and pv < self.Q[idx].val:
                self.swap(pidx, idx)
                idx = pidx
            else:
                return
    def siftdown(self, idx):
        while idx < len(self.Q):
            lc, lcv = self.lchild(idx)
            rc, rcv = self.rchild(idx)
            if lcv and rcv:
                maxcv = max(lcv, rcv)
                maxc = lc if maxcv == lcv else rc
            else:
                maxcv = lcv if lcv else rcv
                maxc = lc if lc else rc

            if self.Q[idx].val < maxcv:
                self.swap(idx, maxc)
                idx = maxc
            else:
                return  # done when loop break
    def insert(self, val, i, j):
        print 'inserting ', val, i, j
        node = MaxHeap.Node(val, i, j)
        self.Q.append(node)
        self.siftup(len(self.Q)-1)
    def extract(self):
        head = self.Q[0]
        print 'extract ', head.val, head.ij
        self.swap(0, len(self.Q)-1)
        self.Q.pop()      # remove the end, the extracted hearder, first
        self.siftdown(0)

def KthMaxSum(p, q, k):
    color = [[0 for i in xrange(len(q)) ] for i in xrange(len(p)) ]
    outl = []
    outk = 0
    i = 0
    j = 0
    heap = MaxHeap()
    heap.insert(p[i]+q[j], i, j)
    color[0][0] = 1

    for i in xrange(k):
        print ' heap ', heap.toString()
        head = heap.head()
        maxv = head.val
        i, j = head.ij
        outl.append((maxv, i, j))
        heap.extract()

        # insert [i+1,j] and [i, j+1]
        if i+1 < len(p) and not color[i+1][j]:
            color[i+1][j] = 1
            heap.insert(p[i+1]+q[j], i+1, j)
        if j+1 < len(q) and not color[i][j+1]:
            color[i][j+1] = 1
            heap.insert(p[i]+q[j+1], i, j+1)
    print 'result ', outl

def testKthMaxSum():
    p = [100, 50, 30, 20, 1]
    q = [110, 20, 5, 3, 2]
    KthMaxSum(p, q, 7)

""" min heap for priority queue """
from collections import defaultdict
class MinHeap(object):
    def __init__(self, size=10):
        self.arr = []
        self.size = 0
        self.val2idx = defaultdict()
    def get(self, idx):
        return self.arr[idx][0]
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
    def swap(self, src, dst):
        srcv = self.get(src)
        dstv = self.get(dst)
        self.arr[src], self.arr[dst] = [dstv], [srcv]
        self.val2idx[srcv] = dst
        self.val2idx[dstv] = src
    def insert(self, val):
        entry = [val]
        self.arr.append(entry)
        self.val2idx[val] = self.size
        self.size += 1
        self.siftup(self.size-1)
    def siftup(self, idx):
        parent = self.parent(idx)
        if parent >= 0 and self.get(parent) > self.get(idx):
            self.swap(parent, idx)
            return self.siftup(parent)
        return parent
    def siftdown(self, idx):
        l,r = self.lchild(idx), self.rchild(idx)
        minidx = idx
        if l and self.get(l) < self.get(minidx):
            minidx = l
        if r and self.get(r) < self.get(minidx):
            minidx = r
        if minidx != idx:
            self.swap(idx, minidx)
            return self.siftdown(minidx)
        return mindix
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
        idx = self.val2idx[v]
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
find kth smallest ele in a union of two sorted list
two index at A, B array, and inc smaller array's index each step for k steps.
(lgm + lgn) solution: binary search. keep two index on A, B so that i+j=k-1.
  i, j are the index that we can chop off from A[i]+B[j] = k, then we found k.
  if Ai falls in B[j-1, j], found, as A[0..i-1] and B[0..j-1] is smaller.
  otherwise, A[0..i] must less than B[j-1], chop off smaller A and larger B.
  recur A[i+1..m], B[0..j]
"""
def findKth(A, m, B, n, k):
    # ensure index i + j = k -1, so next index is kth element.
    i = int(float(m)/(m+n) * (k-1))
    j = (k-1)-i
    
    ai_1 = A[i-1] if i is not 0 else -sys.maxint
    bj_1 = B[j-1] if j is not 0 else -sys.maxint
    ai = A[i] if i is not m else sys.maxint  # if reach end of ary, max value
    bj = B[j] if j is not n else sys.maxint  

    # if either of i, j index falls in others.
    if ai > bj_1 and ai < bj:
        #print "ai ", ai
        return ai
    elif bj > ai_1 and bj < ai:
        #print "bi ", bj
        return bj

    # recur 
    if ai < bj:
        return findKth( A[i+1:], m-i-1, B[:j], j, k-i-1)
    else: # ai > bj case
        return findKth( A[:i], i, B[j+1:], n-j-1, k-j-1)

def testFindKth():
    A = [1, 5, 10, 15, 20, 25, 40]
    B = [2, 4, 8, 17, 19, 30]
    Z = sorted(A + B)
    assert(Z[7] == findKth(A, len(A), B, len(B), 8))
    print "9th of :", Z[9], findKth(A, len(A), B, len(B), 10)
    assert(Z[9] == findKth(A, len(A), B, len(B), 10))
    

""" 
  selection algorithm, find the kth largest element with worst case O(n)
  http://www.ardendertat.com/2011/10/27/programming-interview-questions-10-kth-largest-element-in-array/
"""
def selectKth(A, beg, end, k):
    def partition(A, l, r, pivotidx):
        A.r, A.pivotidx = A.pivotidx, A.r   # first, swap pivot to end
        swapidx = l   # all ele smaller than pivot, stored in left, started from begining
        for i in xrange(l, r):
            if A.i < A.r:   # smaller than pivot, put it to left where swap idx
                A.swapidx, A.i = A.i, A.swapidx
                swapidx += 1
        A.r, A.swapidx = A.swapidx, A.r
        return swapidx

    if beg == end: return A.beg
    if not 1 <= k < (end-beg): return None

    while True:   # continue to loop 
        pivotidx = random.randint(l, r)
        rank = partition(A, l, r, pivotidx)
        if rank == k: return A.rank
        if rank < k: return selectKth(A, l, rank, k-rank)
        else: return selectKth(A, rank, r, k-rank)

# median of median of divide to n groups with each group has 5 items,
def medianmedian(A, beg, end, k):
    def partition(A, l, r, pivot):
        swapidx = l
        for i in xrange(l, r+1):
            if A.i < pivot:
                A.swapidx, A.i = A.i, A.swapidx
                swapidx += 1
        # i will park at last item that is < pivot
        return swapidx-1   # left shift as we always right shift after swap.

    sz = end - beg + 1  # beg, end is idx
    if not 1<= k <= sz: return None     # out of range, None.
    # ret the median for this group of 5 items
    if end-beg <= 5:
        return sorted(A[beg:end])[k-1]  # sort and return k-1th item
    # divide into groups of 5, ngroups, recur median of each group, pivot = median of median
    ngroups = sz/5
    medians = [medianmedian(A, beg+5*i, beg+5*(i+1)-1, 3) for i in xranges(ngroups)]
    pivot = medianmedian(medians, 0, len(medians)-1, len(medians)/2+1)
    pivotidx = partition(A, beg, end, pivot)  # scan from begining to find pivot idx.
    rank = pivotidx - beg + 1
    if k <= rank:
        return medianmedian(A, beg, pivotidx, k)
    else:
        return medianmedian(A, pivotidx+1, end, k-rank)


""" 3sum, a list of integer, 3 of them sum to 0, or 2 of them sum to another.
    to avoid binary search, insert each item into hashtable, and for each pair,
    hash lookup any pair sum, found mean there exists.
""" 
def 3sum(l):
    l = sorted(l)
    for i in xrange(0, len(l)-3):
        a = l[i]
        k = i+1
        j = len(l)-1
        while k < j:
            b = l[k]
            c = l[j]
            if a+b+c == 0:
                print a,b,c
                return
            else if a+b+c < 0:
                k += 1
            else:
                l -= 1

'''
Huffman coding: sort freq into min heap, take two min, insert a new node with freq = sum(n1, n2) into heap.
http://en.nerdaholyc.com/huffman-coding-on-a-string/
'''
class HuffmanCode(object):
    def __init__(self):
        self.minheap = MinHeap()
        self.freq = {}

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


'''
skip list, a probabilistic alternative to balanced tree
https://kunigami.wordpress.com/2012/09/25/skip-lists-in-python/
skiplist can be a replacement for heap. heap is lgn insert and o1 delete.
however, you can not locate an item in heap, you can only extract min/max.
with skiplist, list is sorted, insert is lgn, min/max is o1, del is lgn.
'''
class SkipNode:
    def __init__(self, height=0, elem = None):   # [ele, next[ [l1], [l2], ... ]
        self.elem = elem             # each skip node has an ele, and a list of next pointers
        self.next = [None] * height  # next is a list of header at each level pts to next node

class SkipList:
    def __inti__(self):
        self.head = SkipNode()  # head has
    def findExpressStopAtEachLevel(self, ele):   # find list of express stop node at each level whose greatest value smaller than ele
        """ find q, we begin from the top-most level, go thru the list down, until find node with largest ele le q
        then go level below, begin search from the node found at prev level, and search again for node with largest ele le q
        when found, go down again, repeat until to the bottom.
        keep the node found at each level right before go down the level.
        """
        expressEachLevel = [None]*len(self.head.next)
        levelhd = self.head
        #  for each level top down, inside each level, while until stop on boundary.
        for level in reversed(xrange(len(self.head.next))):  # for each level down: park next node just before node >= ele.(insertion ponit).
            while levelhd.next[level] != None and levelhd.next[level].ele < ele:  # advance skipNode till to node whose next[level] >= ele, then down level
                levelhd = levelhd.next[level]
            expressEachLevel[level] = levelhd
        return expressEachLevel
    def find(self, ele):
        expressList = findExpressStopAtEachLevel(self, ele)
        if len(expressList) > 0:
            candidate = expressList[0].next[0]  # the final bottom level.
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
        expressList = self.findExpressStopAtEachLevel(ele)
        if self.find(ele, expressList) == None :
            # did not find ele, start insert
            for level in range(len(node.next)):
                node.next[level] = expressList[level].next[level]
                expressList[level].next[level] = node
    ''' lookup for ith ele, for every link, store the width(span)(the no of bottom layer links) of the link.
    '''
    def findIthEle(self, i):
        node = self.head
        for level in reversed(xrange(len(self.head.next))):
            while i >= node.width[level]:
                i -= node.width[level]
                node = node.next[level]
        return node.value


""" 
list scan comprehension compute sub seq min/max, upbeat global min/max for each next item.
1. max sub array problem. max[i] = max(max[i-1]+A.i, A.i) or Kadane's Algo. 
    scan thru, compute at each pos i, the max sub array end at i. upbeat.
    if ith ele > 0, it add to max, if cur max > global max, update global max; 
    If it < 0, it dec max. If cur max down to 0, restart cur_max beg and end to next positive.

1.1 max sub ary multiplication.
    scan thru, if abs times < 1, re-start with curv=1. otherwise, upbeat.
    do not care about sign during scan. When upbeat, if negative, cut off from the leftest negative.
    
2. longest increasing sequence. (LIS, variations, box stack)
    ctx.longestseq = []. scan next, append to end if next > end. if small, bisect it into the 
    cxt.longestseq. if it is real head, eventually the seq will override nodes in ctx.longestseq.

3. min sliding window substring in S that contains all chars from T.
    two pointers and two tables solution. [beg..end], and while end++ and squeze beg.
    need_to_found[x] and has_found[x], once first window established, explore end and squeeze beg.
    slide thru, find first substring contains T, note down beg/end/count as max beg/end/count.
    then start to squeeze beg per every next, and upbeat global min string.

4. find in an array of num, max delta Aj-Ai, j > i. the requirement j>i hints 
   slide thru, at each pos, compare to both lowest and highest value. if it set new low, 
   set cur_low and re-count, if it > cur_high, upbeat global max.

5. max distance between two items where right item > left items.
   http://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
   pre-scan, Lmin[0..n]   store min see so far. so min[i] is mono descrease.
             Rmax[n-1..0] store max see so far from left <- right. so mono increase.
   the observation here is, A[i] is shadow by A[i-1] A[i-1] < A[i], the same as A[j-1] by A[j].
   after we have Lmin and Rmax, scan both from left to right, stretch Rmax to the farest right.

6. slide window w in an array, find max at each position.
   a maxheap with w items. heap node(val, idx). when sliding, add rite(val, idx) to heap.
   if prq.top().idx < rite-winsize, max is prq.top(). if not, pop top. continue to pop while top().idx
   not in window(rite-winsz). O(n): deque, loop pop q tail if rite > q.tail.

7. convert array into sorted array with min cost. reduce a bin by k with cost k, until 0 to delete it.
   cost[i], at ary index i, the cost of sorted subary[0..i] with l[i] is the last.
   l = [5,4,7,3] => 5-1, rm 3. cost[i+1], dp each 1<k<i, 
    l[i+1] > l[k], cost[i+1] = cost[k] + del l[k] .. l[i+1]
    l[i+1] < l[k], cost[i+1] = min of make each l[k] to l[i+1], or del l[k]

8. three number sum to zero
   sort first, two point at both end, i, k, the j moving in between.
   for i in 0..sz to try all possibilities.

9. first missing pos int. bucket sort. L[0] shall store 1, L[1]=2. 
    foreach pos, while L[pos]!=pos+1: swap(L[pos],L[L[pos]-1]).

10. max square, square[i,j] = max(square[i,j+1], square[i+1,j], square[i+1,j+1])

11. largest subary with equal 0s and 1s.
    a. change 0 -> -1, reduce to larget subary sum to 0.
    b. in case start from idx other than head, sum[j]==sum[i], max(j-1). hash[sumval] = idx. 
"""
def max_subarray(A):
    # M[i] = max(M[i-1] + A.i, A.i)
    max_ending_here = max_so_far = 0
    for x in A:
        max_ending_here = max(0, max_ending_here + x)  # incl x 
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

''' for each i, either belong to prev seq, or starts its own, deps on whether prev can include it.
    s[i] = max(s[i-1]+A.i, A.i), either expand sum, or start new win from cur '''
def kadane(l):
    max_beg = max_end = max_val = 0
    cur_beg = cur_end = cur_val = 0
    for i in xrange(len(l)):
        cur_end = i
        if l[i] > 0:
            cur_val += l[i]
            upbeat()
        else:
            if l[i] + cur_val > 0:
                cur_val += l[i]
            else:
                cur_val = 0
                cur_start = i + 1

""" max sub arr product with pos and neg ints 
    mps([-2,-3,4,5])
"""
def mps(arr):
  pos,neg,maxsofar = 1,1,1
  for i in xrange(len(arr)):
    cur = arr[i]
    if cur == 0:
      pos = 1
      neg = 1
    elif cur < 0:
      tmp = pos
      pos = max(neg*cur, 1)
      neg = tmp*cur
    else:
      pos *= cur
      neg = min(cur*neg, 1)
    if pos > maxsofar:
      maxsofar = pos
  return maxsofar


""" nlgn LIS, only compute length """
def lis(arr):
  def bisectUpdate(l, val):
    lo,hi = 0, len(l)-1
    while lo != hi:
      mid = (lo+hi)/2
      if val > l[mid]:
        lo = mid+1
      else:
        hi = mid
    if l[lo] > val:  # l[lo] bound to great, if val is great, val appended.
      l[lo] = val # update lo with smaller
    return lo
  lis = []
  lis.append(arr[0])
  maxlen = 0
  for i in xrange(1,len(arr)):
    if arr[i] > lis[-1]:
      lis.append(arr[i])
    else:
      lo = bisectUpdate(lis, arr[i])
  return lis

''' as negative sign can twist, need to keep track both max[] and min[].
p[i] = max(pmin[i-1]*l.i, pmax[i-1]*l.i, l.i),
'''
def max_product(a):
    ans = pre_max = pre_min = a[0]
    cur_beg, cur_end, max_beg, max_end = 0
    for elem in a[1:]:
        new_min = pre_min*elem
        new_max = pre_max*elem
        pre_min = min([elem, new_max, new_min])
        pre_max = max([elem, new_max, new_min])
        ans = max([ans, pre_max])
        if ans is pre_max:
            cur_end, max_end = i
            upbeat(max_beg, max_end)
    return ans


""" 
loop each char, note down expected count and fount cnt. squeeze left edge.
"""
def minWindow(S, T):
    ''' find min substring in S that contains all chars from T'''
    ctx.found_chars = 0     # matched char len
    ctx.expected_cnt = defaultdict(int)    # need to match dict keyed by char
    ctx.found_cnt = defaultdict(int)
    ctx.cur_beg = cur_end = min_beg = min_eng = 0

    for c in T:   # populate dict of expected char cnt
        ctx.expected_cnt[c] += 1
    # sliding thru src string S
    for idx, val in enumerate(S):
        if ctx.expected_cnt[c] == 0:  # do not interested in this char
            continue
        ctx.found_cnt[val] += 1
        if ctx.found_cnt[val] <= ctx.expected_cnt[val]:        
            ctx.found_chars += 1  # contribute to found_chars only when new found.

        # a window found, maintain the window while squeeze left edge.
        if ctx.found_chars == len(T):
            # squeeze left edge, already found > expected
            while ctx.found_cnt[S[cur_beg]] > ctx.expected_cnt[S[cur_beg]] or
                  ctx.expected_cnt[S[cur_beg] == 0:
                ctx.found_cnt[S[cur_beg]] -= 1
                cur_beg += 1
            # upbeat optimal solution
            if idx - cur_beg < min_end - min_beg:
                min_beg, min_end = cur_beg, idx

    return min_beg, min_end

""" max value in sliding window """
def maxValueSlidingWin(A, w):
    maxheap = MaxHeap();
    for i in xrange(w):
        maxheap.add(Node(A.i, i))
    maxWin[i-w] = maxheap.top()
    for i in xrange(w, sz):
        if maxheap.top.idx > i-w:
            maxheap.add(A.i, i)
        else:
            top = maxheap.pop()
            while top.idx < i-w:
                top = maxheap.pop()
        maxWin[i-w] = maxheap.top()


def maxValueSlidingWin(A, w):
    Q = collections.deque()
    for i in xrange(w):
        while q.tail() < A.i:
            q.tail.pop()
        q.append(A.i)
    maxWind[0] = Q.head()
    for i in xrange(w, sz):
        while q.tail() < A.i:
            q.tail.pop()
        q.append(A.i)
        maxWin[i-w] = Q.head if Q.head.idx > i-w


def maxDelta(A):
    ''' in a seq of nums, find max(Aj - Ai) where j > i '''
    cur_beg = cur_end = max_beg = max_end = 0
    for idx, val in enumerate(A):
        if val < A[cur_beg]:
            cur_beg = idx
            continue
        else if val > A[cur_end]:
            cur_end = idx
            if val-A[cur_beg] > A[max_end] - A[max_beg]:
                max_beg, max_end = cur_beg, cur_end
        else:
            # do nothing if it is between cur_beg, cur_end
    return max_beg, max_end

"""
    max distance between two items where right item > left items.
    http://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
    pre-scan, Lmin[0..n]   store min see so far. so min[i] is mono descrease.
             Rmax[n-1..0] store max see so far from left <- right. so mono increase.
    the observation here is, A[i] is shadow by A[i-1] A[i-1] < A[i], the same as A[j-1] by A[j].
    Lmin and Rmax, scan both from left to right, like merge sort. upbeat max-distance.
"""
def max_distance(l):
    sz = len(l)
    Lmin = [l[0]]*sz
    Rmax = [l[sz-1]]*sz
    for i in xrange(1, sz-1:
        Lmin[i] = min(Lmin[i-1], l[i])
    for j in xrange(sz-2,0,-1):
        Rmax[j] = max(Rmax[j+1], l[j])
    i = j = 0   # both start from 0, Lmin/Rmax 
    maxdiff = -1
    while j < sz and i < sz:
        # found one, upbeat, Rmax also mono decrease, stretch
        if Lmin[i] < Rmax[j]:
            maxdiff = max(maxdiff, j-i)
            j += 1
        else: # Lmin is bigger, mov, Lmin is mono decrease
            i += 1
    return maxdiff

""" find idx a<b<c and l[a]<l[b]<l[c], two aux arr, Lmin and Rmax"""
def abc(l):
    ''' Lmin[i] contains idx in the left whose value is < cur, or -1
        Rmax[i] contain idx in the rite whose value > cur. or -1
        then loop both Lmin/Rmax, at i when Lmin[i] > Rmax[i], done
    '''
    pass


"""
    stock profit: calc diff between two neighbor. diff[i] = l[i] - l[i-1], then find
    the max subary on diff[] array. [5 2 4 3 5] = [-3 2 -1 2]
    or scan, compare A.i to both cur_low and cur_high, if A.i set cur_low, re-set, and upbeat.
"""
def stock_profit(l):
    sz = len(l)
    diff = [ l[i+1]-l[i] for i in xrange(sz)]
    maxbeg = maxend = maxval = 0
    curbeg = curend = curval = 0
    for i in xrange(1, sz):
        curend = i
        if diff[i] > 0:
            curval += diff[i]
            upbeat()
        elif curval > diff[i]:
            curval += diff[i]
        elif curval < diff[i]:
            curbeg = curend = i+1
            curval = 0

""" allow 2 transactions, use lp[i] = max_profit[0..i], and up[i] = max_profit[i..n]
    while iterating, fill lp[i] and up[i]
"""
def stock_profit(L):
    sz = len(L)
    for i in xrange(sz):
        # price valley going down from 0..n
        ctx.minp = min(ctx.minp, L.i)
        ctx.lp[i] = max(ctx.lp[i], L.i-ctx.minp)
        # price peak going up from n..0
        j = sz-i-1
        ctx.maxp[j] = max(ctx.maxp[j+1], L.j)
        ctx.up[j] = max(ctx.up[j+1], ctx.maxp[j]-L.j)

    for i in xrange(sz):
        tot = max(tot, ctx.lp[i] + ctx.up[i])

    return tot

""" first missing pos int. bucket sort. L[0] shall store 1, L[1]=2. 
    foreach pos, while L[pos]!=pos+1: swap(L[pos],L[L[pos]-1]).
"""
def firstMissingPos(L):
    def bucketsort(L):
        for i in len(L)-1:
            while L[i] != i+1:
                if L[i] < 0 or L[i] > len(L) || L[i] == L[L[i]-1]:
                    break
                swap(L[i], L[L[i]-1])
    bucketsort(L)
    for i in len(L)-1:
        if L[i] != i+1:
            return i+1
    return len(L)+1


# largest subary with equal 0s and 1s
# change 0 to -1, reduce to largest subary sum to 0.
# in case subary not start from head, find max(j-1) where sum[j]=sum[i]. Use hash[sum].
# http://www.geeksforgeeks.org/largest-subarray-with-equal-number-of-0s-and-1s/
def ls_equal(L):
    sz = len(L)
    sumleft = [0 for i in sz]

    for i in xrange(sz):
        sumleft[i] = sumleft[i-1] + L[i]

    for i in xrange(sz):
        cur_sumleft = sumleft[i]
        if cur_sumleft == 0:
            if i > ctx.maxsz:
                ctx.maxsz = i
                ctx.startidx = 0
        else:
            if ctx.hash[cur_sumleft] is None:
                ctx.hash[cur_sumleft] = i
            else:
                if i - ctx.hash[cur_sumleft] > ctx.maxsz:
                    ctx.maxsz = i - ctx.hash[cur_sumleft]
                    ctx.startidx = ctx.hash[cur_sumleft]
    return ctx.startidx, ctx.maxsz


# for each bar x, find the first smaller left, first smaller right,
# area(x) = first_right_min(x) - first_left_min(x) * X, largestRectangle([6, 2, 5, 4, 5, 1, 6])
# A stk of increase, so when push to stk, i's first left min is known, when pop, its first right min known.
def largestRectangle(l):
  maxrect = 0
  l.append(0)  # sentinel
  l.insert(0, 0)
  stk = []
  first_left_min = [0 for i in xrange(len(l))]
  first_right_min = [len(l) for i in xrange(len(l))]
  first_left_min[0] = 0
  stk.append(0)  # 
  for i in xrange(1,len(l)):
    curval = l[i]
    topidx = stk[-1]
    topval = l[topidx]
    # push to stk when current > stk top, so first smaller left of current set.
    if curval > topval:
      first_left_min[i] = topidx
    # when cur smaller than stk top, stk top right is known, continue pop stk.
    elif curval <= topval:
      first_right_min[topidx] = i  # stk top is peak. update its left/right.
      stk.pop()
      while len(stk) > 0:
        topidx = stk[-1]
        topval = l[topidx]
        if topval > curval:  # continue pop stk if cur is first smaller right of stk top
          first_right_min[topidx] = i
          stk.pop()
        else:
          first_left_min[i] = topidx
          break
    stk.append(i)  # stk in current

  print first_left_min
  print first_right_min
  for i in xrange(len(l)):
    maxrect = max(maxrect, l[i]*(first_right_min[i]-first_left_min[i]-1))

  return maxrect


""" slide window w in an array, find max at each position.
    a maxheap with w items. when slide out, extract maxnode, if max is not 
    the left edge, re-insert maxnode, continue extract until we drop left edge.
    this is how you remove node from heap. O(nlgW)
    To get rid of logW, squeeze w to single, upon new item, shadow/pop medium items.
"""

"""
    convert array into sorted array with min cost. reduce a bin by k with cost k, until 0 to delete it.
    cost[i], at ary index i, the cost of sorted subary[0..i] with l[i] is the last.
    l = [5,4,7,3] => 5-1, rm 3. cost[i+1], dp each 1<k<i, 
    l[i+1] > l[k], cost[i+1] = cost[k] + del l[k] .. l[i+1]
    l[i+1] < l[k], find the first l[j] < l[i+1], and cost[i+1] = min of make each l[j] to l[i+1], or del l[k]
"""
def convert_to_sort(l):
    dp = [sys.maxint for i in xrange(len(l))]
    # solve the cost of each subary [0..i]
    for i in xrange(len(l)):
        cur = l[i]
        for j in xrange(i-1, 0, -1):
            costi = 0
            if l[j] < cur:
                for k in xrange(j,i): delcost += l[k]
                costi = cost[i] + delcost
            else:
                # find first l[j] <= l[i]
                costi += delcost
                while l[j] > cur and j > 0:
                    costi += l[j] - cur
                    j -= 1
                costi += dp[j+1]

            # after each round of j, up-beat
            dp[i] = costi if dp[i] > costi else costi

    # as dp[i] is the cost of with l[i] as the last item.
    # global can be the one with last item bing deleted.
    gmin = sys.maxint
    for i in xrange(len(l)):
        costi = dp[i] + delcost[n-1]-delcost[i]
        gmin = min(costi, gmin)
    return gmin

"""
   three number sum to zero
   sort first, two point at both end, i, k, the j moving in between.
   for i in 0..sz to try all possibilities.
"""
def three_sum_zero():
    for i in xrange(1, n-2):
        j = i  # Start where i is.
        k = n  # Start at the end of the array.

        while k >= j:
            if A[i] + A[j] + A[k] == 0: 
                return (A[i], A[j], A[k])

            #We didn't match. Let's try to get a little closer:
            # If the sum was too big, decrement k.
            # If the sum was too small, increment j.
            if A[i] + A[j] + A[k] > 0: 
                k--
            else: j++
      
    # When the while-loop finishes, j and k have passed each other and there's
    # no more useful combinations that we can try with this i.

""" integer partition problem. break to subproblem, recur equation, start from boundary.
    for item in [0..n], so the first for will solve single item case, then bottom up.
    fence partition, m[n, k] = min(max(m[i-1,k-1], sum[i..n])) for i in [0..n]
    boundary: m[1,k] = A.1, one item, k divider; m[n,1] = sum(A)
    subset sum, s[i, v] = s[i-1,v] + s[i-1, v-A.i]
"""
def integer_partition(A, n, k):
    m = [[0 for i in xrange(n)] for j in xrange(k)]  # min max value
    d = [[0 for i in xrange(n)] for j in xrange(k)]  # where divider is
    # calculate prefix sum
    for i in xrange(A): psum[i] = psum[i-1] + A[i]
    # boundary, one divide, and one element
    for i in xrange(A): m[i][1] = psum[i]
    for j in xrange(k): m[1][j] = A[1]
    for i in xrange(2, n):
        for j in xrange(2, k):
            m[i][j] = sys.maxint
            for x in xrange(1, i):
                cost = max(m[x][j-1], psum[i]-psum[x])
                if m[i][j] > cost:
                    m[i][j] = cost
                    d[i][j] = x     # divide at pos x


""" reg match . *. recursion, loop try 0,1,2...times for *. """
def regMatch(T, P, offset):
    if P is None: return T is None
    if p+1 is not '*':
        return p == s or p is '.' and regMatch(T,P, offset+1)
    else:
        # loop match 1, 2, 3...
        while p == s or p is '.':
            if regMatch(T, P+2, offset+2): return True
            s += 1
        # match 0 time
        return regMatch(T, P, offset+2)


""" """ """ """ """
Dynamic programming, divide and conquer the problem space from minimal, then recursion. F[i,j] = {F[i-1, j-1]}
You can do recursion fn with reducing the space to the leaf. or tabular F[i,j] from i-1,j-1, etc.
""" """ """ """ """
def knapsack(maxw, W, V):
    ''' start from minimal, only item 0, i loop items, w loops weights at each item. 
        v[i, w] is max value for each weight with items from 0 expand to item i, v[i,w] based on f(v[i-1, w]).
        v[i, w] = max(v[i-1, w], v[i-1, w-W[i]]+V[i]) 
    '''
    for i in xrange(n):
        for w in xrange(maxw):
            tab[i,w] = max(tab[i-1, w], tab[i-1, w-W[i-1]] + V[i])
    return tab[n, maxw]
# order does not matter, exclu current i, and incl cuurent i
def coinchange(n, v):
    for i in xrange(n):
        for v in xrange(v):
            c[i,v] = c[i-1,v] + c[i, v-V[i]]
# order matters
def coinchange(n, v):
    for v in V:
        for i in n:
            tab[v] += tab[v-V[i]]

# 2 boundary checks: sz, and offset.
def combination(arr, path, sz, offset, result):
    if sz == 0:
        result.append(path)  # add to result only when sz
        return result
    if offset >= len(arr):
        return
    l = path[:]
    l.append(arr[offset])
    combination(arr, l, sz-1, offset+1, result)
    combination(arr, path, sz, offset+1, result)
    return
"""
   result = []; combinationIter(["a","b","c","d"], [], 2, 0, result); print result;
"""
def combinationIter(arr, path, sz, offset, result):
    if sz == 0:
        result.append(path)
        return result
    if offset >= len(arr):
        return
    for i in xrange(offset, len(arr)):
        l = path[:]
        l.append(arr[i])
        combinationIter(arr, l, sz-1, i+1, result)

# [a [ab [abc]] [ac]] , [b [bc]] , [c]
def powerset(arr, offset, path, result):
    result.append(path)
    for i in xrange(offset, len(arr)):
        l = path[:]
        l.append(arr[i])
        powerset(arr, i+1, l, result)


# [a [ab [abc]] [ac [acb]]], [b [ba [bac]] [bc [bca]]], [c ...]
def permutation(arr, path, offset, result):
    if offset == len(arr)-1:
        path.append(arr[offset])
        result.append(path)
        return
    for i in xrange(offset, len(arr)):
        l = path[:]
        arr[offset], arr[i] = arr[i], arr[offset]
        l.append(arr[offset])
        permutation(arr, l, offset+1, result)
        arr[offset], arr[i] = arr[i], arr[offset]

def perm(arr):
    result = []
    if len(arr) == 1:
        result.append([arr[0]])
        return result
    for i in xrange(len(arr)):
        hd = arr[i]
        l = arr[:]
        del l[i]
        for e in perm(l):
            e.insert(0,hd)
            result.append(e)
    return result

""" permutation rank of a string, at pos i, num of permuations is i!
    find count n on the rite of pos is smaller, n * pos!, with dup, n*pos!/d[i]
"""
from collections import defaultdict
def permRankNoDup(s):
    def ordchar(c):
        return ord(c)-ord('a')
    def fact(n):
        f = 1
        for i in xrange(1,n+1):
            f = f*i
        return f
    def riteSmaller(arr):
        smallcnt = [0]*26
        count = [0]*26
        for i in xrange(len(arr)-1, -1, -1):
            cord = ordchar(arr[i])
            count[cord] += 1
            for j in xrange(cord):
                smallcnt[cord] += count[j]
        return smallcnt
    arr = list(s)
    smallcnt = riteSmaller(arr)
    permu = fact(len(arr))
    rank = 1
    for i in xrange(len(arr)-1):
        c = arr[i]
        ritesmaller = smallcnt[ordchar(c)]
        permu /= (len(arr)-i)
        rank += ritesmaller*permu
        print i, c, ritesmaller, permu, rank
    return rank

from collections import defaultdict
def permRankDup(s):
    def ordchar(c):
        return ord(c)-ord('a')
    def fact(n):
        f = 1
        for i in xrange(1,n+1):
            f = f*i
        return f
    def riteSmaller(arr):
        smallcnt = [0]*26
        count = [0]*26
        for i in xrange(len(arr)-1, -1, -1):
            cord = ordchar(arr[i])
            count[cord] += 1
            for j in xrange(cord):
                smallcnt[cord] += count[j]
        return smallcnt, count
    arr = list(s)
    smallcnt, count = riteSmaller(arr)
    permu = fact(len(arr))
    rank = 1
    for i in xrange(len(arr)-1):
        c = arr[i]
        cidx = ordchar(c)
        ritesmaller = smallcnt[cidx]
        permu = permu / (count[cidx] * (len(arr)-i))
        count[cidx] -= 1
        rank += ritesmaller*permu
        print i, c, ritesmaller, permu, rank
    return rank




"""
;; cons each head to each tail, which is recur result of list without header
(defn all-permutations [things]
  (if (= 1 (count things))
    (list things)
    (for [head things
          tail (all-permutations (disj (set things) head))]
      (do
        (cons head tail)))))
(all-permutations '(a b c))
"""

""" valid parenthese, two branches """
def parenthese(n, lefts, rites, path, result):
    if (lefts == n):
        for i in xrange(rites, n):
            path.append(")")
        result.append(path)
        return
    if (lefts == rites):
        p = path[:]
        p.append("(")
        parenthese(n, lefts+1, rites, p, result)
        return    
    p = path[:]
    p.append("(")
    parenthese(n, lefts+1, rites, p, result)
    p = path[:]
    p.append(")")
    parenthese(n, lefts, rites+1, p, result)


""" word break, O(n2) """
# result=[];path=[];word="catsanddog";print wordbreak(word,0,path,result)
def wordbreak(word, offset, path, result):
    dict = ["cat", "cats", "and", "sand", "dog"]
    if offset == len(word):
        result.append(path)
        return result
    for i in xrange(offset, len(word)):
        if word[offset:i+1] in dict:
            l = path[:]
            l.append(word[offset:i+1])
            wordbreak(word, i+1, l, result)
    return result

""" recursion version """
from collections import defaultdict
def wordbreak(word):
    dict = ["cat", "cats", "and", "sand", "dog"]
    tab = defaultdict(lambda: []) # tab[i] = tab[j] + cw, when j < i
    def dp(offset):
        for i in xrange(offset-1, -1, -1):
            cw = word[i:offset+1]
            if cw in dict:
                if i == 0:
                    tab[offset].append([cw])
                    return
                if len(tab[i-1]) == 0:
                    dp(i-1)
                if len(tab[i-1]) > 0:
                    for e in tab[i-1]:
                        tmp = list(e)
                        tmp.append(cw)
                        tab[offset].append(tmp)
    dp(len(word)-1)   # recursion to rest from head
    return tab[len(word)-1]

""" DP version """
def wordbreak(word, dict):
    dp = [0 for i in xrange(word)]
    for i in xrange(word):
        for j in xrange(i):
            cw = word[j:i+1]
            if cw in dict:
                if j == 0:
                    dp[i] = 1
                elif dp[j]:
                    dp[i] += dp[j]
    return dp[len(word)-1]

from collections import defaultdict
def wordbreak(word):
    dict = ["cat", "cats", "and", "sand", "dog"]
    tab = defaultdict(lambda: [])
    for i in xrange(len(word)):
        for j in xrange(i-1, -1, -1):
            cw = word[j:i+1]
            if cw in dict:
                if j == 0:
                    tab[i].append([cw])
                elif len(tab[j-1]) > 0:
                    for e in tab[j-1]:
                        tmp = list(e)
                        tmp.append(cw)
                        tab[i].append(tmp)
    return tab[len(word)-1]


""" various way for decode """
def calPack(arr):
    prepre, pre = arr[0], arr[1]
    for i in xrange(2,len(arr)):
        curmaxv = max(prepre + arr[i], pre)
        maxv = max(maxv, curmaxv)
        prepre, pre = pre, curmaxv


""" recursion, bottom up, forward, from 0..n-1 """
def decode(dstr, pos):
  if dstr[pos] == '0':  # when forward, no more branch when hits 0
    return 0
  if pos == len(dstr)-1:
    return 1
  if dstr[pos] == '1' or (dstr[pos] == '2' and dstr[pos+1] <= '6'):
    if pos+2 < len(dstr):
      return decode(dstr, pos+1) + decode(dstr, pos+2)
    else:
      return decode(dstr, pos+1) + 1
  else:
    return decode(dstr, pos+1)

""" recursion, top down, backward, from n-1..0 """
def decode(dstr, pos):
  if pos == 0:
    return 1
  if dstr[pos] == '0':  # when backwards, recur when hits 0.
    return decode(dstr, pos-1)
  if dstr[pos-1] == '1' or (dstr[pos-1]=='2' and dstr[pos]<='6'):
    if pos - 2 < 0:
      return decode(dstr, pos-1) + 1
    else:
      return decode(dstr, pos-1) + decode(dstr,pos-2)
  else:
    return decode(dstr, pos-1)

""" recursive build a tree, then print all leaf """
def decode(dstr):
    root = Node(dstr)
    root.left = decode(dstr[1:])
    root.rite = decode(dstr[2:])
    return root

""" DP, with O(n) ary """
def decode(dstr):
  dp = [1 for i in xrange(len(dstr))]
  for i in xrange(1, len(dstr)):
    if dstr[i] == '0':
      dp[i] = dp[i-1]
    elif dstr[i-1] == '1' or (dstr[i-1]== '2' and dstr[i] <= '6'):
      if i < 2:
        dp[i] = 2
      else:
        dp[i] = dp[i-1] + dp[i-2]
    else:
      dp[i] = dp[i-1]
  return dp[len(dstr)-1]

""" DP with O(1) """
def decode(dstr):
  if len(dstr) == 1:
    return 1
  prepre, pre = 1, 1
  for i in xrange(1, len(dstr)):
    if dstr[i] == '0':
      cur = pre
    elif dstr[i-1] == '1' or (dstr[i-1]=='2' and dstr[i]<='6'):
      cur = prepre + pre  # fib seq here.
    else:
      cur = pre
    prepre, pre = pre, cur
  return cur

def decode(dstr):
  if len(dstr) == 1:
    return 1
  prepre = 1
  if int(dstr[0:2]) <= 26 and dstr[1] != '0':
    pre = 2
  else:
    pre = 1
  cur = pre   # in case only 2 digit
  for i in xrange(2, len(dstr)):
    cur = pre
    if dstr[i-1] is '1' or dstr[i-1] is '2' and dstr[i] <= '6':
      cur = prepre + pre
    prepre, pre = pre, cur
  return cur

def decode(dstr):
    if len(dstr) == 1:
        return 1
    pre,cur = 0, 1
    for i in xrange(1, len(dstr)+1):
        if dstr[i-1] == '0':
            cur = 0
        if i < 2 or not (dstr[i-2] == '1' or (dstr[i-2]=='2' and dstr[i-1]<='6')):
            pre = 0
        tmp = cur
        cur = pre + cur
        pre = tmp
    return cur

""" if dup not allowed, shift offset when dfs inclusion branch """
# path=[];result=[];l=[2,3,4,7];subsetsum(l,3,7,path,result);print result
def subsetsum(l, offset, n, path, result):
    if offset < 0:
        return
    p = path[:]
    if n == l[offset]:
        p.append(l[offset])
        result.append(p)
    if n > l[offset]:  # branch incl when when l.i smaller
        p.append(l[offset])
        # dup allowed, offset stay when include, incl 1+ times.
        subsetsum(l, offset,   n-l[offset], p, result)
        subsetsum(l, offset-1, n-l[offset], p, result)
    # always reduce when excl current, with excl l[i] path.
    subsetsum(l, offset-1, n, path, result)

"""
 when for loop enum next items in parallel, effectively as exclude current item branch.
"""
def subsetsum(l, offset, n, path, result):
    if n == 0:
        result.append(path)
        return
    for i in xrange(offset, -1, -1):
        if n >= l[i]:  # only include l[i]
            p = path[:]
            p.append(l[i])
            subsetsum(l, i, n-l[i], p, result)
            # shift offset when no dup allowed, to incl once
            # subsetsum(l, i-1, n-l[i], p, result)


""" DP tab[offset][val] """
from collections import defaultdict
def subsetsum(l, val):
    tab = [[0]*(val+1) for i in xrange(len(l))]  # tab[i][v], l[0:i] has v
    for i in xrange(len(l)):
        for v in xrange(val+1):
            tab[i][v] = []
        v = l[i]
        tab[i][v].append([v])
    for r in xrange(1,len(l)):
        rv = l[r]
        for v in xrange(val+1):
            if len(tab[r-1][v]) > 0:
                tab[r][v].append(list(tab[r-1][v]))
            if len(tab[r-1][v-rv]) > 0:
                p = list(tab[r-1][v-rv])
                for e in p:
                    e.append(rv)
                    tab[r][v].append(e)
    return tab[len(l)-1][val]


def palindromMincut(s, offset):
    ''' partition s into sets of palindrom substrings, with min # of cuts 
        mincut[offset] = 1 + min(mincut[k] for k in [offset+1..n] where s[offset:k] is palindrom)
    '''
    mincut = INT_MAX
    for i in xrange(offset+1, len(s)):
        if isPalindrom(s, offset, i):
            partial = palindromMincut(s, i)   # recursion to leaf, aggregate on top.
            mincut = min(partial + 1, mincut)
    return mincut

def palindromMincut(s):
    ''' convert recursion Fn call to DP tab. F[i] is min cut of S[i:n], solve leaf, aggregate at top.
        F[i] = min(F[k] for k in i..n). To aggregate at F[i], we need to recursive to leaf F[i+1].
        For DP tabular lookup, we need to start from the end, which is minimal case, and expand
        to the begining filling table from n -> 0.
    '''
    n = len(s)
    tab[n-1] = 1  # one char is palindrom
    for i in xrange(n-2, 0, -1):  # start from end that is minimal case, expand to full.
        for j in xrange(i, n-1):
            if s[i] == s[j] and palin[i+1, j-1] is True:   # palin[i,j] = palin[i+1,j-1]
                palin[i,j] = True
                tab[i] = min(tab[i], 1+tab[j+1])
    # by the end, expand to full string.
    return tab[0]

def palindromMincut(s):
    ''' another way to build tab from 0 -> n. F[i] = min(F[i-k] for k in i->0 where s[k,i] is palin)
        To aggregate fill F[i], we need know moving k, f[k], and s[k,i] is palin.
        For DP tabular lookup, we can start from the begin, the minimal case, and expand to entire string.
    '''
    n = len(s)
    tab[1] = 1  # tab[i], the mincut of string s[0:i]
    for i in xrange(n):
        for j in xrange(i, 0, -1):
            if s[i] == s[j] and palin[j+1, i-1] is True:
                palin[j,i] = True
                tab[i] = min(tab[j+1] + 1, tab[i])
    # expand to full string by the end
    return tab[n-1]


if __name__ == '__main__':
    #qsort([363374326, 364147530 ,61825163 ,1073065718 ,1281246024 ,1399469912, 428047635, 491595254, 879792181 ,1069262793], 0, 9)
    qsort3([1,2,3,4,5,6,7,8,9,10], 0, 9)
    qsort3([10,9,8,7,6,5,4,3,2,1], 0, 9)
    qsort3([5,8,2,4,5,1,9], 0, 6)
    qsort3([5,5,5,5,5,5,5], 0, 6)
    #numOfIncrSeq([1,2,4,5,6], 0, [], 3)
    #pascalTriangle(10)
    #pascalTriangle_iter(10)
    #binSearch([1,2,5,9], 3)
    #binSearch([2,4,10], 1)
    #powerset([1,2,3,4,5])
    #testKthMaxSum()
    hf = HuffmanCode()
    hf.encode()
    
    #
    sqrt(8)
    sqrt(0.4)

    # test find insertion postion
    testFindInsertionPosition()
    
    # test find kth
    testFindKth()