#!/usr/bin/env python

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
Linear scan an array: use Extra data structure to store max/min/len seen so far and upbeat on each scan.
1. max conti sum: use max start/end/len to record max seen so far.
2. segment with start-end overlap, sort by start, update min with smallest x value.
3. max sum: a ary with ary[i] = sum(l[0:i]). so sum[l[i:j]) = ary[j] - ary[i]
4. when combination, if at each pos, could have 2 choice, f(n) = f(n-1) + f(n-2)
5. for bit operation, find one bit to partition the set; repeat the iteration.
6. for string s[..], build suffix ary A[..] sorted in O(n) space O(n), s[A[i]] = start-pos-of-substring.
7. For int[range-known], find even occur of ints in place: toggle a[a[i]] sign. even occur will be + and odd will be -.
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

'''
find the common elements between two sorted sequence
tail recusion so no worry of stackoverflow.
'''
def common(l, r, q):
    if not len(l) or not len(r):
        return
    if l[0] == r[0]:
        q.append(l[0])
        return common(l[1:], r[1:], q)
    elif l[0] < r[0]:
        return common(l[1:], r, q)
    else:
        return common(l, r[1:], q)

def testCommon():
    q = []
    l = [2,4,7,9]
    r = [1,2,7]
    common(l,r,q)
    print 'common:', q

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
in-place merge use while ij loop, adjust i,j
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
            tmp = j
            while l[tmp] >= l[i]:
                tmp += 1
            l[i],l[tmp] = l[tmp],l[i]
            continue
    print l
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

"""
BST validation, all left smaller, all right bigger.
the key is, during the recursion, parent must be pass down to all its
subtrees to ensure non of them violates.
if no max/min parameter, we are checking subtree locally without knowing parent cap.
!!!! When recursion, carry the global cap!!!!
"""
def isBST(root, minval, maxval):   # pass in global cap, check globally
    if not root:
        return True

    if root.val > maxval or root.val < minval:
        return False

    # case 1, both left/right good
    if isBST(root.left, minval, root.val) and isBST(root.right, root.val, maxval):
        return True

    return False

'''
find the diameter(width) of a bin tree.
diameter is defined as the longest path between two leaf nodes.
diameter = max(diameter(ltree, rtree) = max( depth(ltree) + depth(rtree) )
    http://www.cs.duke.edu/courses/spring00/cps100/assign/trees/diameter.html
'''
def diameter(root):
    if not root:
        return 0
    ldepth = depth(root.lchild)
    rdepth = depth(root.rchild)

    ldiameter = diameter(root.lchild)
    rdiameter = diameter(root.rchild)

    return max(ldepth+rdepth+1, max(ldiameter, rdiameter))

'''
min vertex set, dfs, when process vertex late, i.e., recursion of 
all its children done and back to cur node, use greedy, if exisit
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


def depth(root):
    if not root:
        return 0
    return 1 + max(depth(root.lchild), depth(root.rchild))

'''
find the sum of one level.
when DFS recursion into a node from the parent, carry the info parent wants to tell the child
and child can use this global info to decide its status, prefix, level, etc
the sum at level is just the total sum of tree till level - tot sum of tree till level-1
'''
def getSum(root, level):
    ''' ret the sum while traversing the tree till the downtolevel'''
    def traverse(node, curlevel, downtolevel):
        sum = 0
        if curlevel == downtolevel:
            sum = node.val
            return sum
        for child in node.children:
            sum += traverse(child, curlevel+1)
        return sum

    return traverse(root, 1, level) - traverse(root, 1, level-1)


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

def findParent(root, node):
    if root.left == node or root.right == node:
        return root
    if root.val < node.val:
        findParent(root.right, node)
    else:
        findParent(root.left, node)

def findLeftMax(node):
    if not node.left:
        return node
    tmpnode = node.left
    while tmpnode:  # right most of the left tree
        tmpnode = tmpnode.right
    return tmpnode

def findRightMin(node):
    if not node.right:
        return node
    tmpnode = node.right
    while tmpnode:  # left most of right tree
        tmpnode = tmpnode.left
    return tmpnode

def findSuccessorByParent(node):
    if node.right:
        return findRightMin(node)

    cur = node
    par = node.parent
    while par.right == cur:
        par,cur = par.parent, par
    return par

''' min of right subtree, or first parent node whose left subtree has me'''
def findSuccessor(node):  #
    if node.right:
        return findRightMin(node)

    while root.val < node.val:  # the next bigger
        root = root.right
    while findLeftMax(root) != node:
        root = root.left

    return root

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


''' return the val, or the idx to insert the val
    first thing first, assign roles to variable, so we can update accordingly.
'''
def binSearch(l, val):
    i = 0   # assign roles first, so we can update variable accordingly.
    j = len(l)-1
    # with =, make sure the logic got run even for single ele.
    while i <= j:  # i==j, single ele, i=mid=0, break by j<i after i incr or j reduce.

        # mid takes math.floor, eqs i when i=j+1
        mid = i + (j-i)/2  # the same as i+j/2,
        if l[mid] == val:
            i = mid
            break
        elif l[mid] < val:  # test <, so i point to insert pos, which is after mid
            i = mid + 1     # i idx move up, [0...sz]
        else:
            j = mid - 1     # j idx move down [-1...sz-1]

    # i=[0..sz], j=[-1..sz-1],
    print l, val, '==>', i
    return i

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

''' pascal triangle, http://www.cforcoding.com/2012/01/interview-programming-problems-done.html
'''
def pascalTriangle(n):
    if n == 1:
        return [1]
    if n == 2:
        return [1,1]

    l = pascalTriangle(n-1)
    q = []
    q.append(l[0])
    for i in xrange(1, len(l), 1):
        q.append(l[i]+l[i-1])
    q.append(l[i])
    print n, ' ==> ', q
    return q

def pascalTriangle_iter(n):
    num = n
    l = []
    l.append([1,1])
    r = 3
    while r < n:
        pre = l[0]
        q = []
        q.append(1)

        mid = (r-1)/2
        for j in xrange(1,mid+1,1):
            q.append(pre[j]+pre[j-1])

        if r%2 > 0:  # odd
            k = mid-1
        else:  # even
            k = mid

        for z in xrange(k, 0, -1):
            q.append(q[z])

        q.append(1)
        print r, '==>', q
        l.pop()
        l.append(q)
        r += 1

def powerset(l):
    q = []
    for i in xrange(len(l)):
        for j in xrange(i+1, len(l), 1):
            q.append(str(l[i])+'-'+str(l[j]))
    print q


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

''' this class impl min heap
    node has key/value and l/r child,
'''
class MinHeap():
    class Node():
        def __init__(self, key, val, l, r):
            self.key = key
            self.val = val
            self.ij = (l,r)   # the i,j index in the two array
            self.lchild = l   # the lchild of this node in the tree
            self.rchild = r

        def toString(self):
            print 'Node : ', self.key, ' ', self.val, ' l:', self.lchild, ' r:', self.rchild
            return 'Node : ', self.key, ' ', self.val, ' l:', self.lchild, ' r:', self.rchild

        ''' recursive in-order iterate the tree this node is the root '''
        def inorder(self):
            if self.lchild:
                self.lchild.inorder()
            self.toString()
            if self.rchild:
                self.rchild.inorder()

    def __init__(self):
        self.Q = []

    def log(self, *k, **kw):
        print k

    def toString(self):
        self.log(' --- heap start ---- ')
        for n in self.Q:
            print n.key, n.val, n.lchild, n.rchild
        self.log(' --- heap end ---- ')

    def head(self):
        return self.Q[0]

    ''' node idx starts from 0 '''
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
        while True:
            pidx, pv = self.parent(idx)
            # swap when parent is bigger, min heap
            if pv and pv > self.Q[idx].val:
                self.swap(pidx, idx)
                idx = pidx
            else:
                return

    def siftdown(self, idx):
        while idx < len(self.Q)-1:
            lc, lcv = self.lchild(idx)
            rc, rcv = self.rchild(idx)
            if lcv and rcv:
                mincv = min(lcv, rcv)
                minc = lc if mincv == lcv else rc
            else:
                mincv = lcv if lcv else rcv
                minc = lc if lc else rc

            # min heap : sift down when node idx is bigger.
            if mincv and self.Q[idx].val > mincv:
                self.swap(idx, minc)
                idx = minc
            else:
                return  # done when loop break

    def insert(self, key, val, i, j):
        self.log('inserting : ', key, val, i, j)
        node = MinHeap.Node(key, val, i, j)
        self.Q.append(node)
        self.siftup(len(self.Q)-1)

    def extract(self):
        self.swap(0, len(self.Q)-1)
        head = self.Q.pop()      # remove the end, the extracted hearder, first
        head.toString()
        self.siftdown(0)
        return head

    def heapify(self):
        m = (len(self.Q)-1)/2
        for i in xrange(m, -1, -1):
            self.siftdown(i)

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
        x = self.head
        #  for each level top down, inside each level, while until stop on boundary.
        for level in reversed(xrange(len(self.head.next))):  # for each level down: park next node just before node >= ele.(insertion ponit).
            while x.next[level] != None and x.next[level].ele < ele:  # advance skipNode till to node whose next[level] >= ele, then down level
                x = x.next[level]
            expressEachLevel[level] = x
        return expressEachLevel

    def find(self, ele):
        expressList = findExpressStopAtEachLevel(self, ele)
        if len(expressList) > 0:
            candidate = expressList[0].next[0]
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
            for i in range(len(node.next)):
                node.next[i] = expressList[i].next[i]
                expressList[i].next[i] = node

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
1. max sub array problem. Kadane's Algo. 
    scan thru, compute at each pos i, the max sub array end at i. upbeat.
    if ith ele > 0, it add to max, if cur max > global max, update global max; 
    If it < 0, it dec max. If cur max down to 0, restart cur_max beg and end to next positive.
    
2. largest increasing sequence. 
    use a list to note down largest so far. scan next, append to end if next > end.
    otherwise, bisect to insert next into existing largest seq. when scan done, we get largest increasing seq at end.

3. min substring in S that contains all chars from T.
    slide thru, find first substring contains T, note down beg/end/count as max beg/end/count.
    then start to squeeze beg per every next, and upbeat global min string.

4. find in an array of num, max delta Aj-Ai, j > i. the requirement j>i hints 
    slide thru, at each pos, if it less than cur_beg, set cur_beg, if it > cur_end, set cur_end, and update global max.

"""
def max_subarray(A):
    max_ending_here = max_so_far = 0
    for x in A:
        max_ending_here = max(0, max_ending_here + x)  # incl x 
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

def minWindow(S, T):
    ''' find min substring in S that contains all chars from T'''
    matched_chars_len = 0     # matched char len
    need_to_match = defaultdict(int)    # need to match dict keyed by char
    matched_chars = defaultdict(int)
    cur_beg = cur_end = min_beg = min_eng = 0

    for c in T:   # populate 
        need_to_match[c] += 1
    # sliding thru src string S
    for idx, val in enumerate(S):
        if need_to_match[c] == 0:  # do not interested in this char
            continue
        matched_chars[val] += 1
        if matched_chars[val] <= need_to_match[val]:        
            matched_chars_len += 1
        if matched_chars_len == len(T):
            # squeeze left edge
            while need_to_match[cur_beg] == 0 || 
                  matched_chars[S[cur_beg]] > need_to_match[S[cur_beg]]:
                  if matched_chars[S[cur_beg]] > need_to_match[S[cur_beg]]:
                    matched_chars[S[cur_beg]] -= 1
                  cur_beg += 1
            # upbeat optimal solution
            if idx - cur_beg < min_end - min_beg:
                min_beg, min_end = cur_beg, idx

    return min_beg, min_end

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