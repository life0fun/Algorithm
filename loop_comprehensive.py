
import sys
import math
from collections import defaultdict
l = []
multi = [[x] for i in xrange(10)]
l.append(e)   # not ret l
l.pop() / l.popleft()
''.join(reversed(txt))
cnt = defaultdict(lambda: 0)
l = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
l = filter(lambda x: n%x==0, [x for x in xrange(2,100)])


''' recursive, one line print int value bin string'''
int2bin = lambda n: n>0 and int2bin(n>>1)+str(n&1) or ''
bin2int = lambda x: int(x,2)
twopower = lambda x: x & x-1

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

""" out[i+j+1]=a[j]*b[i] """
def multiplyStr(a,b):
  la,lb = list(a),list(b)
  out = [0]*(len(la)+len(lb))
  for ib in xrange(len(lb)-1,-1,-1):
    vb = int(lb[ib])
    for ia in xrange(len(la)-1,-1,-1):
      va = int(la[ia])
      out[ia+ib+1] += va*vb
      out[ia+ib] += out[ia+ib+1]/10
      out[ia+ib+1] %= 10
  return out
print multiplyStr("234","5678")

""" a[0..n]*b, out[i]=a[i]*b % 10 """
def multiply(a, b):
  arrb = map(int, list(str(b)))
  out = [0]*12  # num of digits in final value
  c = 0
  j = len(out)-1
  # go from b[n..1]
  for i in xrange(len(arrb)-1,-1,-1):
    iv = arrb[i]
    prod = iv*a + c
    res = prod%10
    c = prod/10
    out[j] = res
    j -= 1
  # now most significant numbers are in big carry.
  while c > 0:
    out[j] = c%10
    j -= 1
    c /= 10
  return out
print multiply(345, 678)

""" merge two sorted arry """
def merge(a,b):
  ia,ib,la,lb=0,0,len(a),len(b)
  out = []
  while ia < la and ib < lb:
    if a[ia] > b[ib]:
      out.append(a[ia])
      ia += 1
    else:
      out.append(b[ib])
      ib += 1
  for i in xrange(ia, la):
    out.append(a[i])
  for j in xrange(ib, lb):
    out.append(b[j])
  return out
print merge([9,8,3], [6,5])

""" the max num can be formed from k digits from a, preserve the order in ary.
key point is drop digits along the way."""
def maxKbits(a, k):
  drops = len(a)-k   # how many ele from a to be dropped
  out = []
  for i in xrange(len(a)):
    # when a[i] grt than last in out and we have room to drop
    while drops > 0 and len(out) > 0 and a[i] > out[-1]:
      out.pop()
      drops -= 1
    out.append(a[i])
  return out[:k]  # only keep first k bits
print reduce(lambda x,y: x*10+y, maxKbits([3,9,5,6,8,2], 2))

""" max k digit num can be formed from a, b, preserve the order in each ary 
idea is i digits from a, k-i digits from b, then merge.
"""
def maxNum(a,b,k):
  def merge(a,b):  # merge two sorted ary
    return [max(a,b).pop(0) for _ in a+b]
  mxnum = 0
  for i in xrange(k+1):
    lout = merge(maxKbits(a,i), maxKbits(b,k-i))
    v = reduce(lambda x,y:x*10+y, lout)
    mxnum = max(mxnum,v)
  return mxnum
print maxNum([3, 4, 6, 5], [9, 1, 2, 5, 8, 3], 5)


''' 1. pick a pivot, put to end
    2. start j one idx less than pivot, which is end-1.
    3. park i,j, i to first > pivot(i++ if l.i <= pivot), j to first <pivot(j-- if l.j > pivot)
    4. swap i,j and continue until i<j breaks
    4. swap pivot to i, the boundary of divide 2 sub string.
    5. 2 substring, without pivot in, recursion.
'''
''' qsort([5,8,5,4,5,1,5], 0, 6)
'''
def qsort(l, beg, end):
  def swap(i,j):
      l[i], l[j] = l[j], l[i]
  # always check loop boundary first!
  if(beg >= end):   # single ele, done.
      return

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
  qsort(l, beg, i-1)   # left half <= pivot
  qsort(l, j+1, end)     # right half > pivot
  print 'final:', l

def qsort(arr, lo, hi):
  def swap(arr, i, j):
    arr[i],arr[j] = arr[j],arr[i]
  def partition(arr, l, r):
    pivot = arr[r]
    wi = l
    for i in xrange(l,r):  # not incl r
      if arr[i] <= pivot:
        swap(arr, wi, i)
        wi += 1
    swap(arr, wi, r)
    return wi
  if lo < hi:  # only recursion when lo < hi.
    p = partition(arr, lo, hi)
    qsort(arr, lo, p-1)  # split into 3 segs, [0..p-1, p, p+1..n]
    qsort(arr, p+1, hi)
  return arr

""" Arrange given numbers to form the biggest number """
def qsort(arr,l,r):
  def comparator(arr, i, j):
    return int(str(arr[i])+str(arr[j])) - int(str(arr[j])+str(arr[i]))
  def swap(arr, i, j):
    arr[i],arr[j] = arr[j],arr[i]
  def partition(arr, l, r):
    wi,pidx,pval = l,r,arr[r]
    for i in xrange(l,r):
      if comparator(arr, i, pidx) <= 0:
        swap(arr, wi, i)
        wi += 1
      swap(arr, wi, pidx)
      return wi
  if l < r:
    p = partition(arr, l, r)
    qsort(arr, l, p-1)
    qsort(arr, p+1, r)
  return arr
arr = qsort([1, 34, 3, 98, 9, 76, 45, 4],0,7)
print int("".join(map(str, list(reversed(arr)))))

""" put 0 at head, 2 at tail, 1 in the middle, wr+1/wb-1 """
def sortColor(arr):
  def swap(arr, i, j):
    arr[i],arr[j] = arr[j],arr[i]
  i,wr,wb=0,0,len(arr)-1
  for i in xrange(len(arr)):
    while arr[i] == 2 and i < wb:
      swap(arr, i, wb)
      wb -= 1
    while arr[i] == 0 and i > wr:
      swap(arr, i, wr)
      wr += 1
  return arr
print sortColor([2,0,1])
print sortColor([1,2,1,0,1,2,0,1,2,0,1])


def hIndex(citations):
  def partition(arr, l, r):
    piv = min(arr, l, r, (l+r)/2)
    swap(arr, piv, r)
    wi = l
    for i in xrange(l, r):
      if arr[i] < arr[piv]:
        arr[wi++] = arr[i]
    swap(arr, wi, r)
    return wi

  sz,l,r,piv,ans = len(citations),0,sz-1,0,0
  while l > r:
    hidx = partition(arr, l, r)
    if citiations[hidx] >= hidx:
      ans = hdix
      l = hidx+1
    else:
      r = hidx-1
  return ans

""" use 0..n bucket to count """
def hIndex(citations):
  n = len(citations)
  bucket = [0]*(n+1)
  for i in xrange(n):
    bucket[min(citations[i],n)] += 1
  for i in xrange(n,0,-1):
    sm += bucket[i]
    if sm >= i:
      ret = i
  return ret

""" only search, enter loop even when only 1 ele, lo <= hi """
def binarySearch(arr, k):
  lo,hi = 0, len(arr)-1
  while lo <= hi:
    mid = lo+(hi-lo)/2
    if arr[mid] == k:
      return mid
    elif arr[mid] < k:
      lo = mid + 1
    else:
      hi = mid - 1
  return -1

""" ret insertion pos, the first pos t <= a[idx]. a[idx-1] < t <= a[idx]. even with dups.
when find high end, idx-1 is the first ele smaller than searching val."""
def bisect(arr, val):
  lo,hi = 0, len(arr)-1
  if val > arr[-1]:
    return False, len(arr)
  # lo != hi, reduced to at least 2 ele, so l=m+1 wont boundary.
  while lo != hi:
    md = (lo+hi)/2
    if val > arr[md] # target grt than mid, ins pos m+1
      lo = md+1
    else:
      hi = md   # target <=, mov r, as we want to find first pos eqs target
  # check lo when out lo = hi
  if arr[lo] == val:
    return True, lo
  else:
    return False, lo
print bisect([1,1,3,5,7])

""" not to find insertion point, check which seg is mono """
def binarySearchRotated(arr,t):
  l,r=0,len(arr)-1
  while l <= r:      # if =, will enter loop even when only one ele.
    mid = l+(r-l)/2
    if t == arr[mid]:
      return True, mid
    # l != mid, so mid+1 wont boundary.
    if arr[l] < arr[mid]:  #[l...mid] is mono increase
      if arr[l] <= t and t < arr[mid]:
        r = mid - 1
      else:
        l = mid + 1
    elif arr[l] > arr[mid]:  # [mid...r] is mono increase
      if arr[mid] < t and t <= arr[r]:
        l = mid + 1
      else:
        r = mid - 1
    else:
      l += 1
  # when out loop not return, not found.
  return False, None
print bisectRotated([3, 4, 5, 1, 2], 2)
print bisectRotated([3], 2)

""" find min in rotated. with dup, advance mid, and check on each mov """
def rotatedMin(arr):
  l,r=0,len(arr)-1
  while l != r:     # into loop only when >= 2 ele, ret l when out
    mid = (l+r)/2
    if arr[mid] > arr[mid+1]:
      return arr[mid+1]
    # first, mov mid to skip dups. if arr[l]=arr[r] pivot is on left
    while arr[l] == arr[mid] and mid < r:
      # check at each mov of mid
      if arr[mid] > arr[mid+1]:
        return arr[mid+1]
      mid += 1
    if mid == r:  # two ends eqs, cut in dup, min on left,  /--|--/ => ---/ \ /---
      r = (l+r)/2  # r = mid
      continue
    ''' l - mid in the first same segment '''
    if arr[l] < arr[mid] and mid < r:
      l = mid+1
    else:
      r = mid     # r=mid-1 only when loop in when l <= r
  return arr[l]
print rotatedMin([4,5,0,1,2,3])
print rotatedMin([4,5,5,6,6,6,6,0,1,4,4,4,4,4,4])
print rotatedMin([4,5,5,6,6,6,6,0,1,4,4,4,4,4,4,4,4])
print rotatedMin([6,6,6,0])
print rotatedMin([0,1])
print rotatedMin([0,1,3])
print rotatedMin([5,6,6,7,0,1,3])
print rotatedMin([5,5,5,5,6,6,6,6,6,6,6,6,6,0,1,2,2,2,2,2,2,2,2])


""" find the max in a ^ ary """
def findMax(arr):
  l,r=0,len(arr)-1
  while l < r:
    mid = l+(r-l)/2
    if arr[mid-1] < arr[mid] > arr[mid+1]:
      return arr[mid]
    if arr[mid] < arr[mid+1]:  # mid .. mid+1 incr, max is on rite
      l = mid+1
    else:         # mid .. mid+1 decrease, max on left
      r = mid
  return arr[r]
print findMax([8, 10, 20, 80, 100, 200, 400, 500, 3, 2, 1])
print findMax([10, 20, 30, 40, 50])
print findMax([120, 100, 80, 20, 0])


''' bisect always check l,r boundary first a[l-1]<t<=a[l] '''
def floorceil(arr, t):
  def floor(arr, t):
    l,r = 0, len(t)-1
    while l < r:
      mid = l + (r-l)/2
      if arr[mid] < t:
        l = mid+1
      else:
        r = mid
    return arr[l] == t ? l : -1
  def ceiling(arr, t):
    l,r = 0, len(t)-1
    while l < r:
      mid = l + (r-l)/2
      if arr[mid] > t:
        r = mid-1
      else:
        l = mid
    return r
  l = floor(arr, t)
  if l == -1:
    return [-1, -1]
  r = ceiling(arr,t)
  return [l, r]
print floorceil([1, 2, 8, 10, 10, 12, 19], 0)
print floorceil([1, 2, 8, 10, 10, 12, 19], 2)
print floorceil([1, 2, 8, 10, 10, 12, 19], 28)


""" inverse count with merge sort """
def mergesort(arr, lo, hi):
  def merge(arr, ai, bi, hi):
    out = []
    i,j,k=ai,bi,hi
    cnt = 0
    while i < bi and j < hi:
      if arr[i] <= arr[j]:
        out.append(arr[i])
        i += 1
      else:
        out.append(arr[j])
        cnt += j-i  # entire left from i->end is bigger than rite[j]
    while i < bi:
      out.append(arr[i])
      i += 1
    while j < hi:
      out.append(arr[j])
      j += 1
    return cnt
  inv = 0
  if lo < hi:
    mid = (lo+hi)/2
    inv = mergesort(arr, lo, mid)
    inv += mergesort(arr, mid+1, hi)
    inv += merge(arr, lo, mid+1, hi)
  return inv

""" rite smaller with merge sort, when merging left and rite,
left[l] > rite[0..r-1], because we mov r when rite[r] < left[l].
so rank[l]+= r.
"""
def riteSmaller(arr):
  def merge(left, rite):
    ''' left rite ary is sorted and contains idx of ary '''
    l,r = 0,0
    out = []
    while l < len(left) and r < len(rite):
      lidx,ridx = left[l],rite[r]
      if arr[lidx] < arr[ridx]:  # mov l until could not. rank+=r
        rank[lidx] += r
        out.append(lidx)
        l += 1
      else:
        out.append(ridx)
        r += 1
    for i in xrange(l, len(left)):
      out.append(left[i])
      rank[left[i]] += r
    for i in xrange(r, len(rite)):
      out.append(rite[i])
    return out  
  def mergesort(arr, l, r):
    if l == r:
      return [l]
    m = (l+r)/2
    left = mergesort(arr, l, m)    # ret ary of idx of origin arr
    rite = mergesort(arr, m+1, r)
    return merge(left,rite)

  rank = [0]*len(arr)
  mergesort(arr, 0, len(arr)-1)
  return rank
print riteSmaller([5, 4, 7, 6, 5, 1])

""" find all pairs sum[l:r] within range [mn,mx]. merge sort
arr, divide and conquer, better than gap n^2.
master therom: T(0, n-1) = T(0, m) + T(m+1, n-1) + C
if the merge part is Constant, it's less than n^2.
To make merge constant, sort the arr.
"""
def countRangeSum(arr,mn,mx):
  ''' merge two sorted arr '''
  def mergeSortedAry(arr1, arr2):
    r, i, j = [], 0, 0
    while i < len(arr1) and j < len(arr2):
      if arr1[i] < arr2[j]:
        r.append(arr1[i])
        i += 1
      else:
        r.append(arr2[j])
        j += 1
      r += arr1[i:] + arr2[j:]
    return r

  def mergesort(arr, prefix, start, end, lower, upper):
    if start >= end:
        return 0
    mid = start + (end - start + 1 >> 1)
    lout = mergesort(arr, prefix, start, mid-1, lower, upper)
    rout = mergesort(arr, prefix, mid, end, lower, upper)

    l, r = mid, mid
    for i in xrange(start, mid):
      while l <= end and prefix[l] - prefix[i] < lower:
        l += 1
      while r <= end and prefix[r] - prefix[i] <= upper:
        r += 1
      count += r - l

    prefix[start:end+1] = mergeSortedAry(
      prefix[start:mid], prefix[mid:end + 1])

    return count

  ''' use prefixsum ary, replacing the origin ary '''
  def mergesort(prefixsum, lo, hi, mn, mx):
    if lo == hi:
      return [prefixsum[lo]] if mn <= prefixsum[lo] <= mx
    mid = (lo+hi)/2
    out = []
    lout = mergesort(prefixsum, lo, mid)
    rout = mergesort(prefixsum, mid+1, hi)
    # check range l..mid and l..end
    mnidx,mxidx = mid+1,hi
    for l in prefixsum[lo:mid+1]:
      while prefixsum[mnidx] - prefixsum[l] < mn:
        mnidx += 1  # mv rite until it bigger
      while prefixsum[mxidx] - prefixsum[l] > mx:
        mxidx -= 1
      out.append([l, mnidx, mxidx])
    out.extend(lout, rout)
    # sort the prefixsum range lo..hi to make merge constant.
    sorted(prefixsum, lo, hi)
    return out
  # use prefix sum ary
  prefixsum = [0]*len(arr)
  for i in xrange(1, len(arr)):
    prefixsum[i] = prefixsum[i-1] + arr[i]
  return mergesort(prefixsum, 0, len(prefixsum), mn, mx)

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
  if m > n:
      return findKth(B,n,A,m,k)
  if m == 0: return B[k-1]
  if k == 1: return min(A[0],B[0])
  ia = min(k/2, m)
  ib = k-ia

  if A[ia-1] < B[ib-1]:
    return findKth(A[ia:], m-ia, B, n, k-ia)
  elif A[ia-1] > B[ib-1]:
    return findKth(A, m, B[ib:], n-ib, k-ib)
  else:
    return A[ia-1]
def testFindKth():
  A = [1, 5, 10, 15, 20, 25, 40]
  B = [2, 4, 8, 17, 19, 30]
  Z = sorted(A + B)
  assert(Z[9] == findKth(A, len(A), B, len(B), 10))
  assert(Z[0] == findKth(A, len(A), B, len(B), 1))
  assert(Z[12] == findKth(A, len(A), B, len(B), 13))

""" 
  selection algorithm, find the kth largest element with worst case O(n)
  http://www.ardendertat.com/2011/10/27/programming-interview-questions-10-kth-largest-element-in-array/
"""
def selectKth(A, beg, end, k):
  def partition(A, l, r, pivotidx):
    A.r, A.pivotidx = A.pivotidx, A.r   # first, swap pivot to end
    widx = l   # all ele smaller than pivot, stored in left, started from begining
    for i in xrange(l, r):
      if A.i <= A[r]: # r is pivot value
        A.widx, A.i = A.i, A.widx
        widx += 1
    A.r, A.widx = A.widx, A.r
    return widx  # the start idx of all items > than pivot

  if beg == end: return A.beg
  if not 1 <= k < (end-beg): return None

  while True:   # continue to loop 
    pivotidx = random.randint(l, r)
    rank = partition(A, l, r, pivotidx)  # the start idx of items grter than pivot.
    if rank == k: return A.rank
    if rank < k: return selectKth(A, l, rank, k-rank)
    else: return selectKth(A, rank, r, k-rank)

# median of median of divide to n groups with each group has 5 items,
def medianmedian(A, beg, end, k):
  def partition(A, l, r, pivot):
    widx = l
    for i in xrange(l, r+1):
      if A.i < pivot:
        A.widx, A.i = A.i, A.widx
        widx += 1
    # i will park at last item that is < pivot
    return widx-1   # left shift as we always right shift after swap.

  sz = end - beg + 1  # beg, end is idx
  if not 1<= k <= sz: return None     # out of range, None.
  # ret the median for this group of 5 items
  if end-beg <= 5:
      return sorted(A[beg:end])[k-1]  # sort and return k-1th item
  # divide into groups of 5, ngroups, recur median of each group, pivot = median of median
  ngroups = sz/5
  medians = []
  for i in xranges(ngroups):
    gm = medianmedian(A, beg+5*i, beg+5*(i+1)-1, 3)
    medians.append(gm)
  pivot = medianmedian(medians, 0, len(medians)-1, len(medians)/2+1)
  pivotidx = partition(A, beg, end, pivot)  # scan from begining to find pivot idx.
  rank = pivotidx - beg + 1
  if k <= rank:
    return medianmedian(A, beg, pivotidx, k)
  else:
    return medianmedian(A, pivotidx+1, end, k-rank)

def removeDup(arr):
  pos, start = 1,2
  for i in xrange(start, len(arr)):
    if arr[i] == arr[pos-1] && arr[i] == arr[pos]:
      continue
    arr[pos] = arr[i]
    pos += 1
  return pos


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

''' uniq id generator '''
insta5.next_id(OUT result bigint)
    our_epoch bigint := 1314220021721;
    seq_id bigint;
    now_millis bigint;
    shard_id int := 5;
BEGIN
    SELECT nextval('insta5.table_id_seq') %% 1024 INTO seq_id;

    SELECT FLOOR(EXTRACT(EPOCH FROM clock_timestamp()) * 1000) INTO now_millis;
    result := (now_millis - our_epoch) << 23;
    result := result | (shard_id << 10);
    result := result | (seq_id);
END;


""" walk with f, where f is for leaf node. for list form, map(walk f ...)
map a curry fn that is recursive calls walk f to non-leaf node """
(defn walk
  [f form]
  (let [pf (partial walk f)]
    (if (coll? form)
      (f (into (empty form) (map pf form)))
      (f form))))

""" outer fn is apply to leaf node. inner fn is for nested list forms.
map inner fn into list
inner is fn need to apply to nested form. outer is fn apply to non-nest form
"""
(defn walk
  "Traverses form, an arbitrary data structure.  inner and outer are
  functions.  Applies inner to each element of form, building up a
  data structure of the same type, then applies outer to the result.
  Recognizes all Clojure data structures. Consumes seqs as with doall."
  
  [inner outer form]
  (cond
   (list? form) (outer (apply list (map inner form)))
   (instance? clojure.lang.IMapEntry form) (outer (vec (map inner form)))
   (seq? form) (outer (doall (map inner form)))
   (instance? clojure.lang.IRecord form)
     (outer (reduce (fn [r x] (conj r (inner x))) form form))
   (coll? form) (outer (into (empty form) (map inner form)))
   :else (outer form)))

""" f is fn apply to leaf form. for non-leaf/list, recursive apply f into """
(defn postwalk
  "Performs a depth-first, post-order traversal of form.  Calls f on
  each sub-form, uses f's return value in place of the original.
  Recognizes all Clojure data structures. Consumes seqs as with doall."
  [f form]
  (walk (partial postwalk f) f form))

(defn prewalk
  "Like postwalk, but does pre-order traversal."
  {:added "1.1"}
  [f form]
  (walk (partial prewalk f) identity (f form)))

""" keep track what node has been cloned."""
def clone(root, storeMap):
  if not storeMap.get(root):
    nroot = Node(root)
    storeMap.set(root, nroot)
  else:
    nroot = storeMap.get(root)
  for c in children(root):
    nc = clone(c, storeMap)
    nroot.children.append(nc)
  return nroot


""" track pre, cur when traverse, and threading left rite most """
def morris(root):
  pre, cur = None, root
  while cur:
    if not cur.left:  # no left, down to rite directly
      out.append(cur)
      pre = cur
      cur = cur.rite
    else:
      # threading left's rite most to root rite before descend
      tmp = cur.left
      # descend to left's rite most child
      while tmp.rite and tmp.rite != cur:
        tmp = tmp.rite

      if tmp.rite == None:
        tmp.rite = cur      # threading to cur
        cur = cur.left      # descend to left
      else:  # already threaded, visit, reset, to rite
        out.append(cur)     # visit
        tmp.rite = None
        pre,cur = cur, cur.rite

""" threading rite child's left most's left to root's left """
def postorderMorris(self, root):
  out = []
  while root:
    out.append(root)
    if root.rite:     # threading rite's left most to parent left
      tmp = root.rite
      while tmp.left:
        tmp = tmp.left
      # rite child's left most child point to parent left
      tmp.left = root.left
      root = root.rite
    else:
      root = root.left  # descend to left
  return out[::-1]

""" pre-order with rite first, then reverse, got post-order 
At each cur, shall move to rite, but first stash cur's left before move.
"""
def postOrder(root):
  stk = []
  pre,cur = None,root
  while cur or len(stk) > 0:
    if cur:
      out.insert(0, cur)
      if cur.left:
        stk.append(cur.left)
      cur = cur.rite
    else:
      cur = stk.pop()
  return out

def postorderTraversal(self, root):
  ans, stack = [], [root]
  while stack:
    tmp = stack.pop()
    if tmp:
      ans.append(tmp.val)
      stack.append(tmp.left)
      stack.append(tmp.right)
  return ans[::-1]

""" stack to stash r or r.rite, when stash r, stk.pop, cur=cur.rite. avoid redo r.left
process cur node, replace cur node from stack if null 
"""
def flatten(r):
  while stk or r:
    if r:   # need visit root, mov to r.left, stash r.rite for later
      # visit cur, replace cur from stk top if leaf, descend to rite
      if not r.left and r.rite:
        r.rite = stk.top()
        r = stk.pop()
      else:
        if r.rite:
          stk.append(r.rite)  # stash rite before move
      r.rite = r.left
      r = r.rite  # advance, if not r, will pop stk
    else:
      r = stk.pop()

def flattenlist(arr):
  stk = []
  stk.append([0, arr])
  while stk:
    idx, l = stk.pop()
    if l[idx] is not list:
      out.append(l[idx])
      stk.append([idx+1,l])
    else:
      stk.append([idx+1,l])
      stk.append([0, l[idx]])


''' tab[i] = num of bst for ary 1..i, tab[i] += tab[k]*tab[i-k] '''
def numBST(n):
  tab = [0]*(n+1)
  tab[0] = tab[1] = 1
  for i in xrange(2,n):
    for r in xrange(1,i+1):
      tab[i] += tab[r-1]*tab[i-r]   # each lchild combine with each rchild. lxr
  return tab[n]


""" find pair node in bst tree sum to a given value, lgn 
down to both ends(lstop/lcur), with stk to rem parent, then mov rite or left """
def pairNode(root, target):
  lstk,rstd = deque(),deque()  # use lgn stk to store tree path parent
  lstop,rstop = False,False
  lcur,rcur = root,root
  while True:
    while not lstop:
      if lcur:
        lstk.append(lcur)
        lcur = lcur.left
      else:
        if not len(lstk):
          lstop = 1
        else:
          lcur = lstk.pop()
          lval = lcur
          lcur = lcur.rite
          lstop = 1
      while not rstop:
        if rcur:
          rstk.append(rcur)
          rcur = rcur.rite
        else:
          if not len(rstk):
            rstop = 1
          else:
            rcur = lstk.pop()
            rval = rcur
            rcur = rcur.left
            rstop = 1
      if lval + rval == target:
        return lcur,rcur
      if lval + rval > target:
        rstop = False
      else:
        lstop = False
  return lcur,rcur


def match(s,p):
  if p[0] == "*":
    while p[0] == "*":
      p += 1
    while not match(s,p[1:]):
      s += 1
    if not s:
      return False
    else:
      return True
  else:
    if s[0] == p[0]:
      return match(s[1:], p[1:])
    else:
      return False

""" check p+1 is star """
def regMatch(s, p):
  if p[1] == "*":
    while s and (s[0] == p[0] or p[0] == "."):
      if regMatch(s,p+2):
        return True
      s += 1
    return regMatch(s,p+2)
  else:
    return (s[0] == p[0] or p[0] == ".") and regMatch(s+1,p+1)


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

""" serde tree with arbi # of children """
def dfsSerde(root, out):
  print root
  out.append(root)
  for c in root.children:
    dfsSerde(c, out)
  print "$"
  out.append("$")

""" deserialize, use stk top is parent of cur """
# 1 [2 [4 $] $] [3 [5 $] $] $
from collections import deque
def deserTree(l):
  stk = deque()
  root = None
  while nxt = l.readLine():
    if nxt is not "$":
      if len(stk) > 0:
        p = stk.top()
        p.children.append(nxt)
      stk.append(nxt)
    else:
      root = stk.pop()
  return root

""" no same char shall next to each other """
from collections import defaultdict
def noAdj(text):
    cnt = defaultdict(lambda: 0)
    sz = len(text)
    for i in xrange(sz):
        cnt[text[i]] += 1
    l = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    start = 0
    out = [None]*sz
    for e in l:
        gap = sz/e[1]
        for s in xrange(e[1]):
            pos = start + s*gap
            while out[pos]:
                pos += 1
                if pos >= sz:
                return False, out
            out[pos] = e[0]
        start += 1
    return True, out

print noAdj("abacbcdc")
print noAdj("aabbc")

""" AVL tree with rank, getRank ret num node smaller. left/rite rotate.
    Ex: count smaller ele on the right, find bigger on the left, max(a[i]*a[j]*a[k])
"""
class AVLTree(object):
  def __init__(self,key):
    self.key = key
    self.size = 1
    self.height = 1
    self.left = None
    self.rite = None
  def __repr__(self):
      return str(self.key) + " size " + str(self.size) + " height " + str(self.height)
  # left heavy, pos balance, right heavy, neg balance
  # rite subtree left heavy, >, LR rotation. left subtree rite heavy, <, RL.
  def balance(self):
    if not self.left and not self.rite:
      return 0
    if self.left and self.rite:
      return self.left.height - self.rite.height
    if self.left:
      return self.left.height
    else:
      return self.rite.height
  def updateHeightSize(self):
    self.height = 0
    self.size = 0
    if self.left:
      self.height = self.left.height
      self.size = self.left.size
    if self.rite:
      self.height = max(self.height, self.rite.height)
      self.size += self.rite.size
    self.height += 1
    self.size += 1
  def rotate(self, key):
    bal = self.balance()
    if bal > 1 and self.left and key < self.left.key:
        return self.riteRotate()
    elif bal < -1 and self.rite and key > self.rite.key:
        return self.leftRotate()
    elif bal > 1 and self.left and key > self.left.key: # < , left rotate left subtree.
        self.left = self.left.leftRotate()
        return self.riteRotate()  # then rite rotate after /
    elif bal < -1 and self.rite and key < self.rite.key: # >, rite rotate rite subtree.
        self.rite = self.rite.riteRotate()
        return self.leftRotate()  # left rotate after \
    return self  # no rotation
  ''' bend / left heavy subtree to ^ subtree '''
  def riteRotate(self):
    newroot = self.left
    newrite = newroot.rite
    newroot.rite = self
    self.left = newrite

    self.updateHeightSize()
    newroot.updateHeightSize()
    return newroot
  ''' bend \ subtree to ^ subtree '''
  def leftRotate(self):
    newroot = self.rite
    newleft = newroot.left
    newroot.left = self
    self.rite = newleft
    
    self.updateHeightSize()
    newroot.updateHeightSize()
    return newroot
  def searchkey(self, key):
    if self.key == key:
      return True, self
    elif key < self.key:
      if self.left:
        return self.left.search(key)
      else:
        return False, self
    else:
      if self.rite:
        return self.rite.search(key)
      else:
        return False, self
  ''' max smaller of the tree '''
  def maxSmaller(self, key):
    if key < self.key:
      if not self.left:
        return None   # no smaller in this sub tree.
      else:
        return self.left.maxSmaller(key)
    if key > self.key:
      if not self.rite:
        return self    # cur is largest smaller
      else:
        rtmx = self.rite.maxSmaller(key)
        node = rtmx ? rtmx : self
        return node
    return self
  ''' rank is num of node smaller than key '''
  def getRank(self, key):
    if key == self.key:
      if self.left:
        return self.left.size + 1
      else:
        return 1
    elif self.key > key: 
      if self.left:
        return self.left.getRank(key)
      else:
        return 0
    else:
      r = 1
      if self.left:
          r += self.left.size
      if self.rite:
          r += self.rite.getRank(key)
      return r
  # return subtree root, and # of node smaller or equal
  def insert(self,key):
    if self.key == key:
      return self
    elif key < self.key:
      if not self.left:
        self.left = AVLTree(key)
      else:
        self.left = self.left.insert(key)
    else:
      if not self.rite:
        self.rite = AVLTree(key)
      else:
        self.rite = self.rite.insert(key)
    # after insersion, rotate if needed.
    self.updateHeightSize()  # rotation may changed left. re-calculate
    newself = self.rotate(key)  ''' current node changed after rotate '''
    print "insert ", key, " new root after rotated ", newself
    return newself  # ret rotated new root
  # delete a node.
  def delete(self,key):
    if key < self.key:
      self.left = self.left.delete(key)
    elif key > self.key:
      self.rite = self.rite.delete(key)
    else:
      if not self.left or not self.rite:
        node = self.left
        if not node:
            node = self.rite
        if not node:
            return None
        else:
            self = node
      else:
        node = self
        while node.left:
            node = node.left
        self.key = node.key
        self.rite = self.rite.delete(node.key)
    self.updateHeightSize()
    subtree = self.rotate(key)
    return subtree


# rite Smaller, and left smaller, the same.
def riteSmaller(arr):
    sz = len(arr)
    root = AVLTree(arr[-1])
    riteSmaller = [0]*sz
    for i in xrange(sz-2, -1, -1):
        v = arr[i]
        rank = root.getRank(v)
        root = root.insert(v)
        riteSmaller[i] = rank
    return riteSmaller

assert(riteSmaller([12, 1, 2, 3, 0, 11, 4]) == [6, 1, 1, 1, 0, 1, 0] )
assert(riteSmaller([5, 4, 3, 2, 1]) == [5, 4, 3, 2, 1])


""" bst with lo as the bst search key, augment max to the max of ed of all nodes under root.
    interval tree annotate with max.
"""
class IntervalTree(object):
  def __init__(self, lo=None, hi=None):
    self.lo = lo
    self.hi = hi
    self.max = max(lo, hi)
    self.left = self.rite = None
  def overlap(self, intv):
    [lo,hi] = intv
    if lo > self.hi or hi < self.lo:
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

''' find max val from overlapping intervals '''
def maxIntervals(arr):
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
class Interval(object):
  def __init__(self):
      self.arr = []
      self.size = 0
  # bisect ret i as insertion point. arr[i-1] < val <= arr[i]
  # when find high end, idx-1 is the first ele smaller than searching val.
  def bisect(self, arr, val):
    lo,hi = 0, len(self.arr)-1
    if val > arr[-1]:
      return False, len(arr)
    while lo != hi:
      md = (lo+hi)/2
      if arr[md] < val:  # lift lo only when absolutely big
          lo = md+1
      else:
          hi = md   # drag down hi to mid when equal to find begining.
    if arr[lo] == val:
      return True, lo
    else:
      return False, lo
  def overlap(self, st1, ed1, st2, ed2):
    if st2 > ed1 or ed2 < st1:
      return False
    return True
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

"""
https://leetcode.com/discuss/79907/summary-divide-conquer-based-binary-indexed-based-solutions
count range sum. i,j so low < prefixsum(j-i) < upper.
"""
def rangeSum(nums, lower, upper):
  return 0 if len(nums) == 0
  prefix = [0]*(len(nums)+1)
  for i in xrange(len(nums)):
    prefix[i+1] = prefix[i] + nums[i]

  return prefixSum(prefix, 0, len(prefix)-1, lower, upper);
  
  def prefixSum(prefix, l, r, lower, upper):
    return 0 if l >= r

    m = l + (r - l) / 2;
    left = prefixSum(prefix, l, m, lower, upper)
    rite = prefixSum(prefix, m+1, r, lower, upper)
    count = left + rite

    merged = [0]*(r - l + 1)
    i = l, j = k = q = m + 1, p = 0
    while i <= m:
      while j <= r and prefix[j] - prefix[i] < lower:
        j++;
      while k <= r and prefix[k] - prefix[i] <= upper:
        k++;
      count += k - j;

      while q <= r and prefix[q] < prefix[i]:
        merged[p++] = prefix[q++]
      merged[p++] = prefix[i++]

    while q <= r:
      merged[p++] = prefix[q++]

    System.arraycopy(merged, 0, prefix, l, len(merged)-1);

    return count;

""" segment tree. first segment 1 cover 0-n, second segment covers 0-mid, seg 3, mid+1..n. Segment num is btree, child[i] = 2*i,2*i+1.
search always starts from seg 0, and found which segment [lo,hi] lies.
"""
import math
import sys
class RMQ(object):
  class Node():
    def __init__(self, lo=0, hi=0, minval=0, minvalidx=0, segidx=0):
      self.lo = lo
      self.hi = hi
      self.segidx = segidx      # segment idx in rmq heap tree.
      self.minval = minval      # range minimal, can be range sum
      self.minvalidx = minvalidx  # idx in origin val arr
      self.left = self.rite = None
  def __init__(self, arr=[]):
    self.arr = arr  # original arr
    self.size = len(self.arr)
    self.heap = [None]*pow(2, 2*int(math.log(self.size, 2))+1)
    # build the seg tree with root segidx
    self.root = self.build(0, self.size-1, 0)[0]
  # 0th segment covers entire, 1th, left half, 2th, rite half
  def build(self, lo, hi, segRootIdx):  # root segRootIdx, lc/rc of segRootIdx
    if lo == hi:
      n = RMQ.Node(lo, hi, self.arr[lo], lo, segRootIdx)
      self.heap[segRootIdx] = n
      return [n, lo]
    mid = (lo+hi)/2
    left,lidx = self.build(lo, mid, 2*segRootIdx+1)    # lc of segRootIdx
    rite,ridx = self.build(mid+1, hi, 2*segRootIdx+2)  # rc of segRootIdx
    minval = min(left.minval, rite.minval)
    minidx = lidx if minval == left.minval else ridx
    n = RMQ.Node(lo, hi, minval, minidx, segRootIdx)
    n.left = left
    n.rite = rite
    self.heap[segRootIdx] = n
    return [n,minidx]
  # segment tree in bst, search always from root.
  def rangeMin(self, root, lo, hi):
    # out of cur node scope, ret max, as we looking for range min.
    if lo > root.hi or hi < root.lo:
      return sys.maxint, -1
    if lo <= root.lo and hi >= root.hi:
      return root.minval, root.minvalidx
    lmin,rmin = root.minval, root.minval
    # ret min of both left/rite.
    if root.left:
      lmin,lidx = self.rangeMin(root.left, lo, hi)
    if root.rite:
      rmin,ridx = self.rangeMin(root.rite, lo, hi)
    minidx = lidx if lmin < rmin else ridx
    return min(lmin,rmin),minidx
  # segment tree in heap tree. always search from root, idx=0
  def queryRangeMinIdx(self, idx, lo, hi):
    node = self.heap[idx]
    if lo > node.hi or hi < node.lo:
      return sys.maxint, -1
    if lo <= node.lo and hi >= node.hi:
          return node.minval, node.minvalidx
      lmin,rmin = node.minval, node.minval
      if self.heap[2*idx+1]:
          lmin,lidx = self.queryRangeMinIdx(2*idx+1, lo, hi)
      if self.heap[2*idx+2]:
          rmin,ridx = self.queryRangeMinIdx(2*idx+2, lo, hi)
      minidx = lidx if lmin < rmin else ridx
      return min(lmin,rmin), minidx

def build(self, arr, lo, hi):
  if lo > hi:
    return None
  node = RMQ.Node(lo, hi)
  if lo == hi:
    node.sum = arr[lo]
  else:
    mid = (lo+hi)/2
    node.left = self.build(lo, mid)
    node.rite = self.build(mid+1, hi)
    node.sum = node.left.sum + node.rite.sum
  return node
def update(self, idx, value):
  if self.lo == self.hi:
    self.sum = value
  else:
    m = (node.lo + node.hi)/2
    if idx <= m:
      update(self.left, idx, value)
    else:
      update(self.rite, idx, value)
    self.sum = self.left.sum + self.rite.sum
def sumRange(self, lo, hi):
  if self.lo == lo and self.hi == hi:
    return self.sum
  m = (node.lo + node.hi)/2
  if hi <= m:
    return sumRange(self.left, lo, hi)
  elif lo >= m+1:
    return sumRange(self.rite, lo, hi)
  return sumRange(self.left, lo, m) + sumRange(self.rite, m+1, hi)

def test():
    r = RMQ([4,7,3,5,12,9])
    print r.rangeMin(r.root,0,5)
    print r.queryRangeMinIdx(0,0,5)


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

""" 
    node contains only 3 pointers, left < eq < rite, always descend down along eq pointer.
"""
class Trie:
  def __init__(self, val):
    self.val = val
    self.next = defaultdict(TrieNode)
    self.leaf = False
class TernaryTree(object):
  def __init__(self, key=None):
    self.key = key
    self.cnt = 1  # when leaf is true, the cnt, freq of the node
    self.leaf = False
    self.left = self.rite = self.eq = None
  def insert(self, word):
    hd = word[0]
    if not self.key:  # at first, root does not have key.
      self.key = hd
    if hd == self.key:
      if len(word) == 1:
        self.leaf = True
        return self
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
      if len(word) == 1:
        return True, self # when leaf flag is set
      elif self.eq:
        return self.eq.search(word[1:])
      else:
        return False, self
    elif hd < self.key:
      if self.left:
        return self.left.search(word)
      else:
        return False, self
    elif hd > self.key:
      if self.rite:
        return self.rite.search(word)
      else:
        return False, self
  def all(self, prefix):
    result = set()
    found, node = self.search(prefix)
    if not found:
      return result
    if node.leaf:
      result.add(prefix)
    if node.eq:   # dfs on eq branch of prefix parent only.
      node.eq.dfs(prefix, result)
    return result
  def dfs(self, prefix, result):
    if self.leaf:
      result.add(prefix+self.key)
    if self.left:
      self.left.dfs(prefix, result)
    if self.eq:
      self.eq.dfs(prefix+self.key, result)
    if self.rite:
      self.rite.dfs(prefix, result)
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

""" find all words in the board, dfs, backtracking and Trie """
def wordSearch(board, words):
  out = []
  def buildTrie(words):
    root = Trie(None)
    for w in words:
      cur = root
      for c in w:
        if not cur.next[c]:
          cur.next[c] = Trie(c)
        cur = cur.next[c]
      cur.leaf = True
      cur.word = w
  def dfs(board, i, j, node):
    c = board[i][j]
    nxtnode = node.next[c]
    if not nxtnode or c == "visited":
      return
    if nxtnode.leaf:
      out.append(nxtnode.word)
      return
    board[i][j] = "visited"
    for nxti,nxtj in nextmove(i,j):
      dfs(board, nxti, nxtj, nextnode) # recur with next trie node
    board[i][j] = c
    board[i][j] = "un-visited"
  root = buildTrie(words)
  # for each cell, do dfs
  for i,j in board:
    dfs(board, i, j, root)
  return out


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
all its next done and back to cur node, use greedy, if exist
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

def liss(root):
  root.liss = max(liss(root.left)+liss(root.rite), 1+liss(root.[left,rite].[left,rite]))


""" populate next right pointer to point sibling in each node
    we can do bfs, level by level. or do recursion.
"""
def populateSibling(root):
  if not root:
      return root
  if root.lchild:
      root.lchild.sibling = root.rchild
  if root.rchild:
      root.rchild.sibling = root.sibling ? root.sibling.lchild : None
  populateSibling(root.lchild)
  populateSibling(root.rchild)

def popNext(root):
  nexthd, nextpre = None,None
  cur = root
  while cur:
    for c in cur.children:
      if not nexthd:
        nexthd = c
        nextpre = c
      else:
        nextpre.next = c
      nextpre = c
    cur = cur.next
  nextpre.next = None
  popnext(nexthd)

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

""" map lambda str(x) convert to string for at each offset for comparsion """
def radixsort(arr):
  def countsort(arr, keyidx):
    out = [0]*len(arr)
    count = [0]*10
    strarr = map(lambda x: str(x), arr)
    karr = map(lambda x: x[len(x)-1-keyidx] if keyidx < len(x) else '0', strarr)
    for i in xrange(len(karr)):
      count[int(karr[i])] += 1
    for i in xrange(1,10):
      count[i] += count[i-1]
    # must go from end to start, as ranki-1.
    for i in xrange(len(karr)-1, -1, -1):
      ranki = count[int(karr[i])]  # use key to get rank, and set val with origin ary
      # ranki starts from 1, -1 to convert to index
      out[ranki-1] = int(strarr[i])
      count[int(karr[i])] -= 1
    return out
  for kidx in xrange(3):
    arr = countsort(arr, kidx)
  return arr
print radixsort([170, 45, 75, 90, 802, 24, 2, 66])

""" exp is 10^i where i is current digit offset, (v/exp)%10, the offset value """
def radixsort(arr):
  def getMax(arr):
    mx = arr[0]
    for i in xrange(1,len(arr)):
      mx = max(mx, arr[i])
    return mx
  #exp is 10^i where i is current digit number
  def countsort(arr, exp):
    count = [0]*10
    output =[0]*len(arr)
    for i in xrange(len(arr)):
      count[(arr[i]/exp)%10] += 1
    for i in xrange(1,10):
      count[i] += count[i-1]
    for i in xrange(len(arr)-1,-1,-1):
      d = (arr[i]/exp)%10
      rank = count[d]
      output[rank-1] = arr[i]
      count[d] -= 1
    return output
  mx = getMax(arr)
  exp = 1
  while mx/exp > 0:
    arr = countsort(arr, exp)
    exp *= 10
  return arr
print radixsort([170, 45, 75, 90, 802, 24, 2, 66])

""" min stack, when push x, push a pair (x, min) """
class MinStack:
  def __init__(self, minv=sys.maxint):
      self.stk = []
  def push(self, x):
      if x < self.minv:
          self.minv = x
      self.stk.append([x,self.minv])
  def push(self,x):
      if x < self.minv:
          self.stk.append(self.minv)
          self.minv = x
      self.stk.append(x)
  def pop(self):
      if self.stk[-1] == self.minv:
          cur = self.stk.pop()
          self.minv = self.stk.pop()
      else:
          cur = self.stk.pop()
      if len(self.stk) == 0:
          self.minv = sys.maxint
      return cur

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
    def __init__(self, mxSize=10):
        self.arr = []
        self.size = 0
        self.mxSize = mxSize
        self.key2idx = defaultdict()
    def get(self, idx):
        if idx >= 0 and idx < self.size:
            return self.arr[idx][0]
        return None
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
    def swap(self, src, dst):
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
        #  for each level top down, inside each level, while until stop on boundary.
        for level in reversed(xrange(len(self.head.next))):  # for each level down: park next node just before node >= ele.(insertion ponit).
            while levelhd.next[level] != None and levelhd.next[level].ele < ele:  # advance skipNode till to node whose next[level] >= ele, then down level
                levelhd = levelhd.next[level]  # store skip node(ele, next[...])
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


""" hash map entry is linked list. entry value is list of [v,ts] tuple.
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
        while l != r:
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

''' for each i, either belong to prev seq, or starts its own.
    s[i] = max(s[i-1]+A.i, A.i), either expand sum, or start new win from cur '''
# arr=[-1, 40, -14, 7, 6, 5, -4, -1]
def maxSumWrap(arr):
    maxsum = kadan(arr)
    for i in xrange(len(arr)):
        wrapsum += arr[i]
        arr[i] = -arr[i]
    maxWrap = wrapsum - kadan(arr)
    return max(maxsum, maxWrap)

def maxcircular(arr):
  si,isum,maxi,maxsum = 0,0,0,0
  rsi,irsum,maxrsi,maxrsum = 0,0,0,0
  totsum = 0
  # kadane
  for i in xrange(len(arr)):
    v = arr[i]
    isum += v
    maxsum = max(maxsum, isum)
    if maxsum == isum:
        maxi = si
    if isum < 0:
      isum = 0
      si = i+1
    # reverse sum
    rv = -v
    irsum += rv
    maxrsum = max(maxrsum, irsum)
    if maxrsum == irsum:
        maxrsi = rsi
    if irsum <= 0:
      irsum = 0
      rsi = i+1
    totsum += v
  return maxi, maxsum, maxrsi, maxrsum, totsum, totsum+maxrsum

print maxcircular([8, -9, 9, -4, 10, -11, 12])
print maxcircular([8, -8, 9, -9, 10, -11, 12])
print maxcircular([10, -3, -4, 7, 6, 5, -4, -1])


""" nlgn LIS, use parent to store each ele's parent """
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
  parent = []
  lis = []
  lis.append([0, arr[0]])
  maxlen = 0
  for i in xrange(1,len(arr)):
    if arr[i] > lis[-1][1]:
      parent[i] = lis[-1][0]
      lis.append([i,arr[i]])
    else:
      lo = bisectUpdate(lis, arr[i])
      lis[lo] = [i, arr[i]]
      parent[i] = lis[lo-1][0]
  i = lis[-1]
  while i:
    print arr[i]
    i = parent[i]
  return lis


""" no same char next to each other """
from collections import defaultdict
def noConsecutive(text):
  freq = defaultdict(int)
  for i in xrange(len(text)):
    freq[text[i]] += 1
  rfreq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
  segs = rfreq[0][1]   # most frequence div arr into k segments.
  sz = (1+len(text))/segs
  print rfreq, segs, sz
  if sz < 2:
    return "wrong"
  nextfree = [-1]*segs
  out = [-1]*len(text)
  for i in xrange(segs):
    out[i*sz] = rfreq[0][0]
    nextfree[i] = i*sz+1
  for k,v in rfreq[1:]:
    segidx = 0
    while nextfree[segidx] > sz:
        segidx += 1
    if segs - segidx < v:
      return "wrong"
    for cnt in xrange(v):
      out[nextfree[segidx]] = k
      nextfree[segidx] += 1
      segidx += 1
  return out

""" [0], [0,1], [1,2,2,3], ... for each"""
def countBits(n):
  out = [0]
  power = 0
  while len(out) < n:
    for i in xrange(2**power): # [0],[0,1],[0,1,2,3],[0..7]
      out.append(out[i]+1)
    power += 1
  return out

""" tab[n] = 1 + tab[n - closest_2_pow] """
def countBits(n):
  tab = [0]*n
  twosquare = 1
  for i in xrange(3,n):
    if i & i-1 == 0:
      twosquare = i
      tab[i] = 1
    else:
      tab[i] = tab[i-twosquare]
  return tab


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


""" sub string by concat all same length keys 
continue mov r when each r+k is in dict. skip exceeding dups.
mov l to r and reset trackings when break out when r+k not in dict.
"""
from collections import defaultdict
def concatKeys(txt, keys):
  keydict = defaultdict(int)  # key count
  out = []
  for k in keys:
    keydict[k] += 1
  k,cnt = len(keys[0]),0
  l,r = 0,0
  curmap = defaultdict(int)
  while l < len(txt):
    r = l
    while r < len(txt):
      wd = txt[r:r+k]
      if wd in keydict:
        curmap[wd] += 1
        if curmap[wd] == keydict[wd]:
          cnt += 1
          if cnt == len(keys):
            while curmap[arr[l:l+3]] > keydict[arr[l:l+3]]:
              l += 3
            out.append(l)
        elif curmap[wd] > keydict[wd]:
          l = keydict[wd]
          continue
        else:
          continue
      else:
        break
    # after break, which r+k not dict word, reset, mov left edge
    l = r+k
    cnt = 0
    curmap.clear()
  return out
print concatKeys("barfoothebarfoobarfooman", ["foo", "bar", "foo"])
print concatKeys("barfoothebarfoobarfooman", ["foo", "bar", "bar"])

""" use expected[] and found[] """
from collections import defaultdict
def concatKeys(s, arr):
  l,r,k = 0,0,len(arr[0])
  out = []
  expect = defaultdict(int)
  count = defaultdict(int)
  seen = 0
  for w in arr:
    expect[w] += 1
  # keep mov rite edge, inner adjust left edge when skip dups
  while r < len(s):
    wd = s[r:r+k]
    if not wd in expect:  # concat must be conti, no intervene.
      r += 1
      l = r
      seen = 0
      count.clear()
      continue
    # calculate rite edge first, as r starts from 0
    count[wd] += 1
    seen += 1
    # appear more caused by dups in left, slide left
    while count[wd] > expect[wd]:
      prewd = s[l:l+k]
      seen -= 1
      count[prewd] -= 1
      l += k    # slide left edge
    if seen == len(arr):
      out.append([l,r])
    # slide rite edge
    r += k
  return out


""" sliding win, outer while loop move rite, inner while loop mov left 
always enque r first, then check r-l+1==m"""
def maxWin(arr, m):
  l,r,mx = 0,0,0
  out,stk = [],[]
  for r in xrange(len(arr)):
    # clean stk off arr[r]
    while len(stk) and stk[-1][1] < arr[r]:
      stk.pop()
    stk.append([r,arr[r]])
    if r-l+1 == m:
      out.append(stk[0])
      if l == stk[0][0]:
        stk.pop(0)
      l += 1
  return out
print maxWin([5,3,4,6,9,7,2],2)
print maxWin([5,3,4,6,9,7,2],3)

""" sliding window, outer loop mov rite edge, inner loop mov left edge 
Always process rite edge first. You can check arr[i], or check win size.
"""
def maxWinSizeMzero(arr, m):
  l,r,zeros,mx=0,0,0,0
  for r in xrange(len(arr)):
    if arr[r] == 0:
      zeros += 1
      if zeros > m:  # squeeze to win=m
        while arr[l] != 0:
          l += 1
        # left edge shrink mov over zero
        l += 1
        zeros -= 1
    mx = max(mx, r-l)
  return mx
print maxWinSizeMzero([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],2)

''' when sliding, always inc r, and only inc win size when r edge is 0 '''
def maxWinSizeWithMzero(arr,m):
  l,r,win,maxWinsize=0,0,0,0
  # init/set point, test, then mov/loop
  while r < len(arr):
    if win <= m:  # mov r even when win size equals, to count end with 1
      # test current r edge first, then mov r edge.
      if arr[r] == 0:
        win += 1
      r += 1
    # Note why else: here is not correct !!!! b/c win+=1 in win<=m
    if win > m:
      if arr[l] == 0:  # always try to mov win size left edge.
        win -= 1
      l += 1
    maxWinsize = max(maxWinsize, r-l)  # l/r park at first item after 0.
  return maxWinsize
print maxWinSizeWithMzero([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],2)

""" 
loop each char, note down expected count and fount cnt. 
"""
from collections import defaultdict
def minwin(arr,t):
  expected = defaultdict(int)
  found = defaultdict(int)
  for c in t:
    expected[c] += 1
  l,cnt,mnw = 0,0,99
  for r in xrange(len(arr)):
    c = arr[r]
    if not c in t:
      continue
    found[c] += 1
    # no value when not expected. dup does not count
    if found[c] <= expected[c]:
      cnt += 1
    if cnt == len(t):
      while arr[l] not in t or found[arr[l]] > expected[arr[l]]:
        if found[arr[l]] > 0:  # squeeze left when it is dup.
          found[arr[l]] -= 1
        l += 1
      if r-l+1 < mnw:   # only update min when cnt.
        mnw = r-l+1
        print arr[l:r+1]
  return mnw
print minwin("azcaaxbb", "aab")
print minwin("ADOBECODEBANC", "ABC")

""" min window with 3 letter words """
def minwin(arr, words):
  found, expected = defaultdict(int), defaultdict(words)
  for i in xrange(len(arr)):
    if arr[i:i+4] not in words:
      continue
    w = arr[i:i+4]
    found[w] += 1
    if found[w] <= expected[w]:
      cnt += 1
    if cnt == len(words):
      while lw = arr[l:l+4] not in words:
        l += 1
      while found[lw] > expected[lw]:
        found[lw] -= 1
        l += 4
      mnw = min(mnw, r-l)        
  return mnw

""" Longest substring with at most 2 distinct chars 
sliding win, i -> win start, j -> the second char in win.
slide k, K++ if A[k] = A[k-1]
case 1: A[k] == A[j], still 2 char, but need to let j = k-1 so i will jmp to j+1.
case 2: A[k] != A[j], i,j are 2 chars, upbeat, and set i = j-1 and update j=k-1.
so i reset to the first char(j=k-1), as last time when (A[k] != A[k-1] and A[k] == A[j]), 
"""
def longestTwoChar(arr):
  i,j = 0, -1, mxlen = 0
  for k in xrange(1, len(arr)):
    if arr[k] == arr[k-1]:
      continue
    if j >= 0 and arr[k] != arr[j]:
      mxlen = max(mxlen, k-i)
      i = j + 1
    j = k -1
  return mxlen


"""
max distance between two items where right item > left items.
http://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
Lmin[0..n]   store min see so far. so min[i] is mono descrease.
Rmax[n-1..0] store max see so far from left <- right. so mono increase.
the observation here is, A[i] is shadow by A[i-1] A[i-1] < A[i], the same as A[j-1] by A[j].
Lmin and Rmax, scan both from left to right, like merge sort. upbeat max-distance.
"""
def max_distance(l):
    sz = len(l)
    Lmin = [l[0]]*sz
    Rmax = [l[sz-1]]*sz
    for i in xrange(1, sz-1:
        Lmin[i] = min(Lmin[i-1], l[i])  # lmin mono decrease
    for j in xrange(sz-2,0,-1):
        Rmax[j] = max(Rmax[j+1], l[j])  # rmax flat decrease
    i = j = 0   # both start from 0, Lmin/Rmax 
    maxdiff = -1
    while j < sz and i < sz:
      # found one, upbeat, Rmax also mono decrease, stretch
      if Lmin[i] < Rmax[j]:
          maxdiff = max(maxdiff, j-i)
          j += 1
      else: # Lmin is bigger, mov lmin as Lmin is mono decrease
          i += 1
    return maxdiff

""" Lmin[i] contains idx in the left arr[i] < cur, or -1
    Rmax[i] contains idx in the rite cur < arr[i]. or -1
    then loop both Lmin/Rmax, at i when Lmin[i] > Rmax[i], done
"""
def triplet(arr):
  mn,nxtmn = 99,99
  for i in xrange(len(arr)):
    v = arr[i]
    if v < mn:
      mn = v
    elif v < nxtmn:
      nxtmn = v
    else:
      return [mn,nxtmn,v]
print triplet([5,4,3,6,2,7])

""" tab[i,j]=LAP of arr[i:j] with k,i,j as first 3.
tab[i,j] = tab[k,i]+1, iff arr[k]+arr[j]=2*arr[i] """
def longestArithmathProgress(arr):
  mx = 0
  tab = [[0]*len(arr) for i in xrange(len(arr))]
  # boundary, tab[i][n] = 2, arr[i],arr[n] is an AP with len 2
  for i in xrange(len(arr)-2, -1, -1):
    tab[i][len(arr)-1] = 2
  for i in xrange(len(arr)-2, -1, -1): # start from n-2
    k,j = i-1,i+1
    while k >= 0 and j < len(arr):
      if 2*arr[i] == arr[k] + arr[j]:
        tab[k][i] = tab[i][j] + 1
        print arr[k],arr[i],arr[j], " :: ", k, i, j, tab[k]
        mx = max(mx, tab[k][i])
        k -= 1
        j += 1
      elif 2*arr[i] > arr[k] + arr[j]:
        j += 1
      else:
        k -= 1
    if k >= 0:  # if k is out, then [i:j] as header 2.
      tab[k][i] = 2
      k -= 1
  return mx
print longestArithmathProgress([1, 7, 10, 15, 27, 29])
print longestArithmathProgress([5, 10, 15, 20, 25, 30])

def LAP(arr):
  mx = 0
  tab = [[0]*len(arr) for i in xrange(len(arr))]
  for i in xrange(1,len(arr)):
    tab[0][i] = 2
  for i in xrange(1,len(arr)-1):
    k,j=i-1,i+1
    while k >= 0 and j <= len(arr)-1:
      if 2*arr[i] == arr[k]+arr[j]:
        tab[i][j] = tab[k][i] + 1
        mx = max(mx, tab[i][j])
        k -= 1
        j += 1
      elif 2*arr[i] > arr[k]+arr[j]:
        j += 1
      else:
        if tab[k][i] == 0:
          tab[k][i] = 2
        k -= 1
  return mx
print LAP([1, 7, 10, 15, 27, 29])
print LAP([5, 10, 15, 20, 25, 30])


""" 
left < cur < rite, for each i, max smaller on left and max rite. 
first, populate max[i] = arr[i]*rmax. Then from left, avl tree find
max left smaller, max = max(max[i]*lmax)
lmax smaller ary is the same as rmax smaller, avl or merge sort
"""
def triplet_maxprod(arr):
  maxl,maxtriplet = -1,1
  rmax=[0]*len(arr)
  mx = 1
  for i in xrange(len(arr)-1,-1,-1):
    if arr[i] < mx:
      rmax[i] = mx*arr[i]
    mx = max(mx, arr[i])

  lmaxsmaller = AVL(arr[0])
  for i in xrange(1, len(arr)):
    maxl = lmaxsmaller.maxSmaller(arr[i])
    maxtriplet = max(maxtriplet, maxl*rmax[i])
    lmaxsmaller.insert(arr[i])
  return maxtriplet
print triplet_maxprod([7, 6, 8, 1, 2, 3, 9, 10])
print triplet_maxprod([5,4,3,10,2,7,8])

''' max prod exclude a[i], keep track of pre prod excl cur '''
def productExcl(arr):
  prod = [1]*len(arr)
  tot = 1
  for i in xrange(len(arr)):
    prod[i] = tot
    tot *= arr[i]
  tot = 1
  for i in xrange(len(arr)-1,-1,-1):
    prod[i] *= tot
    tot *= arr[i]
  return prod

""" max sub arr product with pos and neg ints.
for each ele, track cur max/min, and upbeat tot max.
"""
def maxProd(arr):
  mx,mn=1,1
  for i in xrange(len(arr)):
    v = arr[i]
    cmx = mx
    mx = max(mx, v*mx, v*mn, v)
    mn = min(mn, v*cmx, v*mn, v)
  return mx, mn
print prod([-6,-3])
print prod([-6,3,-4,-5,2])


''' at every point, keep track both max[i] and min[i].
mx[i] = max(mx[i-1]*v, mn[i-1]*v, v),
at each idx, either it starts a new seq, or it join prev sum
'''
def maxProduct(a):
  maxProd = cur_max = cur_min = a[0]  # first el as cur_max/min
  cur_beg, cur_end, max_beg, max_end = 0,0,0,0
  for cur in a[1:]:
      new_min = cur_min*cur
      new_max = cur_max*cur
      # update max/min of cur after applying cur el
      cur_min = min([cur, new_max, new_min])
      cur_max = max([cur, new_max, new_min])
      maxProd = max([maxProd, cur_max])
      if maxProd is cur_max:
          cur_end, max_end = i,i
          #upbeat(max_beg, max_end)
  return maxProd

''' track prepre, pre, cur idx of zeros '''
def maxones(arr):
  mx,mxpos = 0,0
  prepre,pre,cur = -1,-1,-1
  for i in xrange(len(arr)):
    if arr[i] == 1:
      continue
    cur = i
    if cur-prepre-1 > mx:
        mx = cur-prepre-1
        mxpos = pre
    prepre,pre = pre,cur
  if i-prepre > mx:
    mxpos = pre
  return mxpos
print maxones([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
print maxones([1, 1, 1, 1, 0])

# 1-d array, max length, like slideing wind, track l/r edge.
def maxWinM(arr,m):
  l,r,win,maxwinsize = 0,0,0,0
  while r < len(arr):
    if win <= m:
      if arr[r] == 0:
          win += 1
      r += 1  # park r to next window rite edge
    else:
      if arr[l] == 0:
          win -= 1
      l += 1  # park l to next window left edge
    maxwinsize = max(maxwinsize, r-l)  # dist is half include.
  return maxwinsize
print maxWinM([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1], 2)


"""keep track left min, max profit is when bot at min day sell today.
keep track rite max, max profit is when bot today and sell at max day.
"""
def stock_profit(arr):
  sz = len(l)
  maxprofit = 0
  lmin = arr[0]
  for i in xrange(1, sz-1):
    maxprofit = max(maxprofit, arr[i]-lmin)
    lmin = min(lmin, arr[i])
  return maxprofit
    
  rmax = arr[sz-1]
  for i in xrange(sz-2,-1,-1):
    maxprofit = max(maxprofit, rmax-arr[i])
    rmax = min(rmax, arr[i])
  return maxprofit

""" allow 2 transactions, so maxp = max{one transaction on 0..i, one trans on i+1..n}
left trans end at i, max(l[i-1], S[i]-valley)
rite trans start at i, max(r[i+1], peak-S[i])
"""
def stock_profit(L):
    sz = len(L)
    lp,up=[0]*sz,[0]*sz
    valley,peak=sys.maxint,0
    for i in xrange(sz):
        valley = min(valley, L[i])
        lp[i] = max(lp[i-1], L[i]-valley)  # max of lp[i-1],
        # price peak going up from n..0
        j = sz-i-1
        peak = max(peak, L[j])
        up[j] = max(up[j+1], peak-L[j])
    for i in xrange(sz):
        tot = max(tot, lp[i] + up[i])
    return tot
print stock_profit([10, 22, 5, 75, 65, 80])

""" 2 trans. left <- rite, buy at i sell at rmax. left -> rite, buy at valley sell at i. """
def profit(arr):
  sz = len(arr)
  profit=[0]*sz
  rmax = arr[-1]
  # profit[i] carry max profit from [i..n]
  for i in xrange(sz-2,0,-1):
    rmax = max(rmax, arr[i])
    profit[i] = max(profit[i+1], rmax-arr[i])
  lmin = arr[0]
  # max profit of [0..i] = max(maxprofit of [0..i-1], sell_at_i + maxprofit of [i..n])
  for i in xrange(1,len(arr)):
    lmin = min(lmin, arr[i])
    profit[i] = max(profit[i-1], profit[i]+arr[i]-lmin)
  return profit[sz-1]
print 

""" many times. use bot flag. if not bot and i+1 > i, then buy at i. 
if bot, and i+1<i, then sell. sum up every incr segment.(i,j)
"""
def profit_many_trans(arr):
  buy,sz,bot = 0,0,len(arr),False
  profit = 0
  for i in xrange(sz-1):
    # look ahead. buy today only when tomo high(we can sell)
    if not bot and arr[i] < arr[i+1]:
      buy = i
      bot = True
      continue
    if bot and arr[i] > arr[i+1]:  # sell when look ahead down.
      profit += arr[i]-arr[buy]
      bot = False
  if bot and arr[sz-1] > arr[buy]:
    profit += arr[sz-1]-arr[buy]
  return profit
print profit_many_trans([2,4,3,5,4,2,8,10])

""" FSM bot[i]/sold[i]. arr[i], update bot/sold, stash prebot before updat bot.
bot[i]=max(bot[i-1],sold[i-1]-arr[i]),  ; calc bot first.
sold[i]=max(sold[i-1], bot[i-1]+arr[i]) 
"""
def profit_many_trans(arr):
  bot,sold = -sys.maxint, 0
  for i in xrange(len(arr)):
    prebot = bot
    bot = max(bot, sold-arr[i])
    sold = max(prebot + arr[i], sold)
  return sold
print profit_many_trans([2,4,3,5,4,2,8,10])

""" FSM, 3 states, bot/sold/rest. Each state profit at arr[i], so bot[i]/sold[i]/rest[i].
bot[i] = max(bot[i-1], rest[i-1]-arr[i]); not buy at i, or buy with rest money.
sold[i] = max(sold[i-1], bot[i-1]+arr[i]), not sell, or sell
rest[i] = sold[i-1], because cooldown. it is presold
as bot[i] only relies on bot[i-1], can use one var.
"""
import sys
def stock_profit_cooldown(arr):
  bot,sold,rest = -sys.maxint, 0, 0
  for i in xrange(len(arr)):
    presold = sold
    sold = max(sold, bot+arr[i])
    bot = max(bot, rest-arr[i])
    rest = presold
  return max(sold, rest)
print stock_profit_cooldown([1, 2, 3, 0, 2])


""" Atmost up-to K transactions. max profit may involve less than k trans.
2 state, bot[i]/sell[i]. At each arr[i], try all trans 1..k.
bot[i] only relies on bot[i-1], so bot[i][k] can be simplied to bot[k]
For each price, bottom up 1 -> k transations. use k+1 as k deps on k-1. k=0.
sell[i][k] is max profit with k trans at day i.
sell[i] = max(sell[i], bot[i]+p), // carry over prev num in trans i'th profit.
bot[i] = max(bot[i], sell[i-1]-p) // max of incl/excl bot at this price, 
"""
import sys
def topk_profit(arr, k):
  sell,bot=[0]*(k+1), [-9999]*(k+1)
  for i in xrange(len(arr)):
    for trans in xrange(1,k+1):  # bottom up trans at each day.
      sell[trans] = max(sell[trans], bot[trans]+arr[i])
      bot[trans] = max(bot[trans], sell[trans-1]-arr[i])
    print arr[i], bot, "  ", sell
  return sell[k]
print topk_profit([1, 4, 5, 6, 7, 8, 9, 10], 2)
print topk_profit([1, 4, 5, 6, 7, 8, 4, 10], 2)
#print topk_profit([10, 22, 5, 75, 65, 80], 2)

""" with exact k trans, tab[i,j,k] = max profit of arr[i..j] with k trans.
tab[i,j,k] = max(tab[i+x, j, k-1]+(arr[i+x]-arr[i]))
the same as dag with exactly k edges
"""
def profit_topk(arr):
  """ find successive transaction pairs, enum all possible trans,
  and pick the top k trans """
  def valleyPeakPair(arr, st):
    v = st
    while v+1 < len(arr) and arr[v] > arr[v+1]:
      v += 1
    p = v+1
    while p+1 < len(arr) and arr[p] < arr[p+1]:
      p += 1
    if v == p:
      return False, []
    return True, [v,p]

  pairs,profit = [],[]
  v = 0
  found = True
  while found:
    found, [v,p] = valleyPeakPair(arr, v)
    # when prev trans's valley bigger than this valley, no consolidate
    while len(pairs) > 0 and arr[pairs[-1][0]] > arr[v]:
      profit.append(arr[pairs[-1][1]] - arr[pairs[-1][0]])
      pairs.pop()
    # when pre trans valley smaller than this valley, can combine trans.
    while len(pairs) > 0 and arr[pairs[-1][0] < arr[v]]:
      profit.append(arr[pairs[-1][1]] - arr[v])
      v = pairs[-1][0]
      pairs.pop()
    pairs.append([v,p])
  while len(pairs) > 0:
    profit.append(arr[pair[-1][1]] - arr[pairs[-1][0]])
    pairs.pop()

  # calculate the top k profits
  if len(profits) > k:
    profits = sorted(profilts)[n-k:]
  return reduce(lambda t,c: t += c, profits)


""" first missing pos int. bucket sort. L[0]=1, L[1]=2. value from 1..n
    foreach pos, while L[pos]!=pos+1: swap(L[pos],L[L[pos]-1]).
"""
def firstMissing(L):
  def bucketsort(L):
    for i in xrange(len(L)):
      while L[i] > 0 and L[i] != i+1:
        v = L[i]-1
        if L[v] < 0:  # arr[i] is a DUP as already detected and toggled
          L[i] = 0    # Li is a dup, can not swap anymore, set it to 0, skip
          L[v] -= 1
        else:         # L[v] >= 0: when dup, entry was set to 0, we still can swap to it.
          L[i],L[v] = L[v],L[i]
          L[v] = -1
      # natureally in correct pos, turn to -1
      if L[i] == i+1:
        L[i] = -1
  bucketsort(L)
  for i in xrange(len(L)):
    if L[i] == 0:  # ith has i+1 will be -1, missing replaced by dup, so 0.
      break
  return i+1
print firstMissing([4,2,5,2,1])

""" For each item, if it in rite place, toggle arr[i] -= 1 for count,
if it's dup and its rite place already taken, -= 1, otherwise, swap """
def bucketsort_count(arr):
  def swap(arr, i,j):
    arr[i],arr[j] = arr[j],arr[i]
  for i in xrange(len(arr)):
    while arr[i] != i+1 and arr[i] >= 0 and arr[i] != 999:  # skip DUP
      v = arr[i]
      if arr[v] >= 0:    # swap and toggle!
        swap(arr, i, v)
        arr[v] = -1
      else:             # inc and sentinel
        arr[v] -= 1
        arr[i] = 999    # set cur i to SENTI to indicate it is dup
    if arr[i] == i+1:   # toggle
      arr[i] = -1
  return arr
print bucket([1, 2, 4, 3, 2])

""" bucket tab[arr[i]] for i,j diff by t, and idx i,j with k """
def containsNearbyAlmostDuplicate(self, nums, k, t):
  if t < 0: return False
  n = len(nums)
  d = {}
  w = t + 1
  for i in xrange(n):
      m = nums[i] / w
      if m in d:
          return True
      if m - 1 in d and abs(nums[i] - d[m - 1]) < w:
          return True
      if m + 1 in d and abs(nums[i] - d[m + 1]) < w:
          return True
      d[m] = nums[i]
      if i >= k: del d[nums[i - k] / w]
  return False

""" find dup in ary using slow/fast pointers 
arr[a] is the same as a.next, after cycle meet, start from 0.
"""
def findDuplicate(arr):
  slow = fast = finder = 0
  while True:
    slow = arr[slow]
    fast = arr[arr[fast]]
    if slow == fast:
      while finder != slow:
        finder = arr[finder]
        slow = arr[slow]
      return finder


""" max repetition num, bucket sort. change val to -1, dec on every reptition.
    bucket sort, when arr[i] is neg, means it has correct pos. when hit again, inc neg.
    when dup processed, set it to max num to avoid process it again.
"""
# maxrepeat([1, 2, 2, 2, 0, 2, 0, 2, 3, 8, 0, 9, 2, 3])
def maxrepeat(arr):
  def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
  for i in xrange(len(arr)):
    while arr[i] >= 0 and arr[i] != i:
      v = arr[i]
      if v == sys.maxint:  # skip repeat node.
        break
      if arr[v] > 0:  # first time swapped to.
        swap(arr, i, v)
        arr[v] = -1
      else:           # target already swapped, it is repeat. incr.
        arr[v] -= 1
        arr[i] = sys.maxint  # i is repeat node. mark it.
    if arr[i] == i:
      arr[i] = -1
  return arr

def repmiss(arr):
  def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
  for i in xrange(len(arr)):
    while arr[i] > 0 and arr[i] != i+1:
      v = arr[i]-1
      if arr[v] > 0:
        swap(arr, i, v)
        arr[v] = -arr[v]  # flip to mark as present.
      else:         # arr[v-1] < 0:
        print "dup at ", i, arr
        break
  return arr

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

"""
Every row in the matrix can be viewed as the ground. height is the count
of consecutive 1s from the ground. Iterate each ground.
htMatrix[row] = [2,4,3,0,...] the ht of each col if row is the ground
"""
def maxRectangle(matrix):
  # iterate each row, build matrix store height for each col. 
  rows, cols = len(matrix), len(matrix[0])
  htMatrix[0][c] = 1 if matrix[0][c] == 1 else 0 for c in xrange(cols)
  for r in xrange(1, rows):
    for c in xrange(cols):
      if matrix[r][c] == 1:
        htMatrix[r][c] = htMatrix[r-1][c] + 1
  for rowHt in htMatrix:
    mx = max(mx, maxHistogram(rowHt))
  return mx

  ''' ht of each col, maintain an incr stk. when cliff, ht[col] < stk.top, 
  stk.top rect ends. cals max, and keep pop stk '''
  def maxHistogram(rowHt):
    stk, mx = [], 0
    for col in xrange(len(rowHt)):
      while rowHt[stk.top()] > rowHt[col]:
        mx = max(mx, rowHt[stk.top()]*(i-stk[-1]+1))  # left cliff * rite cliff
        stk.pop()
      stk.append(i)
    return mx

""" area[r,c] = (rite_edge[r,c]-left_edge[r,c])*ht. use max left when substract left edge.
left_edge[r,c] = max(left_edge[r-1,c], cur_left_edge), max of prev row's left edge.
rite_edge[r,c] = min(rite_edge[r-1,c], cur_rt_edge), min of prev row's rite edge
as we use max left edge, cur_left_edge = c+1, we use min rite edge, cur_rite_edge = c
"""
def maxRectangle(matrix):
  for r in xrange(len(matrix)):
    ht,left_edge,rite_edge = [0]*n, [0]*n, [n]*n
    cur_left_edge = 0, cur_rite_edge = n
    for c in xrange(len(matrix[0])):
      if matrix[r][c] == 1:
        ht[c] += 1
      else:
        htc[c] = 0
      if matrix[r][c] == 1:
        left_edge[c] = max(left_edge[c], cur_left_edge) # max left when substract
      else:
        left_edge[c] = 0      # reset when matrix 0
        cur_left_edge = c+1   # mov left edge -->
      if matrix[r][c] == 1:
        rite_edge[c] = min(rite_edge[c], cur_rite_edge) # min rite to narrow
      else:
        rite_edge[c] = n    # defaut is n as we using min rite
        cur_rite_edge = c   # shrink rite edge
    for c in xrange(n):
      maxA = max(maxA, (rite_edge[c]-left_edge[c])*ht[c])
  return maxA


""" largest Rect with swap of any pairs of cols 
first, build height col for each row. Then sort it.
"""
def largestRectWithSwapCol(mat):
  pass

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
Dynamic Programming, better than DFS/BFS.
bottom up each row, inside each row, iter each col.
[ [col1, col2, ...], [col1, col2, ...], row3, row4, ...]
At each row/col, recur incl/excl situation.
Boundary condition: check when i/j=0 can not i/j-1.
""" """ """ """ """
def minjp(arr):
  mjp,rite,nxtrite=1,arr[0],arr[0]
  for i in xrange(1,n):
    if i < rite:
      nxtrite = max(nxtrite, i+arr[i])
    else:
      mjp += 1
      rite = nxtrite
    if rite >= n:
      return mjp

# tab[i] : min cost from start->i
import sys
def minjp(arr):
  tab = [sys.maxint]*len(arr)
  tab[0] = 0
  for i in xrange(1,len(arr)):  # bottom up, expand to top. 
    for j in xrange(i):
      if arr[j]+j >= i:  # can reach i
        tab[i] = min(tab[i], tab[j] + 1)
  return tab[len(arr)-1]
assert minjp([1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]) == 3

''' For DAG, enum each intermediate node, otherwise, topsort, or tab[i][j][step]'''
def minpath(G,src,dst,cost,tab):
  mincost = cost[src][dst]
  # for each intermediate node
  for v in G.vertices():
    if v != src and v != dst:
      mincost = min(mincost,
          minpath(G,src,v) +
          minpath(G,v,dst))
      tab[src][dst] = mincost
  return tab[src][dst]
''' For DAG, enum each intermediate node, otherwise, topsort, or tab[i][j][step]'''
def minpath(G, src, dst):
  tab = [sys.maxint]*szie
  tab[0] = 0
  for i in xrange(n):
    for j in xrange(n):
      if i != j:
        tab[i] = min(tab[j]+cost[i][j], tab[i])
  return tab[n-1]

''' mobile pad k edge, tab[src][dst][k] '''
def minpath(G, src, dst, k):
  for gap in xrange(k):
    for s in G.vertices():
      for d in G.vertices():
        tab[s][d][gap] = 0
        if gap == 1 and G[s][d] != sys.maxint:
          tab[s][d][gap] = G[s][d]
        elif s == d and gap == 0:
          tab[s][d][gap] = 0
        if s != d and gap > 1:
          for k in src.neighbor():
            tab[s][d][gap] = min(tab[k][d][gap-1]+G[src][k])
  return tab[src][dst][k]

''' topology sort, tab[i] = min cost to reach dst from i'''
def minpath(G, src, dst):
  def topsort(G,src,stk):
    def dfs(s,G,stk):
      for v in neighbor(G,s):
        if not visited[v]:
          dfs(v,G,stk)
      stk.append(v)
    for v in G:
      dfs(v,G,stk)
    return stk
  toplist = topsort(G,src,stk)
  for i in g.vertices():
    tab[i] = sys.maxint
  # after topsort, leave at the end of the list.
  tab[d] = 0
  for nb in d.neighbor():
      tab[nb] = cost[nb][d]
  for i in xrange(len(toplist)-1,-1,-1):
      for j in i.neighbor():
          tab[j] = min(tab[i]+cost[i][j], tab[j])
  return tab[src]


''' start from minimal, only item 0, i loop items, w loops weights at each item. 
v[i, w] is max value for each weight with items from 0 expand to item i, v[i,w] based on f(v[i-1, w]).
v[i, w] = max(v[i-1, w], v[i-1, w-W[i]]+V[i]) 
'''
def knapsack(arr, varr, W):
  prerow, row = [0]*(W+1), [0]*(W+1)
  for i in xrange(len(arr)):
    row = prerow[:]
    w,v = arr[i], varr[i]
    for ww in xrange(w, W+1):
      if i == 0:     # init boundary, first 
        row[w] = v
      else:
        prev = prerow[ww-w]
        if prev+v > row[ww]:
          row[ww] = max(row[ww], prev+v)
    prerow = row[:]
  return row[W]
print knapsack([2, 3, 5, 7], [1, 5, 2, 9], 18)

def backpack(arr, W):
  pre,row = [0]*(W+1), [0]*(W+1)
  for i in xrange(len(arr)):
    row = pre[:]
    for w in xrange(arr[i], W+1):      
      row[w] = max(row[w], pre[w-arr[i]]+arr[i])
    pre = row[:]
  return row[W]
print backpack([2, 3, 5, 7], 11)


""" when outer loop bottom up value, at each value, each coin tried, so coin order matters, permutation. 
When outer bottom up coins, tab[s, c-1] excl coin c for sum s. its like incl/excl coin.
"""
# tab[3,2] = [12],[21] t(3,2) = [111]+[[2,(1,2)]=[2][1]]+[[1,2]]
def coinchangeSet(coins, V):
  tab = [[0]*len(coins) for i in xrange(V+1)]
  tab2 = [[0]*len(coins) for i in xrange(V+1)]
  for c in xrange(len(coins)):
    tab[0][c] = 1  # when v = coin_value, init pre subproblem tab[0][c] to 1
    tab2[coins[c]][c] = 1  # or init real tab[coins[c]][c] to 1, the accumulate later.
  # outer loops bootom up each value, bottom up each coin at each vaue.
  for v in xrange(1,V+1):
    for c in xrange(len(coins)):
      excl = 0
      if c >= 1:
        excl = tab[v][c-1]
      incl = 0
      if v >= coins[c]:   # when v = coin value, tab[0][c]
        incl = tab[v-coins[c]][c]
      tab[v][c] = excl + incl
      tab2[v][c] += excl + incl   # accumulate on top, as tab2[0][c]=0 here.
  return tab[V][len(coins)-1]
print coinchangeSet([1, 2, 3], 4)

''' outer loop thru all value, and inner thru all coin,
diff order counts, [1,1,1], [1,2], [2,1]
'''
def coinchangePerm(coins, V):
  tab = [0]*(V+1)
  tab[0] = 1   # when coin face value = value, match.
  for v in xrange(1,V+1):
    for c in xrange(len(coins)):
      if v >= coins[c]:
        tab[v] += tab[v-coins[c]]  # XX f(i)=f(i-1)+1
  return tab[V]
print coinchangePerm([1, 2], 3)

""" tot ways for a change. permutation. [2,3] [3,2].
bottom up each val, a new row [c1,c2,...], bottom up coin col first, 
so at coin 2, no knowledge of 3, so 5=[2,3], coin is sorted and bottom up. no [3,2] ever.

if for perm, [2,3] [3,2], bottom up val first, at each value, bottom up coin.

when iter coin 3, tab[3,6,9..]=1; for coin 5, tab[5,10]=1
for coin 10, tab[10]=tab[10]+tab[0], which is #{[5,5],[10]}
hence, do not init tab[c]=1, and not tab[v]=tab[v-c]+1
"""
def coinchange(coins, V):
  tab = [0]*(V+1)
  # can not init all coins. v=8, at 3, count(5,3), at 5, double count(3,5)
  # for v in xrange(1,V+1):
  #     tab[v] = 1
  tab[0] = 1
  # outer loop bottom up coin, tab[s,i-1] mean excl coin i at s.
  for i in xrange(len(coins)):
    fv = coins[i]
    for v in xrange(fv,V+1):
      # sum up k-ways to reach this value.
      tab[v] += tab[v-fv]
      # to get min ways, tab[0]=0, tab[v]=999
      # tab[v] = min(tab[v], 1+tab[v-fv])
  return tab[V]
print coinchange([2,3,5], 10)

# comb can exam head, branch at header incl and excl
def combination(arr, path, sz, pos, result):
  if sz == 0:
    result.append(path)  # add to result only when sz
    return result
  if pos >= len(arr):
    result.append(path)
    return result
  l = path[:]  # deep copy path before recursion
  l.append(arr[pos])
  combination(arr, l,    sz-1, pos+1, result)
  combination(arr, path, sz, pos+1, result)
  return

def comb(arr,r):
  res = []
  if r == 1:  # leaf, need ret a list, wrap leaf as first element.
    for e in arr:
      res.append([e])
    return res
  # incl/excl each hd
  for i in xrange(len(arr)-r+1):
    hd = arr[i]
    narr = arr[i+1:]
    for e in comb(narr, r-1):
      e.append(hd)
      res.append(e)
  return res
print comb("abc", 2)


# Generate all possible sorted arrays from alternate eles of two sorted arrays
def alterCombRecur(a,b,ai,bi,froma,path,out):
  if froma:
    v = a[ai]
    found,bs = bisect(b, v)
    path.append(v)
    for j in xrange(bs,len(b)):
      l = path[:]  # deep cpoy path before each recursion
      alterCombRecur(a,b,ai,j,False,l,out)
  else:
    path.append(b[bi])
    out.append(path)
    found,sa = bisect(a,b[bi])
    for i in xrange(sa, len(a)):
      l = path[:] # deep copy path before each recursion
      alterCombRecur(a,b,i,bi,True,l,out)
def alterComb(a,b):
  out = []
  for ai in xrange(len(a)):  # a new recursion seq
    av = a[ai]
    found,bi = bisect(b, av)
    path = []  # empty new path for each recursion
    if bi < len(b):
      alterCombRecur(a,b,ai,bi,True,path,out)
  return out
print alterComb([10, 15, 25], [1,5,20,30])


""" recur with partial path, for loop each, as incl/excl """
def comb(arr, pos, r, path, res):
  if r == 0:  # stop recursion when r = 1
    cp = path[:]
    res.append(cp)
    return res
  for i in xrange(pos, len(arr)-r+1):  # can be [pos..n]
    path.append(arr[i])
    comb(arr, i+1, r-1, path, res)  # iter from idx+1, as not 
    path.pop()   # deep copy path or pop
  return res
res=[];comb([1,2,3,4],0,2,[],res);print res;


""" incl offset into path, recur. skip ith of this path, next 
recur fn params must be subproblem with offset and partial result.
# [a, ..] [b, ...], [c, ...]
# [a [ab [abc]] [ac]] , [b [bc]] , [c]
"""
def powerset(arr, offset, path, result):
  # subproblem with partial result, incl all interim results.
  result.append(path)
  # within cur subproblem, for loop to incl/excl for next level.
  for i in xrange(offset, len(arr)):
    # compare i to i-1 !!! not i to offset, as list is sorted.
    if i > offset and arr[i] == arr[i-1]:
        continue
    l = path[:]
    l.append(arr[i])
    powerset(arr, i+1, l, result)
path=[];result=[];powerset([1,2,2], 0, path, result);print result;


"""when recur to pos, incl pos with passed in path, or skip pos.
for dup, sort, skip current i is it is a dup of offset.
recur fn params must be subproblem with offset and partial result.
"""
def permDup(arr, offset, path, res):
  def swap(arr, i, j):
    arr[i],arr[j] = arr[j],arr[i]
  if offset == len(arr):
    return res.append(path[:])
  # within subproblem, iterate to incl/excl as head to next level.
  for i in xrange(offset, len(arr)):
    # if i follow offset and is dup of *offset*, skip it. not compare to i-1
    if i > offset and arr[offset] == arr[i]:
      continue
    swap(arr, offset, i)
    path.append(arr[offset])
    # when recur, both arr, path and offset changed as subproblem
    permDup(arr, offset+1, path, res)
    path.pop()
    swap(arr, offset, i)
path=[];res=[];permDup(list("123"), 0, [], res);print res
path=[];res=[];permDup(list("112"), 0, [], res);print res


# need to pass different arr upon each recursion, vs. different offset.
def perm(arr):
  result = []
  if len(arr) == 1:
    result.append([arr[0]])
    return result
  for i in xrange(len(arr)):
    hd = arr[i]
    l = arr[:]i
    del l[i]   # change arr for next iteration
    for e in perm(l):
      e.insert(0,hd)
      result.append(e)
  return result


""" from L <- R, find first a[i] < a[i+1] = pivot.
then swap with first rite larger, and reverse right part.
"""
def nextPermutation(txt):
  for i in xrange(len(txt)-2, -1, -1):
    if txt[i] < txt[i+1]:
      break
  if i < 0:
    return ''.join(reversed(txt))
  pivot = txt[i]
  for j in xrange(len(txt)-1, i, -1):
    if txt[j] > pivot:
      break
  txt[i],txt[j] = txt[j],txt[i]
  l,r = i+1, len(txt)-1
  while l < r:
    txt[l],txt[r] = txt[r],txt[l]
    l += 1
    r -= 1
  return txt

""" iter each head, rank += index(hd,a[i:])*(n-i-1)! 
the index calcu is based off smaller substr by removing current head
"""
from math import factorial
def permRank(arr):
  def index(c, arr):
    return sorted(arr).index(c)
  sz,rank = len(arr),1
  for i in xrange(sz):
    hd = arr[i]
    rank += index(hd, arr[i:]) * factorial(sz-i-1)
  return rank
print permRank("bacefd")

""" find the index of the char of each next subsize, 
rank/factorial(n) is the index of the char in the sorted arr """
from math import factorial
def nthPerm(arr, n):
  out = []
  sz = len(arr)
  cnt = factorial(sz)
  for i in xrange(sz):
    cnt = cnt/(sz-i)   # cnt = (n-1)! = n!/n
    idx = n/cnt
    n -= idx*cnt
    out.append(arr[idx])
    arr = arr[:idx] + arr[idx+1:]   # del arr[idx] after pick
  return "".join(out)
assert(nthPerm("1234", 15), "3241")

""" iter each pos i from 0 <- n. for width w, w! perms.
At each i, there are (n-i)! perms. If there are k eles smaller than cur,
means cur rank += those k packs that is smaller than cur. so rank += k*(n-i)!
with dup, tot is (n-i)!/A!*B!, and we need sum up all k copies of dup ele to rank
"""
from collections import defaultdict
from math import factorial
def permRankDup(arr):
  ''' tot and divide dup! in the counter '''
  def permTotal(n, counter):
    tot = factorial(n)
    for k,v in counter.items():
      tot /= factorial(v)
    return tot
  rank = 1
  counter = defaultdict(int)
  for i in range(len(arr)-1,-1,-1):
    e = arr[i]
    counter[e] += 1
    # tot perms i..n. arr[i]'s rank
    tot = permTotal(len(arr)-i-1, counter)
    for k,v in counter.items():
      if e > k:  # arr[i] > k at rite, sum up v dups of k.
        rank += tot*v
  return rank
assert permRankDup('3241') == 16
print permRankDup('baa')
print permRankDup('BOOKKEEPER') == 10743
assert permRankDup("BCBAC") == 15


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

""" partition s to restore ip, bfs and dfs recursion """
from collections import deque
def restoreIp(s):
  res, q = [], deque()
  q.append([0, 0, []])
  while len(q) > 0:
    pos, segs, ips = q.popleft()
    if pos == len(s):
      res.append(ips)
      continue
    ed = min(len(s), pos+2)
    for i in xrange(pos, ed+1):
      if int(s[pos:i+1]) <= 255 and segs < 4:
        p = ips[:]
        p.append(s[pos:i+1])
        q.append([i+1, segs+1, p])
  return res
print restoreIp("25525511135")

def restoreIp(s, pos, path, res):
  if pos == len(s):
    res.append(path)
    return
  ed = min(len(s), pos+2)
  for i in xrange(pos+1, ed+1):
    if int(s[pos:i+1]) <= 255 and len(path) < 4:
      p = path[:]
      p.append(s[pos:i+1])
      restoreIp(s, i+1, p, res)
  return res
s="25525511135";path=[];res=[];restoreIp(s,0,path,res);print res;

""" valid parenthese, dfs recursion, when reaching pos, carry path"""
def genParenth(n, l, r, path, res):
  if l == n:
    p = path[:]
    for i in xrange(r,n):
      p.append(")")
    res.append("".join(p))
    return res
  path.append("(")
  genParenth(n, l+1,r, path, res)
  path.pop()
  if l > r:
    path.append(")")
    genParenth(n,l,r+1, path, res)
    path.pop()
  return res
path=[];res=[];genParenth(3,0,0,path,res);print res;

from collections import deque
def genParath(n):
  out = []
  q = deque()
  q.append(["(", 1, 0])
  while len(q) > 0:
    [path,l,r] = q.popleft()
    if l == n:
      for i in xrange(2*n-l-r):
        path += ")"
      out.append(path)
    else:
      if l == r:
        p = str(path) + "("
        q.append([p, l+1, r])
      else:  # at this pos, we can append eith l/r, branch.
        p = str(path) + "("
        q.append([p,l+1,r])
        p = str(path) + ")"
        q.append([p,l,r+1])
  return out
print genParath(3)


""" first, find unbalance(l-r), recur each pos, dfs excl/incl each pos, no dp short """
def rmParenth(s, res):
  path = []
  L,R = 0,0
  for i in xrange(s):   # calculate balance(l-r)
    if s[i] == "(":
      L += 1
    else:
      if L != 0:
        L -= 1
      else:
        R += 1
  
  recur(res, s, 0, L, R, 0, path)
  return res
  # recur each pos, L,R is balance of l-r, either L=0 or R=0
  def recur(res, s, pos, L, R, openL, path):
    if pos == len(s) and L == 0 and R == 0 and openL == 0:
      p = path[:]
      res.append(p)
      return
    if pos == len(s) or L < 0 or R < 0 or openL < 0:
      return

    if s[pos] == "(":
      recur(res, s, pos+1, L,   R, openL,  path)  # skip cur L
      recur(res, s, pos+1, L-1, R, openL+1,path.append("(")) # use cur, L-1
      path.pop()
    else if s[pos] == ")":
      recur(res, s, pos+1, L, R-1, openL,   path) # skip cur R, R-1
      recur(res, s, pos+1, L, R,   openL-1, path.append(")"))  # use cur R,
      path.pop()
    else:
      recur(res, s, pos+1, L, R, openL, path.append(s[pos]))
    return

s="()())()";path=[];res=[];print rmParenth(s,0,path,0,0,res)
s="()())()";path=[];res=[];print rmParenth(s,0,path,0,0,res)


def rmParenth(arr, pos, rmpos, res):
  lcnt = 0
  for i in xrange(pos, len(s)):
    if arr[i] == "(":
      lcnt += 1
    else:
      lcnt -= 1
    if lcnt < 0:
      for j in xrange(rmpos, i+1):
        if arr[j] == ")" and (j == rmpos or arr[j-1] != ")"):
          rmParenth(arr[:j]+arr[j+1:], i, j, res)
      return
  res.append(arr)

""" anything we can do with recursion, we should do with bfs """
from collections import deque
def bfsInvalidParen(s):
  res = []
  q = deque()
  q.append([1, ["("], 1, 0])
  while len(q):
    pos,path,l,r = q.popleft()
    if pos == len(s):
      if l == r:
        res.append(path)
      continue
    if s[pos] == "(":
      p = path[:]
      p.append("(")
      q.append([pos+1, p, l+1, r])
    else:
      if l > r:
        p = path[:]
        p.append(")")
        q.append([pos+1, p, l, r+1])
      else:
        p = path[:]
        q.append([pos+1,p,l,r])

        pr = ''.join(map(lambda x:str(x),path)).rindex("(")
        for i in xrange(pr,0,-1):
          if path[i] == ")":
            p = path[0:i] + path[i+1:]
            p.append(")")
            q.append([pos+1, p, l, r])
  return res
print bfsInvalidParen("()())()")

""" DP, recur each i at each length, each tab[i,j] is a list, not a scala. traverse.
tab[i,j] = [[v1, exp1], [v2, exp2], ...] can be formed with arr[i:j]
"""
def addOperator(arr, l, r):
  for gap in xrange(r-l):
    for i in xrange(l, r-l-gap):
      j = i+gap
      for k in xrange(i,j):
        for lv, lexpr in tab[i][k]:
          for rv,rexp in tab[k+1][j]:
            out.append([lv+rv, "{}+{}".format(lexp,rexp)])
            out.append([lv*rv, "{}*{}".format(lexp,rexp)])
            out.append([lv-rv, "{}-{}".format(lexp,rexp)])
    tab[i][j] = out
  return tab[l][r]

""" tab[i,j] = [[v1, exp1], [v2, exp2], ...] """
def addOpsTarget(arr, l, r, target):
  out = []
  tab = addOpsTarget(arr, l, r)
  for v, exp in tab[l][r]:
    if v == target:
      out.append(exp)
  return out


""" add operators to exp forms. As no parenth, so go dfs offset.
If allow parenth, need for each gap, fill tab[i,j] with lv/rv,lexpr/rexpr.
"""
def addOperator(arr, l, r, target):
  '''recursion must use memoize'''
  def recur(arr, l, r):
    if not tab[l][r]:
      tab[l][r] = defaultdict(list)
    else:
      return tab[l][r]
    out = []
    if l == r:
      return [[arr[l],"{}".format(arr[l])]]
    # arr[l:r] as one num, no divide
    out.append([int(''.join(map(str,arr[l:r+1]))),
      "{}".format(''.join(map(str,arr[l:r+1])))])
    for i in xrange(l,r):
      L = recur(arr, l, i)
      R = recur(arr, i+1, r)
      for lv, lexp in L:
        for rv, rexp in R:
          out.append([lv+rv, "{}+{}".format(lexp,rexp)])
          out.append([lv*rv, "{}*{}".format(lexp,rexp)])
          out.append([lv-rv, "{}-{}".format(lexp,rexp)])
    tab[l][r] = out
    return out

  tab = [[None]*(l+1) for i in xrange(r+1)]
  out = []
  for v, exp in dfs(arr, l, r):
    if v == target:
      out.append(exp)
  return out
print addOperator([1,2,3],0,2,6)
print addOperator([1,0,5],0,2,5)


""" dfs recur on each segment moving idx, note down prepre, pre,
and calulate this cur in next iteration"""
def addOperator(arr, target):
  def dfs(a, exp, t, prepre, pre, sign):
    out = []
    preval = sign ? prepre + pre : prepre - pre
    if len(a) == 0:
      if preval == t:
        out.append(exp)
        return
    for i in xrange(1, len(a)):
      if i >= 2 and a[i] == 0:
        continue
      cur = a[:i+1]
      dfs(a[i:], "{}+{}".format(exp, cur), t, preval, cur, True)
      dfs(a[i:], "{}-{}".format(exp, cur), t, preval, cur, False)
      dfs(a[i:], "{}*{}".format(exp, cur), t, prepre, pre*cur, sign)
    return out

  for i in xrange(len(arr)):
    if i >= 2 and arr[i] == 0:
      continue
    dfs(arr[i:], arr[:i+1], target, 0, int(arr[:i+1]), True)


''' bottom up, enum pos row up, [[wd1,wd2,...], [wd3,wd4],...]
tab[i]=arr[0:i] is break. tab[i] = tab[k] and arr[k:i] in dict
plusBetween is one leve complicated when enum values '''
def wordbreak(word, dict):
  tab = [None]*len(word)
  for i in xrange(len(word)):
    tab[i] = []   # init result for ith here
    for j in xrange(i):
      wd = word[j:i+1]
      if wd in dict:
        if j == 0:  # check j=0 to avoid tab[j-1] index out
          tab[i].append([wd])
        elif tab[j-1] != None:
          for e in tab[j-1]:
            tmp = e[:]
            tmp.append(wd)
            tab[i].append(tmp)
  return tab[len(word)-1]
#print wordbreak("leetcodegeekforgeek",["leet", "code", "for","geek"])
print wordbreak("catsanddog",["cat", "cats", "and", "sand", "dog"])

''' top down, tab[i] means arr[i:n] is breakable. tab[i] = tab[i+k] and arr[i:i+k] in dict '''
def wordbreak(word, dict):
  tab = [None]*len(word)
  for i in xrange(len(word)-1, -1, -1):
    tab[i] = []
    for j in xrange(i+1,len(word)):
      wd = word[i:j+1]
      if wd in dict:
        if j == len(word)-1:
          tab[i].append([wd])
        elif tab[j+1] != None:
          for e in tab[j+1]:
            tmp = e[:]
            tmp.append(wd)
            tab[i].append(tmp)
  return tab[0]
print wordbreak("catsanddog",["cat", "cats", "and", "sand", "dog"])

""" tab[ia,ib] means the distance between a[0:ia] and b[0:ib] """
def editDistance(a, b):
  tab = [[0]*(len(b)+1) for i in xrange(len(a)+1)]
  for i in xrange(len(a)+1):
    for j in xrange(len(b)+1):
      # boundary condition, i=0/j=0 set in the loop, i
      if i == 0:
        tab[0][j] = j
      elif j == 0:
        tab[i][0] = i
      elif a[i-1] == b[j-1]:
        tab[i][j] = tab[i-1][j-1]
      else:
        tab[i][j] = 1 + min(tab[i][j-1], tab[i-1][j], tab[i-1][j-1])
    return tab[len(a)][len(b)]

""" distinct sequence of t in s; bottom up, each s[i] row a list, each t[j] a col in row. 
tab[i][j] += tab[i-1][j-1], tab[i-1][j], when j==0, and s.i==t.j, tab[i][0] += 1
incl s[i] for t[j], tab[i-1,j-1], excl, tab[i-1,j], you can do dfs recursion also.
when iter col, cp prev row, new col as the last entry in the new row entry list.
"""
def distinctSeq(s,t):
  tab = [[0]*len(t) for i in xrange(len(s))]
  for i in xrange(len(s)):     # bottom up, i
    for j in xrange(len(t)):
      if i == 0:
        if s[0] == t[j]:
          tab[0][j] = 1
        continue
      tab[i][j] = tab[i-1][j]   # not using s[i], so use tab[i-1]
      if t[j] == s[i]:   # use s[i] only when equals
        if j == 0:  # s[i]==t[0], match in s can start from s[i]
          tab[i][0] = tab[i-1][0] + 1
        else:
          tab[i][j] += tab[i-1][j-1]  # reduce one j to j-1
  return tab[len(s)-1][len(t)-1]
print distinctSeq("aeb", "be")
print distinctSeq("abbbc", "bc")
print distinctSeq("rabbbit", "rabbit")

"""
column is src, and inc bottom up each char in target. only 1 char in target, 2, ..
excl s[j], tab[i,j-1], incl s[j], +tab[i-1,j-1], 
  \ a  b   a   b  c
 a  1  1   2   2  2
 b  0 0+1  1  2+1 3
 c  0  0   0   0  3
"""
def distinct(s,t):
  tab = [[0]*len(t) for _ in xrange(len(s))]
  for i in xrange(len(t)):
    for j in xrange(len(s)):
      if t[i] == s[j]:
        if j == 0:
          tab[i][0] = 1
        else:
          if i == 0:
            tab[i][j] = tab[i][j-1] + 1
          else:
            tab[i][j] = tab[i-1][j-1] + tab[i][j-1]
      else:
        if j > 0:
          tab[i][j] = tab[i][j-1]
  return tab[len(t)-1][len(s)-1]
print distinct("aeb", "be")
print distinct("abbbc", "bc")
print distinct("rabbbit", "rabbit")

""" DP boundary case: tab[0][j]=1 empty target string has 1 subseq in any src.
tab[i][0]=0 : empty src has 0 subseq. """
def distinct(s,t):
  dist = [0]*(len(t)+1)
  dist[0] = 1
  for sidx in xrange(1,len(s)):
    pre = 1
    for tidx in xrange(1, len(t)):
      tmp = dist[tidx]
      if s[sidx-1] == t[tidx-1]:
        dist[tidx] = dist[tidx] + pre
      else:
        dist[tidx] = dist[tidx]
      pre = tmp
  return dist[len(t)]
print distinct("bbbc", "bbc")
print distinct("abbbc", "bc")


""" keep track of 2 thing, max of incl cur ele, excl cur ele """
def calPack(arr):
  incl,excl=arr[0],0
  for i in xrange(1,len(arr)):
    curincl = excl+arr[i]
    curexcl = max(incl,excl)
    incl = max(curincl,curexcl)
    excl = curexcl
  return max(incl,excl)


""" DP with O(1), cur = pre, cur = prepre + pre """
def decode(dstr):
  if len(dstr) == 1:
    return 1
  prepre, pre, cur = 0, 1, 0
  for i in xrange(1,len(dstr)+1):
    cur = pre
    if dstr[i] == '0':
      prepre = 0
      continue
    if dstr[i-1-1] == '1' or (dstr[i-1-1]=='2' and dstr[i-1]<='6'):
      cur += prepre  # fib seq here.
    prepre, pre = pre, cur
  return cur
print decode("122")


""" top down offset n -> 1 when dfs inclusion recursion """
def subsetsumRecur(l, offset, n, path, result):
  if offset < 0:
    return    
  if n == l[offset]:
    p = path[:]
    p.append(l[offset])
    result.append(p)
  if n > l[offset]:  # branch incl when when l.i smaller
    p = path[:]   # deep copy new path before recursion.
    p.append(l[offset])
    # dup allowed, offset stay when include, incl 1+ times.
    # subsetsumRecur(l, offset,   n-l[offset], p, result)
    subsetsumRecur(l, offset-1, n-l[offset], p, result)
  # always reduce when excl current, with excl l[i] path.
  subsetsumRecur(l, offset-1, n, path, result)
#path=[];result=[];l=[2,3,4,7];subsetsum(l,3,7,path,result);print result

""" recur i+1, dup not allowed, recur i, dup allowed. """
def comb(arr, t, pos, path, res):
  if pos >= len(arr):
    return
  if t == 0:
    p = path[:]
    res.append(p)
    return
  if arr[pos] <= t:
    path.append(arr[pos])
    comb(arr, t-arr[pos], pos, path, res)
    path.pop()
  comb(arr, t, pos+1, path, res)
path=[];res=[];comb([2,3,6,7],9,0,path,res);print res;

def comb(arr, t, pos, path, res):
  if t == 0:
    p = path[:]
    res.append(p)
    return
  # recur with incl each a.i
  for i in xrange(pos, len(arr)):
    if arr[i] <= t:
      path.append(arr[i])
      comb(arr, t-arr[i], i, path, res)     # dup allowed
      #comb(arr, n-arr[i], i+1, path, res)   # distinct
      path.pop()
  return res
path=[];res=[];comb([2,3,6,7],9,0,path,res);print res;

def coinchange(arr, V):
  tab = [0]*(V+1)
  tab[0] = 1
  # outer loop bottom up coin, effective as incl the coin, and excl when passed.
  for i in xrange(len(arr)):
    for v in xrange(arr[i],V+1):   # bottom up each value
      tab[v] += tab[v-arr[i]]   # not tab[v]=tab[v-1]+1
  return tab[V]
print coinchange([2,3,5], 10)

""" if different coins count the same as 1 coin only, the same as diff ways to stair """
def nways(steps, n):
  ways[i] = steps[i]
  for i in xrange(steps[1], n):
    for step in steps:
      ways[i] += ways[i-step]
  return ways[n-1]

""" bottom up each pos, at each set[0:pos], each val as a new col, entry is a list
[ [v1-list, v2-list, ...], [v1-list, v2-lsit, ...], ...] 
when dup allowed, recur same pos, tab[pos], else tab[pos-1] """
def subsetSum(arr, t, dupAllowed=False):
  tab = [[None]*(t+1) for i in xrange(len(arr))]
  for i in xrange(len(arr)):
    for s in xrange(1, t+1): # iterate to target sum at each arr[i]
      if i == 0:   # boundary, init
        tab[i][s] = []
      else:        # not incl arr[i], some subset 0..pre has value s
        tab[i][s] = tab[i-1][s][:]
      if s == arr[i]:
        tab[i][s].append([s])
      if s > arr[i]:   # incl arr[i]
        if len(tab[i][s-arr[i]]) > 0:  # there
          if dupAllowed: # incl arr[i] multiple times
            for e in tab[i][s-arr[i]]:
              p = e[:]
              p.append(arr[i])
              tab[i][s].append(p)
          elif i > 0:
            for e in tab[i-1][s-arr[i]]:
              p = e[:]
              p.append(arr[i])
              tab[i][s].append(p)
  return tab[len(arr)-1][t]
print subsetSum([2,3,6,7],9)
print subsetSum([2,3,6,7],7, True)
print subsetSum([2,3,6],8)
print subsetSum([2,3,6],8, True)


""" tab[i,s] i digits sum to s, iterate 0..9, = tab[i-1,s-[1..9]]+[1..9]
not as subset sum, no incl/excl of arr[i]. so NOT cp prev row.
bottom up, start from 1 slot, face=[0..9]; dup allowed.
when each slot needs to be distinct, only append j when it is bigger
"""
def combinationSum(n, sm, distinct=False):  # n slots, sum to sm.
  tab = [[None]*(sm+1) for i in xrange(n)]
  for i in xrange(n):       # bottom up n slots
    for s in xrange(sm+1):  # enum to sm for each slot, tab[i,sm]
      if tab[i][s] == None:
        tab[i][s] = []
      for f in xrange(1, 10):  # each slot face, 1..9
        if i == 0:   # tab[0][0..9] = [0..9]
          tab[i][f].append([f])   # first slot, straight.
          continue
        if s > f and tab[i-1][s-f] and len(tab[i-1][s-f]) > 0:
          for e in tab[i-1][s-f]:
            # if need to be distinct, append j only when it is bigger.
            if distinct and f <= e[-1]:
              continue
            de = e[:]
            de.append(f)
            tab[i][s].append(de)
  return tab[n-1][sm]
print combinationSum(3,9,True)

''' contain sol sz < n, [1,6], [7], [1, 0, 6] '''
def combsum(n, t):
  tab = [[None]*(t+1) for i in xrange(n)]
  for i in xrange(n):
    for v in xrange(10):
      for s in xrange(max(1,v), t+1):
        if not tab[i][s]:
          tab[i][s] = []
        if v == s:
          tab[i][s].append([v])
        else:
          if tab[i-1][s-v]:
            for e in tab[i-1][s-v]:
              de = e[:]
              de.append(v)
              tab[i][s].append(de)
          else:
            if i > 0:
              tab[i][s] = tab[i-1][s][:]
  return tab[n-1][t]
print combsum(3,7)


""" m faces, n dices, num of ways to get value x.
bottom up, 1 dice, 2 dices,...,n dices. Last dice take face value 1..m.
[[v1,v2,...], [v1,v2,...], d2, d3, ...], each v1 column is cnt of face 1..m
tab[d,v] += tab[d-1,v-[1..m]]. Outer loop is enum dices, so it's like incl/excl.
"""
def dice(m,n,x):
  tab = [[0]*n for i in xrange(x+1)]
  ＃ init bottom, one coin only
  for f in xrange(m):
    tab[1][f] = 1
  for i in xrange(1, n):      # bottom up coins, 1 coin, 2 coin
    for v in xrange(x):       # bottom up each value after inc coin
      for f in xrange(m,v):   # iterate each face value
        tab[i][v] += tab[i-1][v-f]
  return tab[n,x]


""" tab[i,v] = tot non descreasing at ith digit, end with value v
bottom up each row i, each digit as v, sum. [[1,2,3...], [1,2..], ...]
tab[n,d] = sum(tab[n-1,k] where k takes various value[0..9]
Total count = ∑ count(n-1, d) where d varies from 0 to n-1
"""
def nonDecreasingCount(n):
  tab = [[0]*10 for i in xrange(n)]
  for v in xrange(10):
    tab[0][v] = 1
  for i in xrange(1,n):
    for v in xrange(10):
      for k in xrange(v+1): # k is capped by prev digit v
        tab[i][v] += tab[i-1][k]
  tot = 0
  for v in xrange(10):
    tot += tab[n-1][v]
  return tot
assert(nonDecreasingCount(2), 55)
assert(nonDecreasingCount(3), 220)

''' digitSum(2,5) = 14, 23, 32, 41 and 50
bottom up each row of i digits, each val is a col, 
[[1,2,..], [v1, v2, ...], ..] tab[i,v] = sum(tab[i-1,k] for k in 0..9)
'''
def digitSum(n,s):
  tab = [[0]*max(s+1,10) for i in xrange(n)]
  for i in xrange(10):
    tab[0][i] = 1
  for i in xrange(1,n):     # bottom up each slot
    for v in xrange(s+1):   # bottom up each sum
      for d in xrange(v):   # foreach face(0..9)
        tab[i][v] += tab[i-1][v-d]
  return tab[n-1][s]
assert(digitSum(2,5), 5)
assert(digitSum(3,6), 21)


"""
count tot num of N digit with sum of even digits 1 more than sum of odds
n=2, [10,21,32,43...] n=3,[100,111,122,...980]
use 2 tables, odd[i,v] and even[i,v]
"""
def digitSumDiffOne(n):
  odd = [[0]*18 for i in xrange(n)]
  even = [[0]*18 for i in xrange(n)]
  for l in xrange(n):
    for k in xrange(2*9):
      for d in xrange(9):
        if l%2 == 0:
          even[l][k] += odd[l-1][k-d]
        else:
          odd[l][k] += even[l-1][k-d]

""" odd[i,s] is up to ith odd digit, total sum of 1,3,ith digit sum to s.
for digit xyzw, odd digits, y,w; sum to 4 has[1,3;2,2;3,1], even digit, sum to 5 is
[1,4;2,3;3,2;4,1], so tot 3*4=12, [1143, 2133, ...]
so diff by one is odd[i][s]*even[i][s+1]
"""
def diffone(n):
  odd = [[0]*10*n for i in xrange(n)]
  even = [[0]*10*n for i in xrange(n)]
  for k in xrange(1,10):
    even[0][k] = 1
    odd[1][k] = 1
  odd[1][0] = 1
  for i in xrange(2,n):
    for s in xrange((i+1)*10):
      for v in xrange(min(s,10)):
        if i%2 == 0:
          even[i][s] += even[i-2][s-v]
        if i%2 == 1:
          odd[i][s] += odd[i-2][s-v]
  cnt = 0
  for s in xrange((n-1)*10):
    if (n-1)%2 == 0:
      cnt += odd[n-2][s] * even[n-1][s+1]
    else:
      cnt += odd[n-1][s] * even[n-2][s+1]
  return cnt
print diffone(3)


""" adding + in between. [1,2,0,6,9] and target 81.
tab[i,sm] exist [0..i] val sum to sm
more complex than subset sum, one loop from i left to coerce k:i as cur num.
bottom up row, for each j to i that forms num, then for each num 
"""
def plusBetween(arr, target):
  def toInt(arr, st, ed):
    return int("".join(map(lambda x:str(x), arr[st:ed+1])))
  tab = [[False]*(target+1) for i in xrange(len(arr))]
  for i in xrange(len(arr)):
    for j in xrange(i,-1,-1):  # one loop j:i to coerce current num
      num = toInt(arr,j,i)
      if j == 0 and num <= target:
        tab[i][num] = True   # tab[i,num], not tab[i,target]
      # enum all values between num to target, for tab[i,s]
      for v in xrange(num, target+1):
        tab[i][v] |= tab[j-1][v-num]
  return tab[len(arr)-1][target]
print plusBetween([1,2,0,6,9], 81)


""" the idea is to use rolling hash, A/C/G/T maps to [00,01,10,11]
or use last 3 bits of the char, c%7. 3 bits, 10 chars, 30bits, hash stored in last 30 bits
"""
from collections import defaultdict
def dna(arr):
  tab = defaultdict(int)
  mask = (1 << 30) - 1  # hash stores in the last 30 bits
  hashv,out = 0,[]
  for i in xrange(len(arr)):
    v = arr[i]
    vh = ord(v)&7
    hashv = ((hashv << 3) & mask) | vh
    if i > 9 and tab[hashv] > 0:
      out.append(arr[i-9:i+1])
    tab[hashv] += 1
  return out
print dna("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")


""" tab[i][j] whether a[0:i] is interleave b[0:j]; tab[i,j] = tab[i-1,j], etc"""
def isInterleave(a, b, c):
  tab = [[[0]*(len(b)+1) for i in xrange(len(a)+1)]]
  for i in xrange(len(s1)+1):
    for j in xrange(len(s2)+1):
      if i == 0 and j == 0:
        tab[i][j] = True
      elif i == 0 and b[j-1] == c[j-1]:
        tab[i][j] = tab[i][j-1]
      elif j == 0 and a[i-1] == c[i-1]:
        tab[i][j] = tab[i-1][j]
      elif c[i+j-1] == a[i-1]:
        tab[i][j] = tab[i-1][j]
      elif c[i+j-1] == b[j-1]::
        tab[i][j] = tab[i][j-1]
  return tab[len(a)][len(b)]

""" 2 string, lenth,ia,ib; 3 dim, in each len, test sublen k/len-k. tab[0..k][ia][ib];
normally, one dim ary with each tab[lenth,i], j=i+len,
here, vary all 3; tab[len,i,j], break lenth in sublen, and i,j with sublen.
tab[l,i,j] = 1 iff (tab[k,i,j] and tab[l-k,i+k,j+k]
"""
def isScramble(a, b):
  tab = [[[0]*len(a) for i in xrange(len(b))] for j in xrange(len(a))]
  for length in xrange(len(a)):
    for ia in xrange(len(a)-length):
      for ib in xrange(len(b)-length):
        if length == 0:
          tab[length][ia][ib] = a[ia] == b[ib]
        for k in xrange(length):
          if tab[k][ia][ib] and tab[length-k][ia+k][ib+k]:
            tab[length][ia][ib] = True
          elif tab[k][ia][ib+length-k] and tab[length-k][ia+k][ib]:
            tab[length][i][j] = True
  return tab[len(a)-1][0][0]


""" next palindrom num, cp left -> rite, inc mid, propagate carry left/rite is mid=9"""
def nextPalindrom(n):    
  s = list(str(n))
  sz = len(s)
  l,r,m = (sz-1)/2,sz/2,(sz-1)/2
  while l >= 0:
    if s[l] != s[r]:
      s[r] = s[l]
    l -= 1
    r += 1
  if int("".join(s)) <= n:
    s[m] = str(int(s[m])+1)
  if sz % 2 == 0:
    s[m+1] = s[m]
  if s[m] == "10":
    s[m] = "0"
    l,r = m-1, min(sz-1, m+1)
    if s[r] == "10":
        s[r] = "0"
        r += 1
    while s[l] == "9":
        s[l] = s[r] = "0"
        l -= 1
        r += 1
    if l >= 0:
        s[l] = str(int(s[l])+1)
        s[r] = s[l]
    else:
        del s[m]
        s.insert(0, "1")
        s.append("1")
  return int("".join(s))
assert nextPalindrom(9) == 11
assert nextPalindrom(99) == 101
assert nextPalindrom(1234) == 1331

""" min insertion into str to convert to palindrom. tab[i,j]=min ins to convert str[i:j]
tab[i][j] = tab[i+1,j-1] or min(tab[i,j-1], tab[i+1,j])
"""
def minInsertPalindrom(arr):
  tab = [[0]*len(arr) for i in xrange(len(arr))]
  for sublen in xrange(1,len(arr)):
    for i in xrange(len(arr)-sublen):
      j = i + sublen
      if arr[i] == arr[j]:
        tab[i][j] = tab[i+1][j-1]
      else:
        tab[i][j] = min(tab[i][j-1], tab[i+1][j])
  return tab[0][len(arr)-1]

def palindromMincut(s, offset):
  ''' partition s into sets of palindrom substrings, with min # of cuts 
      mincut[offset] = 1 + min(mincut[k] for k in [offset+1..n] where s[offset:k] is palindrom)
  '''
  mincut = INT_MAX
  for i in xrange(offset+1, len(s)):
    if isPalindrom(s, offset, i):
      partial = palindromMincut(s, i)   # recursion, aggregate on top.
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

""" next palindrom """
def nextPalindrom(v):
  # arr[0:li] -> arr[li+1:ri]
  def copyLeftToRite(arr, li, ri):
    left = arr[:li+1]
    left.reverse()
    arr[ri:] = left
  arr = list(str(v))
  sz = len(arr)
  leftSmaller = False
  if sz % 2:
    l,r = (sz-1)/2, sz/2
  else:
    l,r = (sz-1)/2, (sz+1)/2
  while l >= 0 and r < sz and arr[l] == arr[r]:
    l -= 1
    r += 1
  if l > 0 and arr[l] < arr[r]:
    leftSmaller = True
  copyLeftToRite(arr, l, r)
  if leftSmaller:
    if sz % 2 == 1:
      mid = (sz-1)/2
      arr[mid] = str(int(arr[mid])+1)
    else:
      ml,mr = (sz-1)/2, sz/2
      arr[ml] = str(int(arr[ml])+1)
      arr[mr] = str(int(arr[mr])+1)
  return int(''.join(arr))

""" palindrom pair, concat the two words forms a palindrom
a) s1 or s2 is blank. 
b) s1 = reverse(s2), 
c) s1[0:cut] is palindrom and s1[cut:]=reverse(s2)
d) s1[0:cut]=reverse(s2) and s1[cut+1:] is palindrom
build a HashMap to store the String-idx pairs.
"""
def palindromPair(arr):
  pass

""" min number left after remove triplet.
[2, 3, 4, 5, 6, 4], ret 0, first remove 456, remove left 234.
tab[i,j] track the min left in subary i..j
"""
def minLeft(arr, k):
  tab = [[0]*len(arr) for i in len(arr)]
  def minLeftSub(arr, lo, hi, k):
    if tab[lo][hi] != 0:
        return tab[lo][hi]
    if hi-lo+1 < 3:  # can not remove ele less than triplet
        return hi-lo+1
    ret = 1 + minLeftSub(arr, lo+1, hi, k)

  for i in xrange(lo+1,hi-1):
    for j in xrange(i+1, hi):
      # lo, i, j is k-spaced triplet, and [lo+1..i-1],[i+1..j-1] can be removed
      if arr[i] == arr[lo] + k and arr[j] == arr[lo] + 2*k and
        minLeftSub(arr, lo+1, i-1, k) == 0 and 
        minLeftSub(arr, i+1, j-1, k) == 0:
          # now triplet lo,i,j also can be removed
          ret = min(ret, minLeftSub(arr, j+1, hi, k))
    tab[lo][hi] = ret
    return ret

''' sub regioin [l..r], consider m as the LAST to burst, so last 3 l,m,r.
matrix product tab[l,r] = tab[l,k]+tab[k+1,r]+p[l-1]*p[m]*p[r]
tab[l,r] = max(tab[l,k]+tab[k+1,r]+arr[l]*arr[k]*arr[r])
'''
def burstBalloon(arr):
  earr = [1] + arr + [1]   # set boundary
  sz = len(earr)
  tab = [[0]*(sz) for i in xrange(sz)]
  for gap in xrange(1, len(arr)+1):
    for l in xrange(len(arr)):  # largest l is 2 less than
      r = l + gap + 1
      if r >= sz:
        break
      for m in xrange(l+1,r):
        tab[l][r] = max(tab[l][r], tab[l][m]+tab[m][r]+earr[l]*earr[m]*earr[r])
  return tab[0][sz-1]
print burstBalloon([3, 1, 5, 8])

""" m[i,j]=matrix(i,j) = min(m[i,k]+m[k+1,j]+arr[i-1]*arr[k]*arr[j]) """
def matrix(arr):
  tab = [[0]*len(arr) for i in xrange(len(arr))]
  for gap in xrange(2,len(arr)):
    for i in xrange(1, len(arr)-gap+1):
      j = i + gap -1
      tab[i][j] = sys.maxint
      for k in xrange(i,j):
        tab[i][j] = min(tab[i][j], tab[i][k] + tab[k+1][j] + arr[i]*arr[k]*arr[j])
  return tab[1][len(arr)-1]
print matrix([1,2,3,4])

''' bfs search on Matrix for min dist '''
from collections import deque
def matrixBFSk(G):
  q = deque()
  for i in xrange(m):
    for j in xrange(n):
      if G[i,j] == 1:  # start from cell val 1
        G[i,j] = 0
        q.append([i,j])  # can do bfs from this cell
      else:
        G[i,j] = sys.maxint  # unknow space init dist to max
  while len(q) > 0:
    [i,j] = q.popleft()
    if G[i,j] < k:
      if G[i+1,j] == sys.maxint:  # cell not explored yet
        G[i+1,j] = G[i,j] + 1     # set dist+1
        q.append([i+1,j])
      if G[i-1,j] == sys.maxint:
        G[i-1,j] = G[i,j] + 1
        q.append([i+1,j])    
      if G[i,j+1] == sys.maxint:
        G[i,j+1] = G[i,j] + 1
        q.append([i,j+1])
      if G[i,j-1] == sys.maxint:
        G[i,j-1] = G[i,j] + 1
        q.append([i,j-1])
  for i in xrange(m):
    for j in xrange(n):
      if G[i,j] < k:
        G[i,j] = 1

""" word ladder, bfs, for each intermediate word, at each position, 
enum all ascii to get new char, recur not visited.
"""
from collections import deque
import string
def wordladder(start, end, dict):
  def getNext(wd, seen):
    nextwds = []
    for i in xrange(len(list(wd))):
      for c in list(string.ascii_lowercase):
        l = list(wd)
        l[i] = c
        nextwd = "".join(l)
        if nextwd == end or (nextwd in dict and not nextwd in seen):
          nextwds.append(nextwd)
    return nextwds
  def replace(wd):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(wd[:i],wd[i:]) for i in xrange(len(wd)+1)]
    replace = [a + c + b[1:] for a,b in splits for c in alphabet if b]
    nextwds = []
    for e in replace:
      nextwd = "".join(e)
      if nextwd == end or (nextwd in dict and not nextwd in seen):
        nextwds.append(nextwd)
        seen.add(nextwd)
    return nextwds
  ''' bfs search '''
  if start == end:
    return
  ladder = []
  seen = set()     # global visited map for all bfs iterations
  q = deque()
  q.append([start,[]])
  while len(q) > 0:
    [wd,path] = q.popleft()
    for nextwd in getNext(wd, seen):
      if nextwd == end:
        path.append(nextwd)
        ladder.append(path)
      else:
        seen.add(nextwd)
        npath = path[:]
        npath.append(wd)
        q.append([nextwd, npath])
  return ladder
start = "hit"
end = "cog"
dict = set(["hot","dot","dog","lot","log"])
print wordladder(start, end, dict)

""" consider every cell in borad as start, dfs find all word in dict
reset visited map for each dfs iteration.
Optimization: build Trie for all words, then backtrack bail out early
"""
def boggle(G, dictionary):
  def isValid(r,c,M,N,seen):
    if r >= 0 and r < M and c >= 0 and c < N and G[r][c] not in seen:
      return True
    return False
  ''' dfs recur from r,c search for all cells '''
  def dfs(G, dictionary, r, c, seen, path, out):
    if "".join(path) in dictionary:
      out.append("".join(path))
    for nr,nc in [[-1,-1],[-1,0],[-1,1]...]:
      if isValid(nr,nc,M,N,seen):
        path.append(G[nr][nc])
        seen.add(G[nr][nc])
        dfs(G, dictionary, nr, nc, seen, path, out)
        path.pop()
        seen.remove(G[nr][nc])
      return out
  M,N = len(G),len(G[0])
  out, seen = [], set()
  for i in xrange(M):     # start from each cell
    for j in xrange(N):
      seen.add(G[i][j])
      path = []
      path.append(G[i][j])
      dfs(G, dictionary, i, j, seen, path, out)
      path.pop()
      seen.remove(G[i][j])
  return out

""" w[i]: cost of word wrap [0:i]. w[j] = w[i-1] + lc[i,j]"""
def wordWrap(words, m):
  sz = len(words)
  #extras[i][j] extra spaces if words i to j are in a single line
  extras = [[0]*sz for i in xrange(sz)]
  lc = [[0]*sz for i in xrange(sz)] # cost of line with word i:j
  for linewords in xrange(sz/m):
    for st in xrange(sz-linewords):
      ed = st + linewords
      extras[st][ed] = abs(m-sum(words, st, ed))
  for i in xrange(sz):
    for j in xrange(i):
      lc[i] = min(lc[i] + extras[j][i])
  return lc[sz-1]


"""
http://www.geeksforgeeks.org/find-the-largest-rectangle-of-1s-with-swapping-of-columns-allowed/
"""

if __name__ == '__main__':
    #qsort([363374326, 364147530 ,61825163 ,1073065718 ,1281246024 ,1399469912, 428047635, 491595254, 879792181 ,1069262793], 0, 9)
    #qsort3([1,2,3,4,5,6,7,8,9,10], 0, 9)
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