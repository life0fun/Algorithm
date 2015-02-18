#!/usr/bin/env python

from collections import defaultdict

def word(s, offset, ctx):
  dict = ["cat", "cats", "and", "sand", "dog"]
  result = []
  sz = len(s)
  if offset == sz:
    result.append(["EOF"])
  for i in xrange(offset+1, sz):
    cw = s[offset:i+1]  # end+1 as s[start:end] as end is not closing.
    if cw in dict:
      for w in word(s, i+1, ctx):  # next offset starts i+1
        if w:
          w.append(cw)
          result.append(w)
  ctx['sol'][offset] = result
  return result


# S = "rabbbit", T = "rabbit"
def seq(s, sidx, t, tidx, parent):
  result = []
  if tidx == len(t):
    result.append(parent)
    return result
  
  if sidx == len(s):
    return result
  
  if s[sidx] == t[tidx]:
    p = parent[:]
    p.append(str(sidx) + ":" + s[sidx])
    for r in seq(s, sidx+1, t, tidx+1, p):
      if r:
        result.append(r)

  for x in seq(s, sidx+1, t, tidx, parent):
    if x:
      result.append(x)

  return result


def zigzag(s):
  def partition(s, n):
    sidxed = [(i,j) for i,j in enumerate(s)]
    z = filter(lambda e: e[1] if e[0] % n-1 else None, sidxed)
    z = map(lambda e: e[1], z)
    return z


def longestvalidp(s):
  stk = []
  curstart = maxl = 0
  for i in xrange(len(s)):
    cur = s[i]
    if cur == "(":
      stk.append(i)
    else:
      if len(stk) == 0:
        curstart = i
      else:
        p = stk.pop()
        maxl = max(maxl, i-p+1)
  return maxl


# for each bar x, find the first smaller left, first smaller right,
# area(x) = first_smaller_right(x) - first_smaller_left(x) * X
def largestRectangle(l):
  maxrect = 0
  l.append(0)  # sentinel
  l.insert(0, 0)
  stk = []
  left_first_smaller = [0 for i in xrange(len(l))]
  right_first_smaller = [len(l) for i in xrange(len(l))]
  left_first_smaller[0] = 0
  stk.append(0)  # 
  for i in xrange(1,len(l)):
    curval = l[i]
    topidx = stk[-1]
    topval = l[topidx]
    # push to stk when current > stk top, so first smaller left of current set.
    if curval > topval:
      left_first_smaller[i] = topidx
    # when cur smaller than stk top, stk top right is known, continue pop stk.
    elif curval <= topval:
      right_first_smaller[topidx] = i  # stk top is peak. update its left/right.
      stk.pop()
      while len(stk) > 0:
        topidx = stk[-1]
        topval = l[topidx]
        if topval > curval:  # continue pop stk if cur is first smaller right of stk top
          right_first_smaller[topidx] = i
          stk.pop()
        else:
          left_first_smaller[i] = topidx
          break
    stk.append(i)  # stk in current

  print left_first_smaller
  print right_first_smaller
  for i in xrange(len(l)):
    maxrect = max(maxrect, l[i]*(right_first_smaller[i]-left_first_smaller[i]-1))

  return maxrect


def longestone(L):
  tab = []  # tab[i] how many 1s between L[i-1], L[i] where both are 0.
  cnt = 0
  maxones = 0
  for i in xrange(len(L)):
    if L[i] == 0:
      tab.append([i, cnt])
      cnt = 0
    else:
      cnt += 1

  tab.append([i, cnt])

  print tab
  for i in xrange(len(tab)-1):
    curidx, lones = tab[i]
    nxtidx, rones = tab[i+1]
    maxones = max(lones+rones+1, maxones)

  return maxones

def closestpair(A, B, t):
  l, r = 0, len(B)-1
  mindiff = 1000
  minl = minr = 0
  while l < len(A) and r >= 0:
    if abs(A[l] +  B[r] - t) < mindiff:
      mindiff = abs(A[l] +  B[r] - t)
      minl = l
      minr = r
    if A[l] +  B[r] > t:
      r -= 1
    elif A[l] +  B[r] < t:
      l += 1
  return [minl, minr]


def common(A,B,C):
  ai,bi,ci = 0,0,0
  result = []
  while ai < len(A) and bi < len(B) and ci <  len(C):
    maxele = max(A[ai],B[bi],C[ci])
    if A[ai] == maxele and B[bi] == maxele and C[ci] == maxele:
      result.append(maxele)
      ai += 1
      bi += 1
      ci += 1
      continue
    if A[ai] < maxele:
      ai += 1
    if B[bi] < maxele:
      bi += 1
    if C[ci] < maxele:
      ci += 1

  print result

def spacestr(L):
  result = []
  hd = L[0]
  if len(L) == 1:
    result.append(hd)
    return result

  for w in spacestr(L[1:]):
    result.append(str(hd+w))
    result.append(str(hd + " " + w))
  return result


def kseq(n, k):
  result = []
  if n == k:
    e = [i for i in xrange(1, n+1)]
    result.append(e)
    return result
  if k == 1:
    for e in xrange(1,n+1):
      result.append([e])
    return result

  for e in kseq(n-1, k-1):
    e.append(n)
    result.append(e)
  for e in kseq(n-1, k):
    result.append(e)

  return result

if __name__ == '__main__':
  ctx = dict()
  ctx['sol'] = dict()
  z = word("catsanddog", 0, ctx)
  print z

  # print seq("rabbbit", 0, "rabbit", 0, [])
  # print longestvalidp(")((((()((()())")

  # print "largest rect "
  # print largestRectangle([6, 2, 5, 4, 5, 1, 6])

  # print "largest ones"
  # print longestone([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
  
  # print "closest pair"
  # print closestpair([1, 4, 5, 7], [10, 20, 30, 40], 50)

  # print common([1, 5, 5], [3, 4, 5, 5, 10], [5, 5, 10, 20])
  
  print spacestr("ABC")  

  print sorted(kseq(7, 3), key=lambda x: x[0])

