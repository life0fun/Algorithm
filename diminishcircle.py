#!/usr/bin/env python

"""
  Facebook Hacker cup Diminishing circle
  http://www.facebook.com/hackercup/problems.php/?round=167482453296629

  basic idea: recursion. f(n,k) = f(n-1,k) = ... 
  [0,    1,   2..k-1, k, k+1, ...,n-1]   ==> f(x) = m % n  k=m%(n-1)
  [n-k-1,n-k, ...n-2,     0,     ,n-k-2] ==> p(x) = (x-k-1)%n, 

  Index x in iteration with n-1 ele maps to x+(k+1)%n index in iteration with n ele.
  the key here is, we can know the winner at last round, but we need to map it to the
  right index in the original array in iteration 0.

  f(n,m),  the n-1 numbers that left, then map to f2(0 base, n-1,m) with x=x-k-1 
            so index i in f2, map back to f1 as i+(k+1)%n, where k=m%(n-1)


  Simple heuristic:
  start from one ele array, winner is the first.
  add one ele into array, winner is m % len(l)
  from here, add one ele, winner add m and mod len(array)


  Haijin Yan, Feb, 2011
"""
import sys, os
import operator
import pdb

class DimCircle(object):
    _INFINITY = 1e3

    def __init__(self, n, k):
        self.p = [i for i in xrange(1,n+1)]
        self.k = k

    # calucation iteration n-1, then calc iteration n based on n-1
    def run(self, n, m):
        winner = 0
        if n == 1:
            return winner 
        # list len add one, winner add m
        for circlesize in xrange(2,n+1):
            winner = (winner+m+1)%circlesize

        return winner


if __name__ == "__main__":
    c = DimCircle(6, 9)
    print c.run(6,9)
    print c.run(5,4)
    print c.run(3,2)
    print c.run(9,3)
