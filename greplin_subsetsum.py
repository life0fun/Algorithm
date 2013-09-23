#!/usr/bin/env python

import sys

"""
find out all the sets in the array where the largest ele in set is the sum of the rest
   0 1 2 3 4 5
   -----------
   1 2 4 5 6 11
when adding 11, call decisionTree(4,11). recursively on two branches.
include 6, then count all prev sets with sum to 5, which DP re-use the result when cal 5.
not include 6, then count all prev sets sum to 11, recursively, and tab down it.
The reason you only need to check the immediate prev is due to sorted list.

tab[i,j] = total solution: no. of _some_ subsets in l[0..i] has val sum to j.
I index into items in the list, J index into the enumerated value space.
Focus on the result, enum all possible result, not on the steps.
"""
def decisionTree(l,root,val,table):
    # find return all subset up to idx k that are sum to n

    #print 'decision tree:' ,root, val, l
    # exit condition, l[k] = n, or n <= 0
    if val <= 0 or root < 0:
        yield False, [0]
        return

    if l[root] == val:
        yield True, [l[root]]
        # can't stop after yield...need to continuously drive down !!!!

    prevroot = root-1
    # not include root, val keep the same and recursive on prevroot
    for found, subset in decisionTree(l, prevroot, val, table):
        if found:
            yield True, subset

    # include root, reduce val by the val of root, recursive on prevroot
    for found, subset in decisionTree(l, prevroot, val-l[root], table):
        if found:
            subset.extend([l[root]])
            yield True, subset

"""
dynamic programming version
all subset stop at root that sum to val.
ith solution is i-1th extension with two branches: include i-1th or not include
when looking up table, make sure indices are great than zero, as python will wrap around index implicitly.
"""
def decisionTreeDP(l,root,val,table):     # l ary, from 0 - root(idx), subset sum to val.
    if val <= 0 or root < 0:
        return 0

    # this is the leaf node that will be hit eventually as we are recursive idx down and val down.
    # Like segment tree or merge sort boundary case.
    if l[root] == val:  # nature check header first.
        table[root][val] += 1
        # can't stop after match here...need to continuously drive down !!!!

    prevroot = root-1
    if prevroot < 0:
        return table[root][val]

    ''' not include prev node, all subsets sum to l[root], eq. sigma(k=1,idx-1,l[root]'''
    if table[prevroot][val] > 0:
        #print 'found:', 'root:', root, 'l[root]:',l[root], 'val:', val, 'prev:', prevroot, 'l[prevroot]:', l[prevroot], 'tab.root.val:', table[root][val], 'tab.prev.val:', table[prevroot][val]
        table[root][val] += table[prevroot][val]
    else:
        table[root][val] += decisionTreeDP(l,prevroot,val,table)

    ''' include prev node, all subsets sum to l[root]-l[prev], eq. sigma(k=1,idx-2, l[root]-l[prev])+l[prev]=l[root]'''
    if val >= l[root]:
        if table[prevroot][val-l[root]] > 0:
            #print 'foundval-root:', 'root:', root, 'l[root]:',l[root], 'val:', val, 'prev:', prevroot, 'l[prevroot]:', l[prevroot], 'tab.root.val:', table[root][val], 'tab.prev.val:', table[prevroot][val]
            table[root][val] += table[prevroot][val-l[root]]
        else:
            table[root][val] += decisionTreeDP(l,prevroot,val-l[root],table)

    return table[root][val]

def subsetSums(l):
    l.sort()
    num = len(l)
    maxv = max(l)+1
    table = [[0]*maxv for i in xrange(num+1)]
    totalcnt = 0

    ''' first method, getting all the sets also'''
    i = 2
    while i < len(l):
        for found, subset in decisionTree(l, i-1, l[i], table):
            if found:
                subset.extend([l[i]])
                yield subset
        i += 1

    ''' second way, just tab down the total counts'''
    for i in xrange(2, len(l), 1):
        totalcnt += decisionTreeDP(l,i-1,l[i],table)

    print 'total count DP:', totalcnt

def testDecisionTree(numbers):
    sumnsets = []

    for subset in subsetSums(numbers):
        sumnsets.append(subset)

    #sumnsets.sort()
    for e in sumnsets:
       print e

    print 'total subset:', len(sumnsets)

if __name__ == '__main__':
    #numbers = [1, 2, 4, 5, 6]
    #numbers = [1, 2, 3, 4, 6]
    numbers = [3, 4, 9, 14, 15, 19, 28, 37, 47, 50, 54, 56, 59, 61, 70, 73, 78, 81, 92, 95, 97, 99]

    testDecisionTree(numbers)
