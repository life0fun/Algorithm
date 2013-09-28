#!/usr/bin/env python

import sys
import math
import pdb

int2bit = lambda x : x > 0 and int2bit(x>>1) + str(x&1) or ''

def int2bit(num, bits=16):
    r = ''    # define result string first
    tmp = num
    while bits:
        r = ('1' if num&1 else '0') + r
        bits = bits-1
        num = num >> 1
    print tmp, " ==> " ,r
    return r

def maxInt():
    i = -1
    r = int2bit(i,32)
    i = 1 << 31
    int2bit(i)

def numofones(num):
    n = 0
    tmp = num
    while num:
        num = num & (num-1)
        n += 1
    print "num %d has %d 1 bits" %(tmp, n)

def sum(num):
    low = 1
    high = 2
    sum = low + high

    while high < num:
        if sum == num:
            print 'sum=%d, start=%d, end=%d' %(sum, low, high)

        if sum + high+1 <= num:
            high += 1
            sum += high
            #print 'sum=%d, start=%d, end=%d' %(sum, low, high)
            continue
        else:
            sum -= low
            low += 1
            #print 'sum=%d, start=%d, end=%d' %(sum, low, high)
            continue

def swap(l, head, tail):
    while head < tail:
        if l[tail] % 2:
            swap(head, tail)
            head += 1
        else:
            tail -= 1

def allnum(nbits):
    if nbits == 1:
        for v in xrange(10):
            yield v
    else:
        for v in xrange(10):
            for n in allnum(nbits-1):
                yield str(v)+str(n)

''' given a list of num, find the small num that concat all of them
    [3, 32, 325, 31, 322, 33, 199]
    key: fill the missing with the most sig digit. align length.
'''
def smallestConcat(l):
    maxlen = 0
    lstr = []
    for e in l:
        maxlen = len(str(e)) if len(str(e)) > maxlen else maxlen
        lstr.append([len(str(e)), str(e)])
    for e in lstr:
        strlen, numstr = e  # auto unpacking
        for i in xrange(maxlen - strlen):
            numstr += numstr[0]  # append the missing with the leading, most sig digit
        e[1] = numstr
    lstr.sort(key=lambda x:x[1])

    print lstr

    minconcat = ''
    for e in lstr:
        strlen, numstr = e
        minconcat += numstr[:strlen] + '-'

    print minconcat

''' find max sum of consecutive numbers in a list'''
def maxSum(l):
    maxstart=maxstop=maxsum = 0
    start = stop = sum = 0

    while stop < len(l):
        sum += l[stop]
        if sum < 0:
            start = stop + 1
            sum = 0
        elif sum > maxsum:
            maxstart = start
            maxstop = stop
            maxsum = sum
        stop += 1

    print l[maxstart:maxstop+1]

'''
given a number n, finds all pairs of numbers x and y such that sum of square = n
'''
def sumofsquare(n, start):
    mid = math.sqrt(n)
    if mid == int(mid):
        return [mid]

    for i in xrange(start, mid, 1):
        high = n - i*i
        l = sumofsqure(n-i*i, start+1)
        if not l:
            l.append(i)
            return l
    return None

'''
Give a string of numbers, add minimum + in between so sum of a given number.
Example:"382834" => 38+28+34 = 100
http://www.topcoder.com/stat?c=problem_statement&pm=2829&rd=5072
'''
def QuickSum(l, sum):
    #l = list(l)
    rows = len(l)
    table = [[-1]*(sum+1) for i in xrange(rows)]

    def decisionDP(l, idx, sum, table):
        #print "DP=", idx, l[idx], sum
        if int(l[idx]) > sum:
            return False, None

        if table[idx][sum] >= 0:
            print l[0:idx], sum, table[idx][sum]
            return True, table[idx][sum]

        for start in xrange(idx,-1,-1): # idx the last idx, range need +1, anything after : need offset one more
            if int(l[start:idx+1]) == sum:
                table[idx][sum] = 0
                print l[start:idx+1], sum, table[idx][sum]
                return True, 0
            elif int(l[start:idx+1]) < sum:
                #print 'Recusion:', l[start:idx+1], sum, sum-int(l[start:idx+1])
                ret, signs = decisionDP(l, start-1, sum-int(l[start:idx+1]), table)
                if ret:
                    table[idx][sum] += (signs+1)
                    print 'sum:', l[start:idx+1], sum, table[idx][sum]
        return False, None

    print '--- starting---', l, sum
    decisionDP(l, len(l)-1, sum, table)

def testQuickSum():
    s = '382834'
    sum = 100
    print '--- testing insert min plus with', s, sum
    QuickSum(s,sum)

"""
vending machine coin change.
greedy wont work, 55, take 2 quater, no nickle, failed.
branch and bound recursion with greedy preference.
1. recursion,  2. recursion with max/min thresh, 3. branch bound recursion with greedy preference.
l carries partial results during recursive, l[0]= # of quarters, l[1] = # of dimes, etc.
when leaf situation reached, ret partial result
"""
def coinChange(tot, nqarters, ndimes, nnickles, l):
    if not tot:
        print ' === final solution: ====', l
        # upBeat(l)
        return True

    #print ' --recursion: ', tot, nqarters, ndimes, nnickles, l
    ql = l[:]   # when you do this deep copy, obj reference already changed.
    if tot >= 25 and nqarters > 0:
        ql[0] += 1
        if coinChange(tot-25, nqarters-1, ndimes, nnickles, ql):
            #l = ql  # here I changed the ref of l to ql, However, the passed in global obj still exist, not changed.
            for i in xrange(len(ql)):  # copy out value to the passed in arg.
                l[i] = ql[i]
            return True
        else:  # no, can not do quarter again.
            return coinChange(tot, 0, ndimes, nnickles, l)

    if tot >= 10 and ndimes > 0:
        ql[1] += 1
        if coinChange(tot-10, nqarters, ndimes-1, nnickles, ql):
            for i in xrange(len(ql)):  # copy out value to the passed in arg.
                l[i] = ql[i]
            return True
        else:  # can not do quarter and dime
            return coinChange(tot, 0, 0, nnickles, l)

    if nnickles >= tot/5:
        l[2] += tot/5
        return coinChange(0, 0, 0, nnickles-tot/5, l)
    else:
        return False

def testCoinChange():
    r = [0,0,0]
    coinChange(65, 4, 4, 4, r)
    print r


''' give l with k values sorted, find the smallest k values taking from
    a(i), a(i)+a(j), a(i)+a(j)+a(l) for any i,j,l < k
    Ex: A[7] = {3,4,5,15,19,20,25}
    output B[7] = {3,4,5,(3+4),(3+5),(4+5),(3+4+5)}
'''
def topKValue(l):
    pass


"""
Given an array of non-negative integers A and a positive integer k, we want to:
Divide A into k or fewer partitions, such that the maximum sum over all the partitions is minimized.

DP[i][k] denotes the min of max sum(partitions) from 0 end at i, with k partition.
i varies from 2..n-1, and k varies from 2..k-1
I was thinking process head, DP[i][k] = min(sum(head), DP[i+1][k-1])
however, this breaks i denotation, from 0 to i.
the key is, chop off last partition from the end, so DP[i][k] keep i idx meaningful.
use cum[j] - cum[i] to calculate sum[i..j]

    const int MAX_N = 100;
    int findMax(int A[], int n, int k) {
      int M[MAX_N+1][MAX_N+1] = {0};
      int cum[MAX_N+1] = {0};
      for (int i = 1; i <= n; i++)
        cum[i] = cum[i-1] + A[i-1];

      for (int i = 1; i <= n; i++)
        M[i][1] = cum[i];
      for (int i = 1; i <= k; i++)
        M[1][i] = A[0];

      for (int i = 2; i <= k; i++) {
        for (int j = 2; j <= n; j++) {
            int best = INT_MAX;
            for (int p = 1; p <= j; p++) {
                best = min(best, max(M[p][i-1], cum[j]-cum[p]));
            }
            M[j][i] = best;
        }
      }
      return M[n][k];
    }
"""

"""
Impl regular expression matching with support of . and *
example (as, ab*s), (abbbs, ab*bs), (abss, ab.*), (abc, abb*c), (abbc, abb*c)
i,j idx moving is too complicated. You have to do recursion.
come up with solution to subproblem, and re-use the solution recursively.
"""
def isMatch(s, p):  # s is src string and p is pattern string
    if p is None:   # completed all patterns, s must be None
        return s is None

    pcur = p[0]
    pnext = p[1]
    hd = s[0]

    if pnext != '*' and pcur != hd and pcur != '.':
        return False

    if pnext != '*' and (pcur == '.' or pcur == hd):
        return isMatch(s[1:], p[1:])

    # now pnext is *, we need to consider both
    # 1. * mean 0, so if pcur != header, 0 will cancel pcur.
    if isMatch(s, p[2:]):
        return True

    # 2. * mean 1, or greedy
    if pcur == hd or pcur == '.' and s is not None:
        return isMatch(s[1:], p)

    return False   # if all above does not match


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def main():
    maxInt()
    numofones(10)
    sum(15)

    smallestConcat([3, 32, 325, 31, 322, 33, 199])

    maxSum([31, -41, 59, 26, -53, 58, 97, -93, -23, 84])

    testQuickSum()

    testCoinChange()

if __name__ == '__main__':
    main()
