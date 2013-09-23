#!/usr/bin/env python

"""
find least positive number returned by n digits with +-*/ expressions.
idea:
    1. First, I thought I could do dynamic programming, with table[n][v] = min( table[n-k][v-k] ).
       However, I could not find obvious relationship between n and n-1.
    2. Then, After some research, It looks you have to do brute force iterate of all combinations at each level.
       So the solution is just for sub problem at depth d, iterate all the depth from 1 to d-1, add/sub/div/multiple
       on top of all the values at that depth, with the current value. 
       This is like finding the powerset of a list.
       Brutal force to calc all the values at each depth and sort the list out to find the min.
"""

import sys
from collections import defaultdict

class Op:
    def __init__(self, val, op):
        self.val = val
        self.op = str(op)

    def __str__(self):
        return self.op

class Exp:
    def __init__(self, num, depth):
        self.num = num
        self.depth = depth
        self.exp = defaultdict(list) 
        self.resultvalues = defaultdict(set)
        self.resultvalues[1].add(num)   # key is number, not string

    def debug(self, *k, **kw):
        #print '', k
        pass

    ''' get the ops for certain val'''
    def getOp(self, val):
        formatv = '{0:.2f}'.format(val)
        ret = ''
        for op in self.exp[formatv]:
            self.debug(val, ' = ', op)
            ret += '[' + op.op + ']'
        return ret

    ''' add computed value to current depth's value '''
    def addCurDepthValue(self, curdep, val, exp):
        formatv = '{0:.2f}'.format(val)
        self.resultvalues[curdep].add(val)
        self.exp[formatv].append(exp)

    ''' for each depth from 1 to d, union all values at depth i and d-i ''' 
    def oneDepth(self, d):
        for i in xrange(2, d+1):   # starting from second level
            self.debug('depth i :', i, self.resultvalues[i-1])
            for ival in list(self.resultvalues[i-1]):
                self.debug('depth j :', self.resultvalues[d-i+1])
                for jval in list(self.resultvalues[d-i+1]):
                    self.debug('depth:' , i, ' ', ival, ' : ', jval)
                    self.addCurDepthValue(d, ival+jval, Op(ival+jval, str(ival)+'+'+str(jval)))
                    self.addCurDepthValue(d, ival-jval, Op(ival-jval, str(ival)+'-'+str(jval)))
                    self.addCurDepthValue(d, ival*jval, Op(ival*jval, str(ival)+'*'+str(jval)))
                    if jval != 0:
                        self.addCurDepthValue(d, 1.0*ival/jval, Op(ival/jval, str(ival)+'/'+str(jval)))
            
    def solve(self):
        for k in xrange(2, self.depth+1):
            self.oneDepth(k)

        ''' solution on the last level'''
        l = list(self.resultvalues[self.depth])
        l.sort()
        l = filter(lambda x: x>0, l)
        self.debug(l)
        for i in xrange(1,sys.maxint):  # find the least positive number
            if i not in l:
                return i

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usgae : ./leastpositive number count'
        sys.exit()

    exp = Exp(int(sys.argv[1]), int(sys.argv[2]))
    #exp = Exp(7, 4)
    least = exp.solve()
    print 'least positive for ', sys.argv[1], ' ', sys.argv[2], 'times is ',  least
    print 'express just before least is:' , least-1, ' ', exp.getOp(least-1)

