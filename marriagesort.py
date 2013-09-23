#!/usr/bin/env python

import sys

"""
marriage sort, working set = [0:sqrt(n)]
"""
def marriageSort(l):
    n = len(l)
    workingsetidx = int(n**0.5)
    workingset = l[0:workingsetidx]
    workingset.sort()
    workingset.extend(l[workingsetidx:])
    l = workingset

    end = n-1
    workingsetidx -= 1
    while workingsetidx >=0 :
        print 'starting idx:end:', workingsetidx, end, l
        i = workingsetidx+1
        while i < end:
            if l[i] >= l[workingsetidx]:
                l[end], l[i] = l[i], l[end]
                print 'one swap i:end:', i, end,l
                end -= 1
            i += 1
        l[workingsetidx], l[end] = l[end], l[workingsetidx]
        workingsetidx -= 1
        end -= 1
        
        print 'one pass done:', l

def testMarriage():
    l = [8,5,7,10,3,2,9,4,6,1]
    marriageSort(l)

if __name__ == '__main__':
    testMarriage()

