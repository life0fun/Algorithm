#!/usr/bin/env python

import sys
import bisect
import copy

"""
Dropbox diet challenge.

A subset sum problem solved by dynamic programming.
tab[i,j][0] = 1 means exist _some_ subsets in l[0...i] that sums to val j. Not necessary from 0. could be a middle chunk as we ret when DP to l[x] = DP(val);
I indexes into list of items, j index into value space.
I varies of item idx[0..len(item)], J varies in value space[0..max]

DP_tab[objidx][val] can be a list of acts that satisfies the val, or just a flag and
store the list of acts in sol dict with compound key.

for solution dict, make compound key: objidx:val, 4:509, map val to be a list of acts of solution
sol['objidx:totval'] = [('riding-scooter', 42), ('coding-six-hours', 467)]

!!! Focus on results, not steps !!!
break the big space into smaller space with enum all the subset vals and recursion at each enum.
two list, pos list and neg list,
Enumerate each val upto max, if both lists exist subset sum to value, bam!
Extra map with key=offset+val, val=[].append(activities)

    Haijin Yan, May, 2011
"""

'''I confess, I should use closure to wrap global vars, Ugly. I'll do it in production'''
allActs = {}   # all activities dict.  [x] = 100
posActs = []   # all pos activities [(x, 100), (y, 200)]
negActs = []


def log(*k):
    print k
    #pass

def getActivity(file):
    f = open(file)
    events = (line.split() for line in f)
    for e in events:
        if len(e) > 1:  # skip the first line abt total actions
            insertActions(e[0].strip(), int(e[1].strip()))

def insertActions(action, cal):
    global allActs, posActs, negActs
    allActs[action] = cal
    if cal > 0:
        posActs.append((action, cal))
    if cal < 0:
        negActs.append((action, -cal))

"""
DP recursion down, table down when subset for the val if found.
return bool that exist some subset, and the list of exact subset of activities
l is the activity cal array sorted by value, curobjidx is the index of cur item and val is enum val.
[(x, 100), (y, 200)] => l[i][0] = activity, l[i][1] = cal

DP args carry global state. tab and sol are global state that got updated at each recursion when solution found.
In clojure, culmulate partial sol into global state and carry the cumulative global state at each recursion.
"""
def DP(l, curobjidx, val, tab, sol):
    # exit condition first, curobjidx < 0
    if curobjidx < 0 or val <= 0:
        return False, None
    if curobjidx >= len(l):  # guard the offset
        curobjidx = len(l)-1

    key = str(curobjidx)+':'+str(val)
    #log('DP: curobjidx:', curobjidx, ' Val:', val)

    ''' First, DP table check before any effort '''
    if tab[curobjidx][val] == 1:  # found in the tab
        log('Tab_found:', key, sol[key])
        return True, sol[key]

    ''' direct shot to leaf case when item val match. e.g. [1 2 3], looking for sum to 5
        the case for subset not starting from 0.  when we DP to 2, we got subset [2 3]'''
    if l[curobjidx][1] == val: # l=[(x,10),(y,20)] direct shot, head match
        tab[curobjidx][val] = 1
        if sol.has_key(key): # should never has prev val as l.curobjidx==val
            sol[key].append(l[curobjidx])
            log('match: leaf: append:', key, l[curobjidx], sol[key])
        else:
            sol[key] = []
            sol[key].append(l[curobjidx]) # append cur obj
            log('match: leaf: new:', key, l[curobjidx], sol[key])
        return True, sol[key]  # find one match, can return directly

    '''break down val, recursive DP branch, include curobjidx, not include curobjidx'''
    if val > l[curobjidx][1]:
        found, setsl = DP(l, curobjidx-1, val-l[curobjidx][1], tab, sol)
        if found:
            tab[curobjidx][val] = 1
            sol[key] = []
            sol[key].extend(setsl)
            sol[key].append(l[curobjidx])  # include current node
            log('match: key:', key, l[curobjidx], sol[key])
            return True, sol[key]  # stop when found

    ''' not include current node, recursion down prev node '''
    found, setsl = DP(l, curobjidx-1, val, tab, sol)
    if found:
        tab[curobjidx][val] = 1
        sol[key] = []
        sol[key].extend(setsl)
        log('match: key:', key, l[curobjidx], sol[key])
        return True, sol[key]

    log('Not found any subset end at:', curobjidx, ' that sum to:', val)
    return False, None

''' DP over the longer list'''
def solve():
    global posActs, negActs

    posActs = sorted(posActs, key=lambda x:x[1])
    posVals = [ k[1] for k in posActs ]
    negActs = sorted(negActs, key=lambda x:x[1])
    negVals = [ k[1] for k in negActs ]

    posmaxval = reduce(lambda x,y:x+y, posVals)
    negmaxval = reduce(lambda x,y:x+y, negVals)
    maxval = posmaxval if posmaxval > negmaxval else negmaxval

    # tab[i][j]=1 means exist some subset at index i sum to val j
    postab = [[0]*maxval for i in xrange(len(posActs))]
    negtab = [[0]*maxval for i in xrange(len(negActs))]

    possol = {}  # possol[key=(i:j)] = [ x, y] the _some_ subset that tab[i][j] refers to.
    negsol = {}
    allsols = []

    for v in xrange(posVals[0], maxval, 1):  # start from min value, enumerate each value up.
        posobjidx = bisect.bisect_right(posVals, v)  # top-down to list head, 0 <- tail
        posfound, posset = DP(posActs, posobjidx, v, postab, possol)

        if posfound:  #only bother to do neg lookup if pos found
            negoff = bisect.bisect_right(negVals, v)
            negfound, negset = DP(negActs, negoff, v, negtab, negsol)
            if negfound:
                log('sol found:', ' sum:', v, ' PosSet:', posset, ' NegSet:', negset)
                onesol = []
                onesol.extend(posset)
                onesol.extend(negset)
                allsols.append(onesol)
                break

    log(' ====== Final =====')
    if len(allsols):
        for e in allsols:
            e.sort()
            for a in e:
                print a[0]
    else:
        print 'no solution'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: drpbx_diet.py testcases'
        sys.exit(1)

    getActivity(sys.argv[1])
    solve()
