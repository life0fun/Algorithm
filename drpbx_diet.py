#!/usr/bin/env python

import sys
import bisect
import copy

"""
Dropbox diet challenge.
A subset sum problem solved with dynamic programming.
tab[i,j][0] = 1 means exist _some_ subsets in l[0...i] that sums to val j.
I indexes into list of items, j index into all possible values.
tab[i,j][1] = [x, y] list of acts for the solution

    Haijin Yan, May, 2011
"""

''' I confess, I should use closure to wrap the global vars, Ugly. I will do it in production code'''
allActs = {}   # all acts dict.  [x] = 100
posActs = []   # all pos activities [(x, 100), (y, 200)]
negActs = []
powersetsum = []
tab = None 
acts = {}

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
    allActs[action] = cal
    if cal > 0:
        posActs.append((action, cal))
    if cal < 0:
        negActs.append((action, -cal))

''' the list is expanding, can not do yield'''
def powerset(l, ps=None):
    if not l:
        return
    tmpl=[]  # cant operate in-place in ps, infinite loop
    for e in ps:
        x = []
        x.extend(e)
        x.append(l[0])
        tmpl.append(x)
    ps.extend(tmpl)
    ps.append([l[0]])
    powerset(l[1:], ps)
    return

''' DP recursion down, return whether subset found, and the exact subset'''
def DP(l, root, val, tab, acts):
    # exit condition first, root < 0
    if root < 0 or val <= 0:
        return False, None
    if root >= len(l):  # guard the offset
        root = len(l)-1

    key = str(root)+':'+str(val)
    log('DP: root:', root, ' Val:', val)

    ''' ask questions, empty search space, look at head, recursion on children'''
    if l[root][1] == val:
        tab[root][val] = 1
        if acts.has_key(key): # should never has prev val as l.root==val
            acts[key].append(l[root])
        else:
            acts[key] = []
            acts[key].append(l[root])
        return True, acts[key]  # find one match, can return directly

    if tab[root][val] == 1:  # found in the tab
        log('Tab found:', key, acts[key])
        return True, acts[key]

    '''recursive decisionDP, include root, not include root'''
    if val > l[root][1]:
        found, setsl = DP(l, root-1, val-l[root][1], tab, acts)
        if found:
            tab[root][val] = 1
            acts[key] = []
            acts[key].extend(setsl)
            acts[key].append(l[root])
            return True, acts[key]

    found, setsl = DP(l, root-1, val, tab, acts)
    if found:
        tab[root][val] = 1
        acts[key] = []
        acts[key].extend(setsl)
        return True, acts[key]
    
    return False, None

''' DP over the longer list'''
def solve():
    l = posActs if len(posActs) > len(negActs) else negActs
    l = sorted(l, key=lambda x:x[1])
    lkey = [k[1] for k in l]
    q = negActs if len(posActs) > len(negActs) else posActs
    q = sorted(q, key=lambda x:x[1])

    dpmaxsum = reduce(lambda x,y:x+y, lkey)
    tab = [[0]*dpmaxsum for i in xrange(len(l)+1)]
    log('sum:', dpmaxsum, ' lkey:', lkey)

    powerset(q, powersetsum)  # l is DP search space, q is value sum space
    log('len q:', len(q), ' len sum:', len(powersetsum), ' sum:', powersetsum)
    ''' forEach combination, reduce the sum, then DP check equiv sum in the other array'''
    for e in powersetsum:
        valsum = reduce(lambda x,y:x+y, [a[1] for a in e])
        if valsum > dpmaxsum:
            continue
        pos = bisect.bisect_right(lkey, valsum)
        log('lkey:', lkey, ' valsum:', valsum, ' valset:', e, ' pos:', pos)
        found, activities = DP(l, pos, valsum, tab, acts)
        if found:
            log('found:', ' sum:', valsum, ' actions:', e, ' otherActions:', activities)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: drpbx_diet.py testcases'
        sys.exit(1)

    getActivity(sys.argv[1])
    solve()
