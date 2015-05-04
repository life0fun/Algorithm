#!/usr/bin/env python

import sys
import pdb

"""
c/java can not return a(set, sequence, collection) to be wrapped by generator
this recursion actually is a DFS(){for each son e : DFS()}

public static String permutation(String s, int pos){
    if(pos == s.length()){ System.out.println("===" + s); }
    char[] ca = s.toCharArray();
    char thisc = ca[pos];
    for(int i=pos;i<s.length();i++){
        char tmpc = ca[i]; ca[pos]=tmpc; ca[i]=thisc;  // swap
        permutation(new String(ca), pos+1);  // happy recursion with new data
        ca[i]=tmpc; ca[pos]=thisc;   // swap back
    }

public static String powerset(String s, int pos, List<String> l){
    if(l == null){
        l = new ArrayList<String>();
        l.append('empty');
    }
    l.append(s[pos]);

    for(String e : l){
        l.append(e+s[pos]);
    }
    powerset(s, pos+1, l);
}
"""

"""
powerset('abc') pass in a string, returns a set of[0, a,b,c, ab,ac, bc, abc]
"""
def powerset(arr, l=None):
    if not arr:
        for e in l:
            print e
        return

    if not l:
        l = []
        l.append('0')

    tmpl = []
    for e in l:
        tmpl.append(e+arr[0]) # concat head of arr to each ele in existing l
    l.extend(tmpl)

    powerset(arr[1:], l)  # happy recursion
    return

"""
return a list, caller for in list to iterate the sequence, list, collection
"""
def permustr(string):
    if not string:
        #yield ""
        return ""

    c = string[0]   # get header, and recursion from header.
    l = []
    subp = permustr(string[1:])  # recursion of the same func on a smaller problem subset.
    #print string
    #print subp

    if not subp:
        l.append(c)
    else:
        for n in subp:
            print "returned..." + n
            for i in range(0, len(n)+1):
                #yield n[len(n)-i] + c + n[i:]
                print "injection:",  i , "==" + n[:i] + ":" +  c + ":" + n[i:]
                l.append(n[:i] + c + n[i:])

    if len(string) == 3 :
        for x in l:
            print "::" + x
        #yield x
    return l

"""
we can also remove each ele in the arg, recursive on the rest, and append the removed ele to the sub result
"""
def permutation(l):
    if len(l) == 1:     # stop on last ele
        return [ l[0] ]

    res = []
    for i in xrange(0, len(l)):
        subl = l[:i] + l[i+1:]   # remove l[i],
        subp = permutation(subl) # perm on subp
        for k in subp:
            k.append(l[i])
        res += subp
    return res

"""
using generator yeild to save the need for []
because the function is a generator, any recursive call to it
need to yield continuously.
"""
def generatorpermustr(string):
    if not string:
        print "yield empty string"
        yield ""
        return

    c = string[0]
    for k in generatorpermustr(string[1:]):
        if not k:
            yield c
        else:
            for i in xrange(len(k)+1):
                yield (k[:i] + c + k[i:])

''' perm list : ret a list of all perm in sublist
    pass in a list : [1,2,3,4] , return [ [1,2,3], [1,3,2], ... ]
'''
def permList(l):
    ret = []
    if len(l) == 1:
        ret.append(l)
        return ret

    hd = l[0]
    for e in permList(l[1:]):
        for pos in xrange(len(e)+1):   # [0...len), half close, remeber to add 1
            ret.append(e[:pos] + [hd] + e[pos:])
    return ret

def perm(l):
    result = []
    if len(l) == 1:
        return [].append(l[0])
    
    for hd in l:
        for r in perm(remove(l, hd)):
            result.append(hd + r)
    return result

'''
Distinct subsets for an array with duplicated elements
Arr = {1,2,2,3,3,4,5} 
Arr_distinct = {1,4,5}
First, gen powset of Arr_distinct, then for each dup item,
add {2} and {2,2} to the result, and combine it. The same for {3}, {3,3}
https://csjobinterview.wordpress.com/2012/10/04/distinct-subsets-for-an-array-with-duplicated-elements/
'''
def subsetWithDup(l):
    distinctL = removeDup(l)
    dupSets = []
    for d in dups(l):
        for cnt in xrange(dups(d)):
            dupSets.append(d*cnt)

    result = powset(distinctL)
    for d in dupSets:
        for r in results:
            result.append(r.append(d))

    return result



''' give a string of n items, find all the subset of k items
    generator version and no generator version
    two cases: 1. incl e, recur of k-1; not incl e, recur of k;
'''
def subsetKofNGen(l, k):
    #print 'entering subsetk:', l, k
    n = len(l)

    if k == 0:  # end of recursion, len k
        yield set()
    else:       # as k=0 stoped, wont recursion on k<0,  no worry of n<k case
        for i in xrange(n):
            e = l[i]
            for s in subsetKofNGen(l[i+1:], k-1):
                #print 'recursion:', s, e
                s.add(e)
                yield(s)

'''
1. Top down: recursive down to leaf and got the result. However, to collect the result,
need to pass the path pref from each level in tmpSolution down to the leaf.

2. Bottom up: recursive extend and return the sub result at each level. Collect full result at root.
'''
def subsetKofNTopDown(l, k, curset, resultl):
    n = len(l)
    if k == 0:  # top down, the end of recursion
        resultl.append(curset)
        return
    else:
        for i in xrange(n):
            e = l[i]
            tmpset = set(curset)   # a deep copy of tmp solution before branch out.
            tmpset.add(e)
            subsetKofNTopDown(l[i+1:], k-1, tmpset, resultl)

''' subset kN DP top down carry temp sol at each iteration, no return val
    first, how to divide to subp, second, handle leaf condition k=1/0 properly
'''
def subsetKNDp(l, n, k, cursol, gsol):
    if n == k:
        for e in l:
            cursol.append(e)
            gsol.append(cursol)
            return
    if k == 1:
        for e in l:
            gsol.append(cursol+[e])
            return

    hd = l[0]
    temp = list(cursol)
    temp.append(hd)
    subsetKNDp(l[1:], n-1, k-1, temp, gsol)
    subsetKNDp(l[1:], n-1, k, cursol, gsol)

''' leaf condition first, sub result at leaf, ret each sub result, and form the full result at root
    look at head, incl head, not incl head.
    The trick here is at the boundary when k=0, down recursion rets empty set, we forget to ret hd at this recursion.
'''
def subsetKNBottomUp(l, k):
    ret = []  # stack var, diff at each recursion level

    if k == len(l):
        ret.append(l)
        return ret

    ''' k=1, the rest of element of the list auto qualify, put them into list with single ele'''
    if k == 1:
        for e in l:
            ret.append([e])
        return ret

    head = l[0]
    ''' include head '''
    for e in subsetKNBottomUp(l[1:], k-1):
        e.append(head)
        ret.append(e)

    ''' not include head '''
    for e in subsetKNBottomUp(l[1:], k):
        ret.append(e)

    return ret

''' branch and bound, only look at head, recursion branch
    the passed in l arguments is 'abcd', not a list
	for flat structure(list), just check head, and recursion on the rest of the list.
	for hierarchy structure(tree), for each node, recursion on each children of the node.
'''
def kl(l, k):
    ret = []

    if k == 1:  # end condition !!!
        return list(l) # for l as 'abcd' format only
        #for e in l:
        #    ret.append([e])
        #return ret

    if len(l) == k:
        ret.append(l)
        return ret

    for e in kl(l[1:], k-1):
        ret.append(l[0]+e)

    for e in kl(l[1:], k):
        ret.append(e)

    #print 'l=', l, ' k= ', k, ' ', ret
    return ret

def levenshteinDist(src, tgt):
    """ DP to calculate levenshteinDist iteratively, en.wiki, keep pre/cur value row and roll over.
    http://www.codeproject.com/Articles/13525/Fast-memory-efficient-Levenshtein-algorithm
    keep two consecutive rows/cols, the min of 3 left upper cells determine cur cell val."""

    if src is tgt:
        return 0

    if len(src) == 0 or len(tgt) == 0:
        return max(len(src), len(tgt))
    
    print 'levenshteinDist :', src, tgt

    # two ary of t.length store prev and cur diff vals and rolling
    preRow = [i for i in xrange(len(tgt)+1)]    # prev row inited to 1-n, mean diff is 1-n
    curRow = [0 for i in xrange(len(tgt)+1)]    # xrange is [), not include

    # we inited preRow, starting from curRow, upper-left corner is 0, so the first cell in curRow is 1
    for i in xrange(len(src)):
        curRow[0] = i + 1   # when filling curRow, the first cell starts from 1, as upper-left corner is 0
        for j in xrange(len(tgt)):
            cost = 0 if src[i] == tgt[j] else 1
            curRow[j+1] = min(preRow[j+1] + 1, curRow[j] + 1, preRow[j] + cost)
        # after one src ele, curRow to preRow, continue
        preRow = curRow[:]
        print i, ':', preRow

    #after all ele in src done comparison to tgt ary, the value in last cell is the one.
    print "Leven dist [ ", src, tgt, " ] = ", curRow[len(tgt)]
    return curRow[len(tgt)]

if __name__ == '__main__':
    print 'powerset abc'
    powerset('abc123')

    #print permutation(['abc', 'xyz', '123'])

    print 'permutation abc'
    p=permustr("abc")

    print 'permutation abc with generator'
    p=generatorpermustr('abc')
    for s in p:
        print 'yield::', s

    print '---- perm list -----'
    for e in permList([1,2,3,4]):
        print e

    print '------ subset len k -----'
    l = [1,2,3,5,6]
    for s in subsetKofNGen(l, 3):
        print 'subset k:', s

    print '------ subset KofN top carry down result set -----'
    reslt = []
    subsetKofNTopDown(l, 3, set(), reslt)
    for s in reslt:
        print 'subset k:', s

    print '\n------ subset with kl  -----'
    print kl('abcd', 2)
    print kl([1,2,3,4], 2)

    print '\n------ subsetKN ------'
    for e in subsetKNBottomUp([1,2,3,4], 2):
        print e

    print ' === levenshteinDist ====='
    levenshteinDist('GUMBO', 'GAMBOL')
    levenshteinDist('BUMBOL', 'GAMBOL')
