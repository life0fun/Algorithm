#!/usr/bin/env python

import sys
import pdb

"""
prime number generator
"""
def factor(n):
    yield 1

    root = n**0.5
    i = 2
    while i<root:
        if n%i == 0:
            yield i
            n = n/i
            root = n**0.5
        else:
            i += 1
    if n > 1:
        yield n

def factorize(n):
    for prime in myprimes(n):
        if prime > n: return

        exponent = 0
        while n % prime == 0:
            exponent, n = exponent + 1, n / prime
        if exponent != 0:
            yield prime, exponent

"""
generate prime num up to n
"""
def myprimes(n):
    if n==2: yield 2
    elif n<2: yield

    #[3,5,7,9,11,13,15,17,19,21,23,25], val=3+2i, i=(val-3)/2, i-(i-1)=2
    table = range(3,n,2)  # odd num array start with 3
    root = n**0.5

    yield 2

    # find a prime, cancel all its odd multiples.
    i = 0   # array with odds num starts with 3, val=3+2i, dist(i,i-1)=2,index=(numval-3)/2 
    while i < root: 
        if table[i]:  # got a prime, cancel all its odd multiples.
            val = table[i]  # the val of this prime, its index in this array=(val-3)/2
            # every one clear out its square half. For big num, its first half already cleared.
            multiples_idx = (val*val-3)/2  
            # each idx dist is 2, so each step is 2*val, val itself is odd, so only odd multiples
            wheel = val   
            while multiples_idx <= len(table)-1:
                #print "current idx %d, val %d, step %d, multiples_idx %d multiple_val %d" %(i,val,step, multiples_idx, table[multiples_idx])
                table[multiples_idx] = 0
                multiples_idx += wheel  # each step is 2*val, odd multiples.
        i += 1

    #(x for x in table if x).next()
    for x in table:
        if x:
            yield x

def testPrime():
    l2 = []
    for x in myprimes(3000):
        l2.append(x)
        print '->:', x

def fib(n):
    prefibnum = 1
    curfibnum = 2  # 0 1 1 2 3 5
    while curfibnum <= n:
        curfibnum, prefibnum = curfibnum+prefibnum, curfibnum

    while True:
        yield curfibnum
        curfibnum, prefibnum = curfibnum+prefibnum, curfibnum

def fibPrime(n):
    fibg = fib(n)
    primeg = myprimes(n*10)

    fibn = fibg.next()
    primen = primeg.next()
    while True:
        if fibn == primen:
            break;
        fibn = fibg.next() if fibn < primen else fibn
        primen = primeg.next() if fibn > primen else primen

    print 'fibprime : ', fibn

    primefactorl = []
    for primefactor, exp in factorize(fibn+1):
        #print 'factor:', primefactor
        primefactorl.append(primefactor)

    print "sum prime factor:", sum(primefactorl)


"""
 a num is called n-factorful if it has exactly n distinct prime factor
 e.g, 1 is 0-factorful, p and p^2 is 1th factor, c(2,p) is 2th factor[2*3,2*5,2*7,3*5,3*7,5*7]
 find all primes, and power set of any 2, find all 2th, powerset of any 3, find all 3th.
"""
def NFactorful(start, end, n):
    allprimes = []
    for p in myprimes(end):
        allprimes.append(p)

    print allprimes

    upper = end/n
    allprimes = filter(lambda x: x<upper, allprimes)
    print allprimes

    for l in ncomb(allprimes, n):
        product = reduce(lambda x,y:x*y, l)
        #print l, '=', product
        if product > start and  product < end:
            print 'Good =', product, ' =>' ,l
            yield l

def testNFactorful():
    for n in NFactorful(5,100,2):
        print n
    for n in NFactorful(5,100,3):
        print n
    #for n in NFactorful(5,100,4):
    #    print n


if __name__ == '__main__':
    fibPrime(227000) 
