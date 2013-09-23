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
    half = len(table)/2

    # find a prime, cancel all its odd multiples.
    i = 0   # array with odds num starts with 3, val=3+2i, dist(i,i-1)=2,index=(numval-3)/2 
    while i < root: 
        if table[i]:  # got a prime, cancel all its odd multiples.
            val = table[i]  # the val of this prime, its index in this array=(val-3)/2
            # every one clear out its square half. For big num, its first half already cleared.
            multiples_idx = (val*val-3)/2  
            # each idx dist is 2, so each step is 2*val, val itself is odd, so only odd multiples
            step = val   
            while multiples_idx <= len(table)-1:
                #print "current idx %d, val %d, step %d, multiples_idx %d multiple_val %d" %(i,val,step, multiples_idx, table[multiples_idx])
                table[multiples_idx] = 0
                multiples_idx += step  # each step is 2*val, odd multiples.
        i += 1

    #(x for x in table if x).next()
    yield 2
    for x in table:
        if x:
            yield x

def testPrime():
    l2 = []
    for x in myprimes(100):
        l2.append(x)
        print '->:', x
    print len(l2)

def findUgly(nth):
    primefactors = [2,3,5]
    result = [1]
    minugly = result[0]
    idx = 1

    idx_closest_twosmultiple = 0
    idx_closest_threesmultiple = 0
    idx_closest_fivesmultiple = 0

    # recursively, assume result already sorted.
    while idx < nth:
        minugly = min(2*result[idx_closest_twosmultiple], 3*result[idx_closest_threesmultiple], 5*result[idx_closest_fivesmultiple])

        result.append(minugly)
        idx += 1

        while result[idx_closest_twosmultiple] * 2 <= minugly:
            idx_closest_twosmultiple += 1
        while result[idx_closest_threesmultiple] * 3 <= minugly:
            idx_closest_threesmultiple += 1
        while result[idx_closest_fivesmultiple] * 5 <= minugly:
            idx_closest_fivesmultiple += 1

    print result

"""
 test: ncomb([i for i in xrange(1,10)], 2)
"""
def ncomb(l, width):
    if len(l) < width:
        return
    print l

    end = len(l)
    i=0
    while i <= end-width:
        head = l[i]
        next = i+1
        while next <= end-width+1:
            yield [head]+l[next:next+width-1]
            next += 1
        i += 1

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


def happynum(n):
    cycle = set()
    round = 0
    while n != 1 and round < 100:
        if n in cycle:
            print 'cycle :', n
            return False

        cycle.add(n)
        round += 1
        strn = str(n)
        sum = 0
        for i in xrange(len(strn)):
            sum += pow(int(strn[i]),2)
        n = sum
    print n

"""
home prime is concatenating the digits of the prime factors
repeating until the result is prime
"""
def homeprime(n):
    strval = ""
    for pf in factor(n):
        l.append(pf)
        strval += str(pf)
    print strval

    while not isPrime(int(strval)) :
        del l[:]
        for pf in factor(int(strval)):
            l.append(pf)
            strval += str(pf)
        print strval

def GCD(a,b):
    while b > 0:  # stop when smaller one is 0, means fully divided.
        (a,b) = (b, a%b)  # b always greater than a%b

    print "GCD(%d %d) = %d" %(a, b, a)
    return b

def lcm(a, b): return a * b / GCD(a, b)

def coprime(a, b):
    if GCD(a,b) == 1:
        return True
    return False

# Euler Phi totient: total 
def phi(n):
    r = 1
    for p,e in factorize(n):
        print p, e
        r *= (p - 1) * (p ** (e - 1))
    print r

def gcd_extension(a, b):
    if a % b == 0:
        return (0, 1)
    else:
        x, y = gcd_extension(b, a%b)
        print "gcd_extended x %d, y %d" %(y, x-(y*(a/b)))
        return (y, x-(y*(a/b)))

if __name__ == '__main__':
    testPrime()
    findUgly(20)
    testNFactorful()
    GCD(252,105)
    gcd_extension(120, 23)
    phi(61*53)
    happynum(17)

