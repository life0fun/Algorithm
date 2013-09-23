#! /usr/bin/python

"""
Bloom filter: a bitmap, a hash func, and a input method.
Haijin Yan, Feb 2011
"""

import os,sys
import array as arr
import hashlib

class BloomFilter:
    def __init__(self, hashobj=None):
        self.hashobj=hashobj
        self.bitmap = arr.array('B', [0]*1024)

    def showByte(self, n):
        print ''.join(str((n>>i)&1) for i in xrange(7,-1,-1))
    def setNthBit(self, n):  # 1 byte, 8 bits
        self.bitmap[n>>3] |= 1<<(n%8)
        self.showByte(self.bitmap[n>>3])
    def clearNthBit(self, n):
        self.bitmap[n>>3] &= ~(1<<(n%8))
        self.showByte(self.bitmap[n>>3])
    def getNthBit(self, n):
        self.showByte(self.bitmap[n>>3])
        return self.bitmap[n>>3] & 1<<(n%8)
    def isNthBitSet(self, n):
        self.showByte(self.bitmap[n>>3])
        return True if self.bitmap[n>>3] & 1<<(n%8) > 0 else False

    def testBitmap(self):
        bf.setNthBit(10)
        bf.clearNthBit(11)
        print bf.isNthBitSet(10)

    def getKeyDigest(self,key):
        self.hashobj = hashlib.md5()
        self.hashobj.update(key)
        return self.hashobj.hexdigest()

    def setMembership(self, key):
        digest = self.getKeyDigest(key)
        for i in xrange(0,len(digest), 4):
            numval = int(digest[i:i+4], 16)
            self.setNthBit(numval%(1024*8))

    def testMembership(self, key):
        digest = self.getKeyDigest(key)
        for i in xrange(0,len(digest), 4):
            numval = int(digest[i:i+4], 16)
            if not self.isNthBitSet(numval%(1024*8)):
                print "No...key %s not set!" %(key)
                return False

        print "Yes...key %s set!" %(key)
        return True

if __name__ == "__main__":
    bf = BloomFilter()
    bf.testBitmap()
    bf.setMembership("bloom")
    bf.setMembership("filter")

    bf.testMembership("bloom")
    bf.testMembership("bloom ")
