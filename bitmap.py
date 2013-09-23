#!/usr/bin/env python

import sys

"""
 formular; map[n/8] |= 1<<n%8
 numbers are integers, start from 0, so div 8 gives byte offset, mod 8 = [0-7] gives bit idx within the byte.
 0=[byte 0, idx 0], 7=[byte 0, idx 7], 8=[byte 1, idx 0]
"""
   
int2bin = lambda n: n>0 and int2bin(n>>1)+str(n&1) or ''
   
class BitMap(object):
    def __init__(self, size):
        super(BitMap, self).__init__()  # super(me, self), avoid ref base class explicitly. super().__init__().
        self.bitmap = bytearray(size)   # bitmap.decode('utf-8')
    
    def intTobin(self, n, width=8):
        return ''.join([str((n>>i) & 1) for i in range(width-1, -1, -1)])
        
    def dump(self):
        #val =  ''.join([ int2bin(i) for i in self.bitmap ])
        #return ''.join([ int2bin(i) for i in self.bitmap ])[::-1]
        return ''.join([ self.intTobin(i) for i in self.bitmap[::-1] ])

    def hexToByte(self, hexstr):
        bytes = []
        hexstr = ''.join( hexstr.split(" "))
        for i in xrange(0, len(hexstr), 2):
            bytes.append( chr(int(hexstr[i:i+2], 16)))
        return ''.join( bytes)
        
    ''' val are integers, starts from 0, so div 8 gives byte offset, mod 8 give bit idx '''
    def setBit(self, val):
        byteoff = val>>3
        idx = val%8   # one byte has 8 bits
        self.bitmap[byteoff] |= (1<<idx)
        #print val, ' ' , bin(self.bitmap[byteoff])
        print val, ' ' , self.dump()
    
    def getBit(self, val):
        byteoff = val>>3
        idx = val%8
        print self.bitmap[byteoff]
    
    def clrBit(self, val):
        byteoff = val>>3
        idx = val%8
        self.bitmap[byteoff] &= ~(1<<idx)
        print self.bitmap[byteoff]
    
    def toggle(self, val):
        byteoff = val>>3
        idx = val%8
        self.bitmap[byteoff] ^= (1<<idx)
    
    def test(self):
        self.setBit(1)  # offset val start from 1
        self.setBit(10)  # offset val start from 1
        self.setBit(7)
        self.setBit(8)
    
"""
this class define a bitmap with each bit spans w bits.
this can be used for counting bloom filter where add incr the count and remove dec the count.
"""
class BitMapWBit(BitMap):
    def __init__(self, size, w):
        self.width = w
        self.bytek = 8/self.width
        self.mask = pow(2, self.width)-1
        super(BitMapWBit, self).__init__(size)

    def setBit(self, n):
        byteoff = n / self.bytek
        idx = n % ( 8 / self.bytek)
        self.bitmap[byteoff] |= (1 << (idx * self.width))
        print ' offset :',  byteoff, idx, idx*self.width
        print n, ' ' , self.dump()

    def getBit(self, n):
        byteoff = n / self.bytek
        idx = n % ( 8 / self.bytek)
        # get the bits in the idx range; the real int val is after shift.
        val = (self.bitmap[byteoff] & (self.mask << idx*self.width )) >> (idx*self.width)
        print n, val, ' ', self.bitmap[byteoff], self.mask, idx, self.mask << (idx*self.width)
        
    def test(self):
        print ' ------ bitmap w test ------'
        self.setBit(1)
        self.setBit(5)
        self.getBit(5)
        #super(BitMapWBit, self).test()

if __name__ == '__main__':
    bm = BitMap(10)
    bm.test()
    bmw = BitMapWBit(10, 2)
    bmw.test()
