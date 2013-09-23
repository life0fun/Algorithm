#!/usr/bin/env python

""" 
suffix array demo
""" 
import sys

class SuffixArray:
    def __init__(self, text):
        self.text = text
        self.suffixarray = []

    def getSuffixArray(self, text=None):
        if text:
            self.text = text

        for i in xrange(len(self.text)):
            self.suffixarray.append(self.text[i:])

        self.suffixarray.sort()
        print self.suffixarray

    # the longest common len between adj pairs
    # issi is the longest for mississipi
    def longestDupSubstring(self, text=None):
        del self.suffixarray[:]
        self.getSuffixArray(text)

        longestcommonlen = 0
        longestdupsubstridx = 0

        for i in xrange(len(self.suffixarray)-1):
            a = self.suffixarray[i]
            b = self.suffixarray[i+1]
            for k in xrange(min(len(a),len(b))):
                if a[k] != b[k]:
                    break
            if k > longestcommonlen:
                print a, b
                longestcommonlen = k
                longestdupsubstridx = i

        print 'longestDupSubstring:', self.suffixarray[longestdupsubstridx][:longestcommonlen]

if __name__ == '__main__':
    sa = SuffixArray("mississipi")
    sa.getSuffixArray()
    sa.longestDupSubstring()
