#!/usr/bin/env python

import os,sys
import operator

class PackBox():

    def __init__(self):
        self.boxlist = []
        self.area = 0
        self.nboxes = 0
        self.packx = 0
        self.packy = 0
        
    def getBoxes(self, file):
        # generator expression with ( for e in list )
        log = open(file)
        entrylist = (line.split() for line in log)
        for xy in entrylist:
            if len(xy) > 1:  # more than one field
                self.insertSort((int(xy[0].strip()), int(xy[1].strip())))

        self.boxlist = sorted(self.boxlist, key=operator.itemgetter(0), reverse=True)

        print '---- boxes list ----'
        for t in self.boxlist:
            print t
        print '-- total area:', self.area

    def insertSort(self, xy):
        maxxy = int(max(xy))
        minxy = int(min(xy))
        t = tuple((maxxy, minxy))

        self.boxlist.append(t)
        self.area += maxxy*minxy

    def pack(self):

        largestbox = self.boxlist[-1]


    def test(self):
        self.boxlist.append((6,4))
        self.boxlist.append((6,5))
        self.boxlist.append((8,1))
        self.boxlist.append((9,1))

        self.pack()

        self.boxlist.append((2,2))
        self.boxlist.append((7,1))
        self.boxlist.append((7,4))
        self.boxlist.append((6,5))

        self.pack()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: packbox boxfile'
        sys.exit(1)

    pb = PackBox()
    pb.getBoxes(sys.argv[1])





