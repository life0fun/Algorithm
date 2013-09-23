#!/usr/bin/env python

import os,sys
import operator
from bisect import bisect
import pdb

"""
DropBox Diet puzzle
Find items from an array that sum is zero.

The algorithm first compute the power set of all actions, and 
matching the positive items to negative items
"""

class Diet():

    def __init__(self):
        self.posActions = {}
        self.negActions = {}
        self.posSum = {}
        self.negSum = {}
        self.solution = []
        
    def getActivity(self, file):
        # generator expression with ( for e in list )
        log = open(file)
        events = (line.split() for line in log)
        for e in events:
            if len(e) > 1:  # more than one field
                self.insertActions(e[0].strip(), int(e[1].strip()))

        # sort the actions
        #self.sorted_positives = sorted(self.posActions.items(), key=lambda x: x[1])

        print '---- begin activities---'
        for e in self.posActions.items():
            print e
        print '---- +++ --------'
        for e in self.negActions.items():
            print e
        print '---- end activities ----'

    def insertActions(self, action, cal):
        if cal > 0:
            self.posActions[action] = cal
        else:
            self.negActions[action] = cal

    def powerSet(self, actions={}, summation={}):
        l = []
        extl = []
        s = 0
        for k,v in actions.items():
            for subl in l:
                newsubl = subl[:]
                newsubl.append(k)
                extl.append(newsubl)
            extl.append([k])
            for e in extl:
                l.append(e)
            del extl[:]

        for e in l:
            s = 0
            s += sum([actions[x] for x in e])
            summation[s] = e

    def findActionsSumTo(self, sum):
        if bisect(self.sorted_positives, sum):
            print 'finding sum %d with index %d' % (sum,sorted_positives)

    def reduce(self):
        # first, prepare data
        self.powerSet(self.posActions, self.posSum)
        self.powerSet(self.negActions, self.negSum)

        # loop thru all summation keys
        for pk, pv in self.posSum.items():
            try:
                nk = -pk
                if self.negSum[nk]:
                    nv = self.negSum[nk]
                    self.solution.append(pv+nv)
            except KeyError:
                pass

        for e in self.solution:
            print e


    def reduceBlock(self, block):
        pass

    def test(self):
        self.powerSet(self.posActions, self.posSum)
        self.powerSet(self.negActions, self.negSum)

        print '========='
        for e in self.posSum.items():
            print e
        for e in self.negSum.items():
            print e

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: drpbx_diet.py diet.log'
        #sys.exit(1)

    diet = Diet()
    diet.getActivity('diet.log')
    #diet.test()
    diet.reduce()
