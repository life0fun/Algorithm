#!/usr/bin/env python
"""
This is NP-hard. http://algowiki.net/wiki/index.php?title=Edmonds's_algorithm
1. remove all edges going into the root node
2. For each node, select only the incoming edge with smallest weight
2. Now the edge set contains n edges, each points one node.
3. If there are cycles, the edge set will be serveral small disjoint sets.
3. For each circuit that is formed:
    edge "m" is the edge in this cycle with minimum weight
    Combine all the circuit's nodes into one pseudo-node "k"
    For all edges each edge "e" entering a node in "k" in the original graph:
        edge "n" is the edge currently entering this node in the circuit
        when cycle formed, each node can only have edge in.
        track the minimum modified edge weight between each "e" based on the following:
            modWeight = weight("e") - ( weight("n") - weight("m") )
            On edge "e" with minimum modified weight, add edge "e" and remove edge "n"

"""

import os,sys
from decimal import *
import copy
import operator
import pdb

class FaceBull():
    def __init__(self):
        self.machines = {}
        self.compounds = set()
        self.distMatrix = {}
        self.dijkstraMatrix = {}
        self.startpoint = None
        
    def getMachines(self, file):
        logf = open(file)
        locs = (line.split() for line in logf)  #generator
        for l in locs:
            machine, csrc, cdst, cost = l
            self.machines[machine] = (csrc, cdst, cost)
            self.compounds.add(csrc)
            self.compounds.add(cdst)

        print '-----------' 
        for k,v in self.machines.items():
            print k, v
        print self.compounds

    # calculate the shortest path between each pair of locations
    def dijkstra(self, start, dist={}, parent={}):
        q = self.distMatrix.keys()
        for k in q:
            dist[k] = 1e3
            parent[k] = None
        dist[start] = 0
        parent[start] = start

        tmpd = copy.deepcopy(dist)
        while q:
            node,cost = sorted(tmpd.iteritems(), key=operator.itemgetter(1))[0]
            if cost == 1e3:
                break

            q.remove(node)
            tmpd[node] = 1e3

            for k,v in self.distMatrix.get(node).items():
                if dist[k] > dist[node] + v:
                    tmpd[k] = dist[k] = dist[node] + v
                    parent[k] = node  # key is cur node, value is parent

        #print 'source', start, ' cost:', dist
        #print 'source', start, ' path:', parent

    def allDijkstra(self):
        for node in self.locProb:
            path = {}
            cost = {}
            self.dijkstra(node, cost, path)
            self.dijkstraMatrix[node] = dict(path=path, cost=cost) 

        for k,v in self.dijkstraMatrix.items():
            print '-->', k, v

    def expectedValue(self):
        # sort prob
        l = sorted(self.locProb.iteritems(), key=operator.itemgetter(1), reverse=True)
        time = 0
        print '=========== startingpoint:', self.startpoint
        start = self.startpoint

        while l:
            place, prob = l.pop(0)
            path = self.dijkstraMatrix[start]['path']
            cost = self.dijkstraMatrix[start]['cost']
            print 'next stop==>', place, ':', prob
            print 'path:', path
            print 'cost:', cost
            time += prob * cost[place]  # time converted to Decimal
            print ' next stop cost:', cost[place], ' and time:', time
            start = place

        print time.quantize(Decimal('.01'))
            

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: fb_bull.py cf'
        sys.exit(1)

    fb = FaceBull()
    fb.getMachines(sys.argv[1])
