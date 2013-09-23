#!/usr/bin/env python
import sys
import copy
import collections
import random

"""
Edmonds Karp max flow min cut algorithm for directed Graph.
http://www.topcoder.com/tc?module=Static&d1=tutorials&d2=maxFlow
http://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm

1. Each iteration by calling BFS will find a path from src to sink.
2. During BFS finding a path to sink, the path capacity is the smallest cap of all edges in a path.
3. BFS from parent to child, carry the path min. If the capacity from parent to child is bigger
   than the current path min, current node did not shrink path cap. Carry the path min from parent.
4. Otherwise, current node shrinks the path min, set the path min at the current node to be the
   Residual capacity(from parent to current) at the node.
5. When reaching sink, a path is find with path min cap. Otherwise, no path to sink.
6. MaxFlow keep finding path, cumulate path min to maxflow, reduce redsidual capacity of each edge.
"""

''' vertices are numbered from 1->n '''
def MaxFlowMinCut(S, T, G):  # graph, source, sink
    vertices = len(G)
    C = G #[[0] for i in 2*xrange(vertices)]  # capacity table C[src][dst]
    ResC = [[0]*vertices for i in xrange(vertices)]  # residual capacity table C[src][dst]
    F = [[0]*vertices for i in xrange(vertices)]  # flow table C[src][dst]
    min_capacity = [0]*vertices
    P = [0]*vertices  # Parent table
    Q = collections.deque()
    MinCut = []

    def BFS(Q, P, source, min_capacity):
        V = [0]*vertices  # visited talbe
        for i in xrange(vertices):
            P[i] = -1
            V[i] = 0
            min_capacity[i] = 1<<31  # a vertex's min cap means the cap of this vertex to sink.

        Q.clear()
        Q.append(source)  # always start from source
        V[source] = 1
        print 'source', source

        ''' first, found the augmenting-path'''
        while len(Q):
            node = Q.popleft()
            print 'pop out:', node
            for nb in xrange(vertices):   # all the node in graph, not only the nb
                if V[nb] == 0 and ResC[node][nb] > 0: # not visited, and have ResC
                    V[nb] = 1
                    P[nb] = node
                    min_capacity[nb] = min(ResC[node][nb], min_capacity[node])
                    print 'find path', node, nb, ResC[node][nb], min_capacity[nb]
                    if nb == T:
                        return True
                    Q.append(nb)
        return False # did not find any augmenting-path

    ''' recursively BFS to find each path and sum max flow '''
    ResC = copy.deepcopy(C)
    max_flow = 0
    while BFS(Q, P, S, min_capacity):  # BFS continue to find a path to sink with min capacity
        max_flow += min_capacity[T] # add capacity of the path into max flow
        where = T
        print P
        while where != S:
            prev = P[where]
            F[prev][where] += min_capacity[T]
            F[where][prev] -= min_capacity[T]
            ResC[prev][where] -= min_capacity[T]
            ResC[where][prev] += min_capacity[T]  # this one put a reverse residual capacity
            if ResC[prev][where] == 0 and C[prev][where] > 0: # must have capacity from prev to where.
                MinCut.append((prev,where)) # this is min cut edge
            where = prev
    print MinCut

def testMaxFlowMinCut():
    G = [[0]*7 for i in xrange(7)]
    G[0][1] = G[0][3] = G[2][0] = 3
    G[1][2] = 4
    G[2][3] = 1
    G[2][4] = G[3][4] = 2
    G[3][5] = 6
    G[4][1] = G[4][6] = 1
    G[5][6] = 9
    MaxFlowMinCut(0, 6, G)

"""
 Independent set: a set(vertices) that no two are adjacent. No edge connects any two.
 Maximal or Maximum indep set, MST, max flow, min cut, bipartite maximum match, planar color
"""
def MaximalIndependentSet(G):
    MIS = set()  # maximal independent set
    MaxMIS = []  # maximum of all maximal independent sets

    ''' get the max degree of any node in a un-directed graph'''
    def degree(G):
        maxdegnode = 0
        maxdegs = 0
        for v in G.keys():
            if len(G[v]) > maxdegs:
                maxdegnode = v
                maxdegs = len(G[v])

        return maxdegnode, maxdegs

    def SlowMIS():
        addtoset = True
        for v in G.keys():
            for nb, cost in sorted(G[v].iteritems(), key=lambda x:x[1]):
                if nb in MIS:
                    addtoset = False
                    break
            if addtoset == True:
                MIS.add(v)

    def FastMIS():
        random.seed()

        for v in G.keys():
            degree = len(G[v].keys())
            prob = 1.0/2*degree
            if random.Random() <= prob:
                MIS.add(v)
                for nb, cost in sorted(G[v].iteritems(), key=lambda x:x[1]):
                    if nb in MIS:
                        if len(G[nb].keys()) > degree:
                            MIS.remove(v)
                        elif len(G[nb].keys()) == degree:
                            if degree % 2 :
                                MIS.remove(nb)
                            else:
                                MIS.remove(v)

            '''clean up set'''
            for v in MIS:
                for nb in G[v]:
                    del G[nb]
                del G[v]

    '''http://www.eurojournals.com/ejsr_23_4_09.pdf'''
    def recursiveMIS(G):
        def removeNode(G, nodes):
            for v in G:
                if v in nodes:
                    del d[v]
                    continue
                for node,cost in d[v].items():
                    if node in nodes:
                        del d[v][node]

        node, degs = degree(G)
        if degs == 0:
            return len(G.keys())

        G1 = copy.deepcopy(G)
        removeNode(G1, [node])
        G2 = copy.deepcopy(G1)
        nb = G2[node].keys()
        removeNode(G2, nb)
        return max(recursiveMIS(G1), recursiveMIS(G2))

if __name__=='__main__':
    testMaxFlowMinCut()
