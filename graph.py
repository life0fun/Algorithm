#!/usr/bin/env python

"""
  Python Graph module
  Graph matrx as dict with value as a list

  Haijin Yan, Jan, 2011
"""
import sys, os
import operator
import pdb
import copy

class Graph(object):
    _INFINITY = 1e3

    def __init__(self):
        self.graph = {
            'A': ['B', 'C'],
            'B': ['C', 'D'],
            'C': ['D'],
            'D': ['C'],
            'E': ['F'],
            'F': ['C']
            }

        self.wgraph = {
            "A":{"B":3,"C":4},
            "B":{"D":2},
            "C":{"D":1,"E":1,"F":6},
            "D":{"H":1},
            "E":{"H":7},
            "F":{"G":1},
            "G":{},
            "H":{"G":3}
        }

        self.dijkstraMatrix = {}
    
    # No need to take care of many local temp path objects.
    # in c/c++, you need track those objs globally to avoid mem leak.
    # always use generator path if func return a list
    # Branch-Bound DFS with recursion.
    def findPath(self, start, end, path=[]):
        #print 'find path start with:', start
        path = path + [start]  # create a local path obj with cur node in.
        if start == end:    # found path, now we can return path
            yield path
        try:
            for nb in self.graph.get(start):
                if not self.cyclic(path, nb):
                    #newpath = self.findPath(nb, end, path)
                    #if newpath:
                    #    yield newpath
                    for p in self.findPath(nb, end, path):
                        yield p
        except KeyError:
            return #None  # return the None singleton object

    # whether we are recursive over a cycle in the graph
    def cyclic(self, path, nextnode):
        return True if len(path) > 2 and path[-2] == nextnode else False 

    def findAllPaths(self, start, end, path=[]):
        paths = set()
        for path in self.findPath(start, end):
            paths.add(tuple(path))

        print paths

    # find the shortest path
    def findShortestPath(self, start, end, path=[]):
        shortestPath = [ 0 for i in xrange(100)]  # set to a huge num.
        for path in self.findPath(start, end):
            if len(path) < len(shortestPath):
                shortestPath = path

        print 'shortest Path:', shortestPath
    
    # in recusive mode, need to avoid cyclic
    def acycle(self, path, node):
        for e in path:
            if nb in e:  # the key in d test, has_key(k)
                return True
        return False

    # detect cycle in graph
    def detectCycle(self):
        # first, create color dict
        color = dict()
        for k in self.wgraph.keys():
            color[k] = 0 

        def recursiveDetect(root):
            hascycle = False
            if color[root] == 0:
                color[root] = 1
                # recursive on all its child
                for nb,cost in self.wgraph.get(root).items():
                    if color[nb] == 1:
                        print 'cyclic:', k, nb
                        hascycle = True
                    elif color[nb] == 0:
                        hascycle =  recursiveDetect(nb)
            # set root color to black after done with root
            color[root] = 2
            return hascycle

        for k in self.wgraph.keys():
            if recursiveDetect(k):
                print 'there is a cycle starting from:', k
            else:
                color[k] = 2    # done, black

    # generator of all paths between start, end
    def findWeightedPath(self, start, end, path=[]):
        if start == end:
            yield path, 

        try:
            for nb, cost in self.wgraph.get(start).items():
                if not acycle(path, nb):
                    path = path.append(dict({ nb : cost }))
                    for p in findWeightedPath(nb, end, path):
                        yield p
        except KeyError:
            return


    # Heap sort of dist[i] to source and  disjointed sets
    # For a given source, the algorithm finds the shortest path between that vertex 
    # and every other vertex.
    def dijkstra(self, source, dist={}, parent={}):
        # init everybody to inf and source to 0
        for k in self.wgraph:
            dist[k] = Graph._INFINITY
            parent[k] = None 
        dist[source] = 0
        parent[source] = source

        # we need to use priqueue, heap here! This code is wrong.
        tmpd = copy.deepcopy(dist)            # make a copy so we can remove items
        q = self.wgraph.keys()  # all the vertex
        while q:
            # sort the dict to an array and return the smallest one on top
            # [(1,'z'),(2,'x'),(3,'y')].sort(key=lambda x:x[1])
            node,cost = sorted(tmpd.iteritems(), key=operator.itemgetter(1))[0]
            if dist[node] == Graph._INFINITY:  # the remain nodes are unaccessible
                break   

            q.remove(node)
            tmpd[node] = Graph._INFINITY   #del tmpd[node] sorted not to find it again.
            # relax thru next node's neighbor 
            for k,v in self.wgraph.get(node).items():
                if dist[k] > dist[node] + v:
                    dist[k] = tmpd[k] = dist[node] + v 
                    parent[k] = node 
                    tmpd[k] = dist[k]

        #print 'Dijkstra dist From :', source, dist
        #print 'Dijkstra path From :', source, parent

    def findAllDijkstra(self):
        q = self.wgraph.keys()
        while q:
            source = q.pop(0)
            pathd = dict()
            costd = dict()
            self.dijkstra(source, costd, pathd)
            d = dict(cost=costd, path=pathd)
            self.dijkstraMatrix[source] = d

        for e in self.dijkstraMatrix.items():
            print e

if __name__ == "__main__":
    g = Graph()
    #g.findPath('A', 'E')
    #g.findAllPaths('B','D')
    #g.findShortestPath('B','D')
    #g.dijkstra('A')
    g.findAllDijkstra()
    g.detectCycle()
