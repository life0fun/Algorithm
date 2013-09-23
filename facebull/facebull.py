#!/usr/bin/python

"""
Branch-and-cut to prune the search space for asymmetric TSP. 
Recursion in each branch, in each bound calculation.
ATSP Finds min weight hamiltonia cycle using DP with each vertix visisted only once.
FB find a set of edge with min weight, vertex can be visited +1 times.
ex:http://www.ce.rit.edu/~sjyeec/dmst.html
"""

import os,sys
from copy import deepcopy
#from decimal import *
#import pdb

class facebull():
    def __init__(self):
        self.MAX = 1<<32
        self.matrix = {}
        self.machines = {}
        self.nodeOutEdges = {}  # self.srcNodeEdges[front_door]={'under_bed': 5}
        self.nodeInEdges = {}
        self.allNodeMatrix = {}
        self.compounds = set()
        self.startpoint = None
        self.tab = {}
        self.optval = self.MAX
        self.optosold = {}
        self.optisold = {}
        
    def log(self, *k):
        #print k
        pass

    class Edge():
        def __init__(self, src, dst, costl):
            self.src = src
            self.dst = dst
            self.costl = list(costl) # do a copy
        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.src == other.src and self.dst == other.dst and self.costl[0] == other.costl[0] and self.costl[1] == other.costl[1]
            return False
        def __ne__(self, other):
            return not self.__eq__(other)
        
    def logNodeEdges(self, d, reverse):
        self.log(d)
        '''
        for k,v in d.items():
            for dst,l in v.items():
                if not reverse:
                    print k, '>>>', dst, ' : ', l[0]
                else:
                    print k, '<<<', dst, ' : ', l[0]
                print k, dst, l
                print v
        '''
    def getMachines(self, file):
        logf = open(file)
        locs = (line.split() for line in logf)  #generator
        for l in locs:
            if not l:
                continue

            machine, csrc, cdst, cost = l
            self.matrix[csrc+' '+cdst] = [0]*2 
            self.matrix[csrc+' '+cdst][0] = int(cost)
            self.matrix[csrc+' '+cdst][1] = machine
            self.compounds.add(csrc)
            self.compounds.add(cdst)
            # {c2: {c4:[580,580,0], c5:[3,3,0], ...  }
            # the first cost is orig edge weight, the second one is modified cost, 0 if incl the edge
            self.nodeOutEdges.setdefault(csrc, {}).update(dict({cdst : [int(cost), int(cost), 0]}))
            self.nodeInEdges.setdefault(cdst, {}).update(dict({csrc : [int(cost), int(cost), 0]}))

        for k,v in self.matrix.items():
            self.log(k, v)
        self.log('compounds:', self.compounds)
        self.logNodeEdges(self.nodeOutEdges, False)
        self.logNodeEdges(self.nodeInEdges, True)
    
    ''' get the next unvisited out-going edge(color flag, costl[2] = 0) 
        not in any particular order, just get one edge from the pool whose color flag is blank
    '''
    def getUnvisitedEdge(self, gd):
        for src, edged in gd.items():
            for dst, costl in edged.items():
                if costl[2] == 0:
                    self.log('unvisited:', src, dst)
                    return src, dst
        return None, None 
    
    def setEdgeVisited(self, ogd, igd, src, dst):
        ogd.get(src).get(dst)[2] = 1
        igd.get(dst).get(src)[2] = 1
    ''' update the runtime cost of edge to 0 after included the edge into partial solution set'''
    def setEdgeWT(self, val, ogd, igd, src, dst):
        ogd.get(src).get(dst)[0] = val
        igd.get(dst).get(src)[0] = val
        
    def stronglyConnected(self, osold, isold):
        edges = []
        for src, childrend in osold.items():
            for dst, tochild in childrend.items():
                edges.append((src, dst))

        if len(self.compounds) > len(edges):   # edges must be at least equals to vertices.
            return False
        
        #each node must have at least one egde in/out
        if self.compounds != set([e[0] for e in edges]) or self.compounds != set([e[1] for e in edges]):
            return False
        
        self.log(osold)
        ''' from any node, bfs all the edges, should be full cycle'''
        start = edges[0][0]
        s = set()
        q = []
        q.append(start)
        while len(q):   # BFS search, prevent cycle
            v = q.pop(0)
            s.add(v)
            for k in osold.get(v).keys():
                self.log(v, k)
                if not k in s:
                    q.append(k)
                    
        if self.compounds != s:
            return False
        return True
                        
    def insertSolEdge(self, ogd, igd, osold, isold, src, dst, costl):
        l = list(costl)
        osold.setdefault(src, {}).update({dst:l}) # {c2: {c4:[580,580,0]}}
        isold.setdefault(dst, {}).update({src:l}) # [origWT, runtimeWT, flag]
        self.setEdgeWT(0, ogd, igd, src, dst)
        #ogd.get(src).get(dst)[0] = 0   # set the cost to 0 after including the edge
        #igd.get(dst).get(src)[0] = 0 
        return self.nodeOutEdges.get(src).get(dst)[1]  # return the orig cost between src-dst
    
    ''' upset the known optimal solution so far with the new solution, if better'''
    def upBeat(self, val, osold, isold):
        if val < self.optval:
            if self.stronglyConnected(osold, isold):
                self.optval = val
                self.optosold = osold
                self.optisold = isold
                self.log('upbeat:', val, osold)
        
    ''' @deprecated. we are solving strong connected span tree problem'''
    # calculate the shortest path between each pair of locations
    def dijkstra(self, start, dist={}, parent={}):
        q = list(self.compounds)
        #dist.k is from start->k, populate and return dist and path map 
        for k in q:
            dist[k] = sys.maxint
            parent[k] = None
        dist[start] = 0
        parent[start] = []

        # be aware if no route to start node, matrix might not have start.
        tmpdist = deepcopy(dist)

        while q:
            ''' resort the dist before extractmin after relax with prev min'''
            node,cost = sorted(tmpdist.iteritems(), key=lambda x: x[1])[0] #('k', 5)
            if cost == sys.maxint: # the min arch is un-reachable from src, break;
                break

            q.remove(node) 
            del tmpdist[node] 

            # if the intermediate node has any route out
            if self.nodeOutEdges.has_key(node):
                for k,v in self.nodeOutEdges.get(node).iteritems():
                    if dist[k] > dist[node] + v:   # relax the map of start to k thru interim node
                        tmpdist[k] = dist[k] = dist[node] + v  #update tmp dist
                        parent[k] = list(parent[node])  # key is cur node, value is parent
                        parent[k].append(node)

    ''' @deprecated. we are solving strong connected span tree problem'''
    def allDijkstra(self):
        for node in self.compounds:
            path = {}
            cost = {}
            self.dijkstra(node, cost, path)
            self.allNodeMatrix[node] = dict(path=path, cost=cost) 
            #{'path': {'front_door': 'front_door', 'in_cabinet': None, 'behind_blinds': 'front_door', 'under_bed': 'front_door'}, 'cost': {'front_door': 0, 'in_cabinet': 2147483647, 'behind_blinds': 9, 'under_bed': 5}}
            self.log('root:', node, ' Matrix:', self.allNodeMatrix[node])

    ''' find a detour path that is cheaper than the direct flight cur->end '''
    def triangleInEquality(self, ogd, igd, parent, cur, end, curwt, upperbd, res):
        for child, tochild in ogd.get(cur).items(): # loop thru all children of cur node
            #self.log('triangle:', parent, cur, child, end, curwt, tochild[0], upperbd)
            if parent == child or curwt+tochild[0] >= upperbd: #use the current wt(could be reduced to 0), not the orig wt.
                continue  # skip if child edge back to parent, or (cur-next)=(parent-end) when started
            if(cur, child) in res: # when DFS/BFS, must check cycle.
                continue
            if child == end:
                res.append((cur,child))  # ok, reach the end, append it
                return True, curwt+tochild[0]  # found and return
            # push this round  intermediate edge into partial result list and recursion
            res.insert(0, (cur,child))
            # recursion, parent <- cur, cur <- child, with updated partial weight and result set.
            prune, wt = self.triangleInEquality(ogd, igd, cur, child, end, curwt+tochild[0], upperbd, res)
            if prune:
                #self.log('triangle:', start, '->', next, res)
                return prune, wt
            res.pop(0)  # not prunbable, remove this edge that was pushed in before recursion.
        return False, upperbd  # not found after looping all children
    
    ''' prune branches of a node, if incl the branch, optimal solution still worse
        a branch is just one child path. remove the child if a->c is worse than a->b->c
    '''
    def pruneOneNode(self, ogd, igd, node):  
        self.log(' === pruneNode:', node, ogd.get(node).items())
        for c, wl in ogd.get(node).items():  # for each edge (node->c)
            if not wl[0]:
                continue  # edge already 0 wt, skip the edge
            res = []
            prune, wt = self.triangleInEquality(ogd, igd, node, node, c, 0, wl[0], res) # parent=cur, no cost=0
            if prune:
                self.log('prune edge:', node, '->', c, res)
                self.pruneOneEdge(ogd, igd, node, c)
    
    ''' prune one edge, del the edge from out dict and in dict '''
    def pruneOneEdge(self, ogd, igd, src, dst):
        del ogd.get(src)[dst]  # del the key from both out dict and in dict
        del igd.get(dst)[src]
        
    ''' with the current partial solution, out-bound and in-bound edge dist,
        prune the branches that apparently worse than the known optimal solution so far
    '''
    def pruneOnePass(self, ogd, igd):
        for node in self.compounds:   # apply to each node
            self.pruneOneNode(ogd, igd, node)
            
    ''' For each edge with start/end node, recursion, ogd is dict of outgoing edges. 
        the partial solution so far before the recursion: curval is the cost 
        osold is dict of outgoing edges in partial solution set 
        curval is the sum of cost of edges.
    '''
    def DP(self, ogd, igd, start, end, osold, isold, curval):
        if curval > self.optval:
            return curval
        
        self.log('DP:start:', start, end, curval, ogd, '===', osold)
        # must take the edge if it is the only out edge of the start
        if not osold.has_key(start) and len(ogd.get(start)) == 1: 
            self.setEdgeVisited(ogd, igd, start, ogd.get(start).keys()[0])
            curval += self.insertSolEdge(ogd, igd, osold, isold, start, ogd.get(start).keys()[0], ogd.get(start).values()[0])
            self.pruneOnePass(ogd, igd)   # update partial solution, prune, and recursion        
            nextsrc, nextdst = self.getUnvisitedEdge(ogd)  # recursion on next out-going egde
            self.log('DP:singleout:', start, nextsrc, nextdst, ogd, '==', osold)
            if not nextsrc:  # all edge traversed
                self.upBeat(curval, osold, isold)
                return curval
            # happy recursion with next edge and updated partial solution
            return self.DP(ogd, igd, nextsrc, nextdst, osold, isold, curval)
            
        # must take the edge if it is the only in-bound edge of the end node
        elif not isold.has_key(end) and len(igd.get(end)) == 1: 
            self.setEdgeVisited(ogd, igd, igd.get(end).keys()[0], end)
            curval += self.insertSolEdge(ogd, igd, osold, isold, igd.get(end).keys()[0], end, igd.get(end).values()[0])
            self.pruneOnePass(ogd, igd)   # update partial solution, prune, and recursion
            nextsrc, nextdst = self.getUnvisitedEdge(ogd)  # recursion on next out-going edge
            self.log('DP:singlein:', end, nextsrc, nextdst, ogd, '==', osold)
            if not nextsrc:
                self.upBeat(curval, osold, isold)
                return curval
            # happy recursion with next edge and updated partial solution
            return self.DP(ogd, igd, nextsrc, nextdst, osold, isold, curval)
            
        ''' edge not the single out/in edge of node start/end, branch out with incl or not.
            branch: optimal solution incl the edge, find the partial solution val, prune.
            branch: optimal solution not incl the edge, no add to partial sol, no prune, continue, 
            end when all edge has been visited, branched with incl/excl.
        '''
        self.setEdgeVisited(ogd, igd, start, end)  # the edge has been visited
        if curval + self.nodeOutEdges.get(start).get(end)[1] < self.optval: # not overflowed yet
            inclogd = deepcopy(ogd)    # dup partial solution before branch
            incligd = deepcopy(igd)    # preserve data of this branch/run
            inclosold = deepcopy(osold)
            inclisold = deepcopy(isold)
            inclcurval = curval + self.insertSolEdge(inclogd, incligd, inclosold, inclisold, start, end, inclogd.get(start).get(end))
            # curval updated and prune when adding the edge into the solution
            self.pruneOnePass(inclogd, incligd)
            nextsrc, nextdst = self.getUnvisitedEdge(inclogd)
            self.log('DP:incl:', start, end, nextsrc, nextdst, inclogd, '=', inclosold)
            self.log('DP:incl:', inclcurval)
            if not nextsrc:   # done traversing all edges with better wt, update upper bound
                self.upBeat(inclcurval, inclosold, inclisold)
                return inclcurval
            self.DP(inclogd, incligd, nextsrc, nextdst, inclosold, inclisold, inclcurval) 
        
        '''the edge has been visited, excl the edge, excl the edge by setting its wt to be infinity.
           no adding the edge to sol, no curval update, no prune, move to next, reuse the matrix dict
        '''
        nextsrc, nextdst = self.getUnvisitedEdge(ogd)
        if not nextsrc: 
            self.log('Done...not better than optimal, nothing')
            return
        
        self.log('DP:excl:', start, end, nextsrc, nextdst, ogd,'=', osold)
        self.pruneOneEdge(ogd, igd, start, end)
        return self.DP(ogd, igd, nextsrc, nextdst, osold, isold, curval)
        
    def scss(self):  # strong connected spanning subgraph
        #self.allDijkstra()
        #self.unitTest()
        self.pruneOnePass(self.nodeOutEdges, self.nodeInEdges)
        
        ''' for each edge, decision tree to incl it or not '''
        ''' ogd is a {dict} of outgoing edges for each node, key is node, val is {} '''
        ogd = deepcopy(self.nodeOutEdges)  #{c2: {c4:[580,580,0], c5:[3,3,0], ...  }
        igd = deepcopy(self.nodeInEdges)
        osold = {}
        isold = {}
        src, dst = self.getUnvisitedEdge(ogd)
        self.log('starting:', src, dst)
        self.DP(ogd, igd, src, dst, osold, isold, 0)
        
        self.log("=======Optimal Solution========")
        print self.optval
        self.log(self.optosold)
        self.formatResult()
            
    def formatResult(self):
        edges = []
        for src, childrend in self.optosold.items():
            for dst, tochild in childrend.items():
                edges.append(int(self.matrix[src+' '+dst][1][1:]))
        edges.sort(cmp=None, key=None, reverse=False)
        for i in xrange(len(edges)-1):
            print edges[i],
        print edges[i+1]

    def unitTest(self):
        print self.getUnvisitedEdge(self.nodeOutEdges)
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: facebull.py cf'
        sys.exit(1)

    fb = facebull()
    fb.getMachines(sys.argv[1])
    fb.scss()
