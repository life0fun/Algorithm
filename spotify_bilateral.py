#!/usr/bin/python

import os,sys
import collections

"""
Spotify puzzle:  Hopcroft-Karp for maximum matching.
minimum vertex cover problem, find a min set of vertex that covers all edges(projects).
a match is a set of non-adj edges(no two edges share a vertex). an unmatched node is node none of its edges in matching.
Maximal matching: every edge intersect with edges in the matching. 
Maximum matching: a matching with max # of edges.
Perfect(complete) matching: a matching with all vertexes included.(edges contains all the vertex).
Aug Path against a match: a path alternate between matching edges and non-matching edges with both end vertex not matches.
Each aug path expand the match with one more edge. from subproblem to complete. Alternate ensures existing matched node still be matched.
Algorithm:
  1. init a simplest matching.
  2. for each unmatched lhs vertex, find an aug path, 
       new_matching = (Aug_path - matching_edges) + (matching_edges - matching_edges_in_aug_path)
  3. repeat until no aug path 
"""
class Bilateral():
    def __init__(self):
        self.MAX = 1<<32
        self.friend = '1009'  # the id of friend
        self.friendpeer = None
          
        ''' lhs/rhs is a map, key=vertex, val=[lhs:rhs] if vertex matches, val=[0] if vertex unmatched.
            matching =[(lhs:rhs)...]
        '''
        self.Q = collections.deque()
        self.G = {}
        self.rG = {}  #key=rhs, val={lhs:0, ...}
        self.matching = [] # a matching edge repr : 'lhs:rhs'
        self.lhs = {}      # k=vertex, val=[matching_edge]
        self.rhs = {}
        self.pair = {}   # pair for hopcroft-karp algorithm, unpaired/unmatched vertex = nil
        self.dist = {}   # aug path length. val is [0..inf], for paired vertex,(in matching), dist=path len, unpaired, dist=aug path length
        self.matchingpeer = {} # key is vertex, value is vertex of the other end of the matching edge
        self.mincover = set()
        
    def log(self, *x):
        #print x
        pass

    def getProjectsStdin(self):
        data = sys.stdin.readlines()
        locs = (line.split() for line in data)
        for l in locs:
            if len(l) == 2:
                self.G.setdefault(l[0], {}).update({l[1]:0})
                # self.rG.setdefault(l[1], {}).update({l[0]:0}) # rG populated inside initMatching
                self.log(' stdin: ', l[0], l[1])
        self.log(self.G)

    def getProjects(self, file):
        logf = open(file)
        locs = (line.split() for line in logf)  #generator
        i = 0
        for l in locs:
            if len(l) == 2:  # strictly take 2 and 3 in a line so comment lines can be bypassed
                i += 1
                self.G.setdefault(l[0], {}).update({l[1]:0})
                # self.rG.setdefault(l[1], {}).update({l[0]:0}) # rG populated inside initMatching
                
        self.log(self.G)

    def pplprojrank(self, pplprojd):
        # this sort of 1k ppl, should be ok ?
        l = sorted(pplprojd.items(), key=lambda e: len(e[1]), reverse=True)
        return l[0]   # return the top gun
                
    def addMatching(self, edge):
        lv, rv = edge.split(':')
        self.lhs[lv].append(edge)
        self.rhs[rv].append(edge)
        self.matching.append(edge)
    def delMatching(self, edge):
        lv, rv = edge.split(':')
        self.lhs[lv].remove(edge)
        self.rhs[rv].remove(edge)
        self.matching.remove(edge)

    ''' get un-matched vertex from lhs ?. There are cases that a node never be matched, use a queue to prevent inf loop '''
    def getUnmatched(self):
        self.Q.clear()   # clear Q first
        for k,v in self.lhs.items():
            if not len(v):
                self.Q.append(k)
                
        if len(self.Q):
            return True
        else:
            return False
    
    ''' DFS search for an aug path, when branching bound, dup data structure'''
    def DFSAugPath(self, v, augpath):
        self.log('-- DFS find Aug Path with unmatching node:' + v + ' : existing aug path:', augpath, ' Matching:', self.matching)
        if v in self.G:  # from left->right
            for rv in self.G[v].keys():  # for all the rhs endpoint edges v h reachable
                ''' prevent cycle'''
                curedge = v+':'+rv
                if curedge in augpath:  #edge already in path
                    continue
                if curedge in self.matching:  # alternate path, edge from lhs->rhs must not be in  matching
                    continue
                
                augpath.append(curedge)  # append this alternate path
                
                '''rv does not have a matching, cool, found an unmatched rhs node, done.'''
                if not len(self.rhs[rv]): 
                    return True, augpath
                ''' rv either matched vertex or unmatched, now it has a matching edge from rv-lv'''
                if len(self.rhs[rv]) > 0:
                    medge = self.rhs[rv][0] # edge in match, lhs - rhs
                    augpath.append(medge)   # return back to lhs, add this matching edge, DFS recursion
                    self.log(' DFS recursion: ', v, rv, medge, augpath)
                    found, path = self.DFSAugPath(medge.split(':')[0], augpath)  # DFS recursive until an unmatched node is found
                    ''' DFS recursion in this path failed, pop stack, both medge and curedge, clean up for next DFS '''
                    if not found:
                        augpath.remove(medge)
                        augpath.remove(curedge)
                        continue
                    else:
                        return found, path  # return the found augpath
                    
            ''' dfs done all v's children, no path from v works. For vertex v, dead end, return fail so caller can pop stack'''
            return False, augpath
            
    ''' update existing matching with augpath. remove common edges among existing matching and augpath, keep the diff edges''' 
    def updateMatching(self, augpath):
        for edge in augpath:
            if edge in self.matching:
                self.delMatching(edge)
            else:
                self.addMatching(edge)
        self.log(' ------------ Updated Matching ---------')
        self.log(self.matching)
    
    ''' repeatively get each unmatchable vertex and DFS on it until cant matchable '''
    def getMaxMatching(self):
        unmatchable = set()
        while self.getUnmatched():
            if set(self.Q) == unmatchable:
                break
            
            while len(self.Q):  # loop thru the list of unmatchable vertex
                unmatchv = self.Q.popleft()
                augpath = []
                found, augpath = self.DFSAugPath(unmatchv, augpath)
                if found:
                    if unmatchv in unmatchable:
                        unmatchable.remove(unmatchv)
                    self.updateMatching(augpath)
                else:  
                    ''' this vertex could not be matched '''
                    self.log(' never matched vertex:' + unmatchv)
                    unmatchable.add(unmatchv)
                
        self.log(' ---- final matching ---')   # (['B:1', 'E:3', 'C:2', 'A:4', 'D:5'])
        self.log(self.matching)
        
    ''' we got maximum matching, find the min cover vertex. vertex set covers all edges
    ''' 
    def getMinCover(self):
        for e in self.matching:
            lv, rv = e.split(':')
            self.log('matching:', lv, rv, self.lhs[lv], self.rhs[rv])
            
            if len(self.G[lv]) > len(self.rG[rv]) or ( len(self.G[lv]) == len(self.rG[rv]) and lv == self.friend):
                self.mincover.add(lv)
                for vr in self.G[lv].keys():
                    del self.G[lv][vr]
                    del self.rG[vr][lv]
            elif len(self.G[lv]) < len(self.rG[rv]) or ( len(self.G[lv]) == len(self.rG[rv]) and rv == self.friend):
                self.mincover.add(rv)
                for vl in self.rG[rv].keys():
                    del self.rG[rv][vl]
                    del self.G[vl][rv]
            else:  # equals but none is my friend, choose any
                self.mincover.add(lv)
                for vr in self.G[lv].keys():
                    del self.G[lv][vr]
                    del self.rG[vr][lv]
                    
        self.log(' ----- min cover -------')
        self.log(self.mincover)
        
    ''' after maximum matching edges found, figure out the min set cover'''
    def getMinCoverLTR(self):
        self.log(' max matching pair :', self.pair)
        self.log(' max matching edges:', self.matching)
        l = set(self.G.keys())
        r = set(self.rG.keys())
        t = set()
        if self.G.has_key(self.friend) and self.pair[self.friend] != 'nil':
            self.friendpeer = self.pair[self.friend]
            self.log('frend:friendpeer', self.friend, self.friendpeer)
            
        # populate t, unmatched from l, + all vertex in R and L that is alternatively reachable
        # for each unmatched vertex in L, we add into T all vertices that occur in a path alternating between unmatched/matched edges.
        # ex: c->2->b->4->d, get rid of bcd, make 2,4 in min vertex cover.
        for k in self.G.keys():
            if self.pair[k] == 'nil':  # for each unmatched vertex in L, find all its alt paths 
                self.allAlternatePath(k, t)
            
        #self.mincover = (l - t) | (r & t)
        #self.mincover =  l.difference(t).union(r.intersection(t))
        self.mincover =  (l - t) | (r & t)
        self.log(' mincoverLTR: ', self.mincover)
    
    def edge2set(self, edges, t):
        for e in edges:
            l,r = e.split(':')
            t.add(l)
            t.add(r)
            
    def allAlternatePath(self, k, t):
        if k in t:
            return
        
        allaltpath = []
        altpath = []
        ''' dfs all the children of k, find all altpath and put them into all path list'''
        found, foundpath = self.DFSAlternatePath(k, altpath, allaltpath)
        if found:
            self.log('good alternatePath: ', k, foundpath)
            self.edge2set(foundpath, t)
        else:
            for p in allaltpath:
                if len(p) > 0:
                    self.log('bad alternatePath: ', k, allaltpath[0])
                    self.edge2set(p, t)
                    return
            
    ''' dfs find all alter path start with k, and pick one that avoid friend. k must be lhs vertex'''
    def DFSAlternatePath(self, v, altpath, allpath):
        self.log('-- DFS find alterpath :', v, altpath)
        leaf = True
        for rv in self.G[v].keys():     # for all V's peers in R, sorted(self.G[v].keys(), key=lambda x:x):
        #for rv in sorted(self.G[v].keys(), key=lambda x:x):
            if self.pair[rv] != 'nil':  # peer must be in matching, so alter path back to L thru matching
                edge = v+':'+rv
                if not edge in altpath and not edge in self.matching:   # prevent cycle
                    leaf = False  # there is a path out, not a leaf layer
                    nextaltpath = list(altpath)
                    nextaltpath.append(edge)  # append no-matching edge v-rv
                    nextaltpath.append(self.pair[rv]+':'+rv)      # append matching edge, now altpath end at lhs
                    found, foundpath = self.DFSAlternatePath(self.pair[rv], nextaltpath, allpath)
                    if found:
                        return True, foundpath   # ret immediately if found, no need to loop all peers.
                    else:
                        self.log('DFS alterpath not avoiding friend, keep trying to find another path')
                    
        if leaf:   # check and return only reach leaf level
            # ret false if this altpath contains friend, need to continue, remember this path into allpath.
            if self.friendpeer and self.friend+':'+self.friendpeer in altpath:
                allpath.append(altpath)   # before false can be returned, allpath is gurantee to have one path
                self.log('DFS altpath include friend, this path is false, try again')
                return False, None
            else:
                return True, altpath
        else:  # if not leaf, then we could not find an altpath path to avoid friend
            return False, None   # at this point, none of children path can avoid friend, return false
        
    
    ''' Hopcroft-Karp algorithm, http://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm
        G = lhs U rhs U {NIL} NIL is a special null vertex 
    '''
    def checkUnmatchedVertexByBFS(self):
        for v in self.G:
            if self.pair[v] == 'nil':  # for unpaired vertex, init aug path len to 0
                self.dist[v] = 0
                self.Q.append(v)       # enqueue un-paired vertex
            else:
                self.dist[v] = self.MAX  # for paired vertex, init aug path to max,
        self.dist['nil'] = self.MAX
        
        while len(self.Q) > 0: 
            v = self.Q.popleft()         # v from q is an unpaired vertex
            for rv in self.G[v].keys():  # for each peer of v, 
                self.log(v, rv, self.pair[v], self.pair[rv])
                if self.dist[self.pair[rv]] == self.MAX:         # rv has a match back to lv
                    self.dist[self.pair[rv]] = self.dist[v] + 1  # alternate, change match to unmatch
                    if self.pair[rv] != 'nil':    # do not enqueue invalid nil node
                        self.Q.append(self.pair[rv])             # enq the alternatively changed back to unmatched node,
                    
        return self.dist['nil'] != self.MAX   # unmatched node paired to nil, if alternated found, nil's dist changed. not max anymore.
            
    ''' DFS find an un-matched node with aug path length k. ret true if found an unmatched node, False otherwise '''
    def DFS(self, v):
        if v != 'nil':
            self.log(' DFS with :', v)
            for rv in self.G[v].keys():
                if self.dist[self.pair[rv]] == self.dist[v] + 1:   # only dfs for aug path of length > k
                    if self.DFS(self.pair[rv]):
                        self.pair[rv] = v
                        self.pair[v] = rv
                        return True
            # done with all v's edges, not found any un-matched node, all node matches, set dist to inf, ret false
            self.dist[v] = self.MAX
            return False
        return True   # nil node means an un-matched rhs when calling into this func, ret true Found. 

                    
    def HopcroftKarp(self):
        for v in self.G:
            self.pair[v] = 'nil'
        for v in self.rG:
            self.pair[v] = 'nil'
        self.pair['nil'] = 'nil'
            
        matches = 0
        del self.matching[:]
        while self.checkUnmatchedVertexByBFS():  # BFS check unmatched vertex, pair[v]=nil
            for v in self.G.keys():
                if self.pair[v] == 'nil':  # unmatched vertex
                    if self.DFS(v):
                        matches += 1
                        
        # now for each edge, select the node with max degree                
        self.log(' final matchings: ', matches)
        for v in self.pair:
            self.log('pairs:', v, self.pair[v])
            if v != 'nil' and self.pair[v] != 'nil':
                if v in self.G.keys():   # only lhs got added
                    edge = v+':'+self.pair[v]
                    self.addMatching(edge)
        
    ''' ----------------------------------
        Maximum matching with max independent set of graph (L,R,E).
        G is repr by a dict of vertex in L to a list of their nbs in V
       ----------------------------------
    '''
    def MaxIndepSet(self):
        self.pair.clear()
        for v in self.G:
            self.pair[v] = 'nil'
        for v in self.rG:
            self.pair[v] = 'nil'
            
        for l in self.G:
            for r in self.G[l]:
                if self.pair[r] == 'nil':
                    self.pair[r] = l     # set mat
                    break
        
        while True:
            unmatched = []     # a list of unmatched vertex in v
            prelayernbR = {}   # list of nb in prev layer for v in V
            prelayernbL = dict([l, unmatched] for l in self.G)
            for r in self.pair:  # for each paired r, del from L its peer
                if self.pair[r] != 'nil':
                    del prelayernbL[self.pair[r]]
            layer = list(prelayernbL)
            
            # extend layer incremently one by one
            while layer and not unmatched:
                newlayer = {}
                for l in layer:
                    for r in self.G[l]:
                        if r not in prelayernbR:
                            newlayer.setdefault(r,[]).append(l)
                
                layer = []
                for r in newlayer:
                    prelayernbR[r] = newlayer[r]
                    if r in self.pair:
                        if self.pair[r] != 'nil':
                            layer.append(self.pair[r])
                            prelayernbL[self.pair[r]] = r
                        else:
                            pass
                    else:
                        unmatched.append(r)
            
            # is unmatched done ?
            if not unmatched:
                unlayered = {}
                for l in self.G:
                    for r in self.G[l]:
                        if r not in prelayernbR:
                            unlayered[r] = None
                
                ''' we've done, get the max matching'''
                self.log(' MaxIndepSet : final matchings: ', self.pair)
                tmppair = self.pair.copy()
                for v in tmppair:
                    if tmppair[v] == 'nil':   # skip unmatched pairs.
                        continue
                    
                    self.pair[tmppair[v]] = v   # populate the reverse of the edge
                    self.log('MaxIndepSet pairs:', v, self.pair[v])
                    if v in self.G.keys():   # only lhs got added
                        edge = v+':'+self.pair[v]
                    else:
                        edge = self.pair[v]+':'+v
                    self.addMatching(edge)
                    
                return self.pair
            
            # dfs search thru layers to find alternate path, ret true if path found
            def dfs(r):
                if r in prelayernbR:
                    l = prelayernbR[r]
                    del prelayernbR[r]
                    for u in l:
                        if u in prelayernbL:
                            pu = prelayernbL[u]
                            del prelayernbL[u]
                            if pu is unmatched or dfs(pu):
                                self.pair[r] = u
                                return True
                return False
            
            for v in unmatched:
                dfs(v)        
        
        
    ''' ----------------------------------
        main and unit test code    
       ----------------------------------
    '''
    def initMatching(self):
        for k,v in self.G.items():
            self.lhs.setdefault(k, []) # init to empty
            for vk, vv in v.items():
                self.rhs.setdefault(vk, [])
                self.rG.setdefault(vk, {}).update({k:vv})  # update will append dict items
        
        # add friend first
        if self.G.has_key(self.friend) and len(self.G[self.friend]) > 0:
            friendr = self.G[self.friend].keys()[0]
            self.addMatching(self.friend+':'+friendr)
        #self.addMatching('A:1')
        #self.addMatching('C:3')
        #self.addMatching('D:4')
        
    def maximumMatching(self):
        self.log( ' ----- finding maximum matching ----' )
        self.initMatching()
        
        algol = ['SimpleAugpath', 'HopcroftKarp', 'MaxIndepSet', 'BranchBound']
        algo = algol[1]  # manually select algo here
        
        if algo == algol[0]:
            self.getMaxMatching()
        elif algo == algol[1]:
            self.HopcroftKarp()
        elif algo == algol[2]:
            self.MaxIndepSet()

        self.getMinCoverLTR()
        print len(self.mincover)
        for v in self.mincover:
            if v != 'nil':
                print v

    def unitTest(self):
        ''' G final maximum matching (['B:1', 'E:3', 'C:2', 'A:4', 'D:5'],)  ['1', '3', '2', '4', 'D'] or abcde'''
        self.G = {
             'A':{'1':0, '2':0, '4':0},
             'B':{'1':0 },
             'C':{'2':0, '3':0},
             'D':{'4':0, '5':0},
             'E':{'3':0 }
        }
        
        #''' a little more complex example:  ['A:1', 'B:2', 'C:4', 'E:3'], min-cover ['A', '2', 'E', '4'] only
        self.G = {
             'A':{'1':0, '2':0, '3':0},
             'B':{'2':0 },
             'C':{'2':0, '4':0},
             'D':{'2':0, '4':0},
             'E':{'3':0, '5':0}
        }
        #'''
        
        self.maximumMatching()
         
if __name__ == '__main__':
    bilat = Bilateral()
    #bilat.unitTest()
    bilat.getProjectsStdin()   #  cat test/spotify.data | python spotify_bilateral.py
    bilat.maximumMatching()
    
    
    
