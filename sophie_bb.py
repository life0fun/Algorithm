#!/usr/bin/python

import os,sys
from decimal import *
#import copy
#import pdb

"""
DP with Branch-and-Bound prune.
Compute the lower and upper bound at each level(partial solution with a path prefix)
Upper bound is the min solution encountered so far.
Lower bound of a partial solution: 
    no_of_nodes_left * min_dist_to_those_nodes * sum_of_prob_of_those_nodes
Prune:  lower_bound >= upper bound.
"""

class Sophie():
    def __init__(self):
        self.MAX = 1<<64
        self.locProb = {}  # key is loc name, val is prob
        self.distMatrix = {}  # key is src loc name, val=dic[dist_loc]=distance
        self.dijkstraMatrix = {}  #key is src loc, val=dict(path={}, cost={})
        self.sortedDist = {}  # [('front_door', 0), ('under_bed', 5),...]
        self.startpoint = None
        self.tab = {}
        self.upperbound = {}  # key is path, val is min expected for this path. 
        self.minexp = self.MAX
        
    def log(self, *x):
        #print x
        pass

    def getLocations(self, file):
        logf = open(file)
        locs = (line.split() for line in logf)  #generator
        for l in locs:
            if len(l) == 2:  # strictly take 2 and 3 in a line so comment lines can be bypassed
                if not self.startpoint:
                    self.startpoint = l[0].strip()
                    #self.locProb[l[0].strip()] = Decimal(l[1].strip())
                    self.locProb[l[0].strip()] = float(l[1].strip())
                #elif Decimal(l[1].strip()) - 0 == 0:
                elif float(l[1].strip()) == 0:
                    continue
                else:
                    #self.locProb[l[0].strip()] = Decimal(l[1].strip())
                    self.locProb[l[0].strip()] = float(l[1].strip())
            if len(l) == 3:  # populate two way matrix, src->dist, dist->src
                #self.distMatrix.setdefault(l[0].strip(), {}).update(dict({l[1].strip() : Decimal(l[2])}))
                self.distMatrix.setdefault(l[0].strip(), {}).update(dict({l[1].strip() : float(l[2])}))
                #self.distMatrix.setdefault(l[1].strip(), {}).update(dict({l[0].strip() : Decimal(l[2])}))
                self.distMatrix.setdefault(l[1].strip(), {}).update(dict({l[0].strip() : float(l[2])}))
                # self.distMatrix[front_door]={'under_bed': 5}

        #self.locProb = sorted(self.locProb.iteritems(), key=lambda x:x[1])
        self.log(self.distMatrix)


    # calculate the shortest path between each pair of locations
    def dijkstra(self, start, dist={}, parent={}):
        q = self.distMatrix.keys()
        for k in q:
            dist[k] = self.MAX
            parent[k] = None
        #dist to start is 0, infinit to others
        dist[start] = 0
        parent[start] = start

        # be aware if no route to start node, matrix might not have start.
        tmpdist = dist.copy()  #tmpdist = copy.deepcopy(dist)

        while q:
            ''' resort the dist array everytime extract min'''
            ''' use min heap, so the heap is maintained on every insertion'''
            node,cost = sorted(tmpdist.iteritems(), key=lambda x: x[1])[0]
            #self.log('tmpdist: start:', start, ' nextnode:', node, ' cost:', cost, '=>' , tmpdist)
            if cost == self.MAX: # the min arch is un-reachable from src, break;
                break

            q.remove(node)
            del tmpdist[node] # extract min is smallest vertex in Q

            # if the intermediate node has any route out
            if self.distMatrix.has_key(node):
                for k,v in self.distMatrix.get(node).iteritems():
                    if dist[k] > dist[node] + v:
                        tmpdist[k] = dist[k] = dist[node] + v  #update tmp dist
                        parent[k] = node  # key is cur node, value is parent

    def allDijkstra(self):
        for node in self.distMatrix:
            path = {}
            cost = {}
            self.dijkstra(node, cost, path)
            self.dijkstraMatrix[node] = dict(path=path, cost=cost) 
            self.sortedDist[node] = sorted(self.dijkstraMatrix[node]['cost'].iteritems(), key=lambda x:x[1])
            #{'path': {'front_door': 'front_door', 'in_cabinet': None, 'behind_blinds': 'front_door', 'under_bed': 'front_door'}, 'cost': {'front_door': 0, 'in_cabinet': 2147483647, 'behind_blinds': 9, 'under_bed': 5}}

        for k,v in self.dijkstraMatrix.items():
            self.log('-----')
            self.log(k)
            self.log(v)

    def subsetSumProb(self, path):
        return reduce(lambda x,y:x+y, [self.locProb[e] for e in path])

    ''' pass in the min path with start at the head, find the cumulate dist'''
    def cumulatePathExp(self, path):
        tot = 0
        l = []
        i = 1
        exp = 0
        while i < len(path):
            gap = self.dijkstraMatrix[path[i-1]]['cost'][path[i]]
            tot += gap
            exp += tot*self.locProb[path[i]]
            #self.log('head:', path[i-1], ' next:', path[i], ' dist:', gap, ' prob:', self.locProb[path[i]], ' tot:', tot, ' exp:', exp) 
            i += 1
        return exp

    ''' update the upper bound for the cur path after DP tab '''
    def updateUpperBound(self, curpathkey):
        if not self.upperbound.has_key(curpathkey):
            self.upperbound[curpathkey] = [0]*2
            self.upperbound[curpathkey][0] = self.tab[curpathkey][0]  # set min path
            self.upperbound[curpathkey][1] = self.tab[curpathkey][1]  # set min exp
        elif self.upperbound[curpathkey][1] > self.tab[curpathkey][1]:
            self.upperbound[curpathkey][0] = self.tab[curpathkey][0]  # set min path
            self.upperbound[curpathkey][1] = self.tab[curpathkey][1]  # set min exp
       
        if self.minexp > self.upperbound[curpathkey][1]:
            self.minexp = self.upperbound[curpathkey][1]
        return

    '''give a node and already discovered nodes, heuristic the low bound of this path
       find the prob left, and the min dist to the undiscovered nodes from curnode
       path do not contain nextnode, ctime does not include cur to next
    '''
    def nodeROWMinOut(self, node, s):
        for e in self.sortedDist[node]:
            if not e[0] in s and e[0] in self.locProb:
                return e[0],e[1]
    
    ''' find the min incoming edge for the node from the set '''
    def nodeROWMinIn(self, enternode, ctime, s):
        rowexp = 0
        for node in s:
            if node == enternode: # cur->enternode already counted before entering 
                continue
            #for i in xrange(len(self.sortedDist[node])):
            #    if self.sortedDist[node][i][0] in s:
            #        rowexp += (ctime+self.sortedDist[node][i][1])*self.locProb[node]
            #        self.log('node:', node, 'node_min_in:', self.sortedDist[node][i], ' rowexp:', rowexp)
            #        break
            if not self.dijkstraMatrix.has_key(node):
                return self.MAX
            rowexp += (ctime+self.sortedDist[node][1][1])*self.locProb[node]
            continue
        self.log('Set:', s, ' minexp:', rowexp)
        return rowexp

    # parent, etc end with cur but not incl nextnode
    def heuristicROW(self, cur, nextnode, parentpath, curexp, cumultime):
        # first, prepare the row set
        s = set(parentpath)
        rows = set(self.locProb)
        self.log('parentpath:', parentpath, ' rowset:', rows)
        rows = rows.difference(s)

        # second, cal exp from cur => nextnode
        curtonext = self.dijkstraMatrix[cur]['cost'][nextnode]
        tot = cumultime + curtonext
        lowexp = tot*self.locProb[nextnode]

        # cal the min-in edge for all the nodes in ROW set
        rowexp = self.nodeROWMinIn(nextnode, tot, rows)
        lowexp += rowexp

        self.log('parentpath:', parentpath, ' rowset:', rows, ' rowset_lowexp:', rowexp, ' tot_lowexp:', lowexp)
        
        return lowexp

    ''' parentpath has cur in end but does not contain next node, 
        curexp and ctime incl cur node but not include cur to next
    '''
    def pruneBigLowBound(self, cur, nextnode, parentpath, curexp, cumultime):
        if curexp + self.heuristicROW(cur, nextnode, parentpath, curexp, cumultime) < self.minexp:
            return False  # do not prune if lowbound is within
        return True

    ''' traveling salesman DP BFS  BB algorithm 
        the nodes are divided into two sets, discovered parentpath set, and undiscovered ROW set
        after exploring ROW, records down in DP tab the min cost of exploring ROW given discovered
        parentpath set as the key.
    '''
    def DP(self, matrix, start, cur, minpath, parentpath, cumultime):
        # cur node has no out-going arch, no recursion on it.
        if not cur in self.dijkstraMatrix:
            return self.MAX

        # the path key is all prev visited node up to this iter + curnode
        # in this problem, the path order does matter because of _cumul_*prob
        curpath = list(parentpath[:-1])
        #curpath.sort()
        curpath.append(cur)
        curpathkey = str(curpath)
        self.log('curpathkey:', curpathkey, ' minpath:', minpath, ' cumultime:', cumultime)

        '''asks 2 questions, empty search space, cons results of children'''
        if len(curpath) == len(self.locProb):  # exit strategy first!!!
            ''' cur to start form the hamilton cycle, no more further cost'''
            thispathmin = self.cumulatePathExp(minpath)
            if not self.tab.has_key(curpathkey):
                self.tab[curpathkey] = [0]*3
                self.tab[curpathkey][0] = minpath
                self.tab[curpathkey][1] = self.cumulatePathExp(minpath)
            elif self.tab[curpathkey][1] > thispathmin:
                self.tab[curpathkey][0] = minpath
                self.tab[curpathkey][1] = self.cumulatePathExp(minpath)
            #self.tab[curpathkey][2] = totalexp
            self.log('**CYCLE:', parentpath, ' cumultime:', cumultime, ' minpath:', minpath)
            self.updateUpperBound(curpathkey)
            return self.tab[curpathkey][1]

        ''' DP tab records the min cost of ROW after curpath set'''
        if self.tab.has_key(curpathkey):  # get the min from DP tab directly
            tabminpath = self.tab[curpathkey][0]
            minpath.extend(tabminpath[len(minpath):])  # the minpath of ROW since curpathkey
            tabexp = self.cumulatePathExp(minpath)   
            self.log('tab found:', curpathkey, ' tabexp:', tabexp, ' minpath:', minpath)
            return tabexp
            #pass

        ''' DP tab not found, DFS BB children, and recursion on each child'''
        minexp, curexp = self.MAX, self.cumulatePathExp(minpath) 
        nextnode, nextminpath = None, None 
        sortednodeprob = sorted(self.locProb.iteritems(), key=lambda x:x[1], reverse=True)
        sortednodedist = sorted(self.dijkstraMatrix[cur]['cost'].iteritems(), key=lambda x:x[1])
        for node, prob in sortednodedist:
            if node in curpath:
                continue

            # check if the node is reachable from curnode
            if not node in self.dijkstraMatrix[cur]['cost'] or not node in self.locProb or self.dijkstraMatrix[cur]['cost'][node] == self.MAX:
                continue

            '''Prune the node if projected low bound greater than min exp'''
            if self.pruneBigLowBound(cur, node, parentpath, curexp, cumultime):
                self.log('cur:', cur, ' prune:', node, ' parentpath:', parentpath, ' minpath:', minpath) 
                tmpl = ['0', '8', '2', '15', '3', '14', '10']
                tmpl.sort()
                tmpl.append('12')
                #if  node == '6' and curpath == tmpl:
                #    sys.exit(1)
                continue

            tmpminpath = list(minpath) 
            parentpath.append(node)
            tmpminpath.append(node)    # already appended the next node here
            nextcumultime = 0 # cumultime+self.dijkstraMatrix[cur]['cost'][node]
            nextexp = 0       #totalexp + prob*nextcumultime
            nextexp = self.DP(self.dijkstraMatrix, start, node, tmpminpath, parentpath, cumultime+self.dijkstraMatrix[cur]['cost'][node])
            parentpath.pop()

            if nextexp < minexp:
                minexp, nextnode, nextminpath = nextexp, node, tmpminpath

        '''if cant find next node from cur before cycle done, just ret maxint'''
        if not nextnode:
            return self.MAX

        # cons nextnode, and its result, into the return value 
        ''' nextminpath is a dup of cur min with next node appended, hence remove dup part'''
        minpath.extend(nextminpath[len(minpath):])  

        self.tab[curpathkey] = [0]*3
        self.tab[curpathkey][0] = minpath   # curpathkey end by cur node
        self.tab[curpathkey][1] = self.cumulatePathExp(minpath)

        ''' now update upper bound'''
        self.updateUpperBound(curpathkey)

        self.log('curpath:', parentpath, 'cur->next:',self.dijkstraMatrix[cur]['path'][nextnode], ' next:', nextnode, ' nextmin:', nextminpath, 'cumultime:', cumultime, 'minexp:', self.tab[curpathkey][1], ' minpath:', minpath) 
        return self.tab[curpathkey][1]

    def expectedValue(self):
        minpath = []
        path = []
        start = self.startpoint
        path.append(start)
        minpath.append(start)
        #self.heuristicROW(start, start, [], 0, 0)
        exp = self.DP(self.dijkstraMatrix, start, start, minpath, path, 0)
        self.log('======= final result =======')
        self.log('expected:', exp, ' full path:', minpath)
        self.log('verified minpath exp:', self.cumulatePathExp(minpath), ' upperbound:', self.minexp)

        #print 'all Places:', self.locProb, ' minpath:', minpath
        for k in self.locProb:
            if not k in minpath:
                print "-1.00"
                return
        
        #print Decimal(exp).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
        print "%.2f" % (exp)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: sophie locations'
        sys.exit(1)

    sp = Sophie()
    sp.getLocations(sys.argv[1])

    #sp.dijkstra('front_door')
    sp.allDijkstra()
    sp.expectedValue()
