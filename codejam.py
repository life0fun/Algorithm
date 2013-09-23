#!/usr/bin/env python
"""
all the exercise for code jam
"""
import sys
import math
import operator
import bisect
import copy
import collections
import pdb

'''
find out all primes less than n.
first, create odd num array starts from 3, [3,5,7,9], val=idx*2+3, idx=(val-3)/2
foreach num in odd array, starts from its square, wipe out all it multiples.
iterate for each num in odd array until sqrt of n
'''
def myprimes(n):
    #l = [ 3+2i for i in xrange(1000)]
    l = range(3, n, 2)  # l is the odd array starting from 3.
    root = n ** 0.5
    stopidx = int((root-3)/2)

    # start from 3's multiples [3,5,7,9,11,13,15,17,19,21,23,25,27]
    # continue to 5's multiple, 7's multiple...#{loops}'s multiple
    for i in xrange(stopidx):  # i is the idx into odd array
        if l[i] == 0:        # already wiped out, done
            continue
        wheel = l[i]         # wipe out l[i]'s multiples, starting from its square, prev multiples already wiped out.
        wheelpow = pow(wheel,2)  # wheelpower = 9
        idx = (wheelpow-3)/2     # idx=3, l[3] = 9
        while idx <= len(l)-1:   # strike to the end of list
            #print 'striking out:', wheel, idx, l[idx]
            l[idx] = 0
            idx += wheel         # wheel size is step

    yield 2
    for p in l:
        if p:
            yield p

"""
Uses Euclidean algorithm to determine a^-1(mod m)
ax=1(mod m), ax+my=1(mod m), gcd(a,m)=1, => ax+by=gcd(a,b)
brute force: [ x if (a*x) %m == 1 else 0 for x in xrange(m) ]
"""
# Iterative Algorithm (xgcd)
def iterative_egcd(a, b):
    x, y, u,v = 0, 1, 1,0
    while a != 0:
        q,r = b/a, b%a;
        m,n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    return b, x, y

# Recursive Algorithm
def recursive_egcd(a, b):
    '''Returns a triple (g, x, y), such that ax + by = g = gcd(a,b).'''
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = iterative_egcd(a, m)  # or recursive_egcd(a, m)
    if g != 1:
        return None
    else:
        return x % m

def DeRNGed(primeupper, l):
    ''' a*ni+b=n{i+1}
        j=n+1
        a*nj+b=n{j+1}
        a*(ni-nj)=n{i+1}-n{j+1}
        a = n{i+1}-n{j+1}*{(ni-nj)^-1}
    '''
    a = b = 0
    lastidx = len(l)-1
    for p in myprimes(primeupper):
        if p < max(l):
            continue
        for i in xrange(lastidx-1): #size-1 is the last indx
            for j in xrange(i+1,lastidx-1,1):
                if l[i] == l[j]:
                    return l[i]  # dead loop, can return safely
                a = (((l[i+1]-l[j+1]+p)%p)*modinv((l[i]-l[j]+p)%p,p) % p)
                b = ((l[i+1]-a*l[i])%p+p)%p  # to make sure b>0
                #print a, b
                badguess = 0
                for k in xrange(1,len(l),1):
                    if l[k] != (a*l[k-1]+b)%p:
                        badguess = 1
                        break
                if not badguess:
                    print a, b, p
                    print 'Winner: the next is', (a*l[lastidx]+b)%p
                    return (a*l[lastidx]+b)%p

def testDeRNGed():
    print modinv(8,5) # 4 {8=3mod5}

    l = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
    primeupper = 10**2
    assert DeRNGed(primeupper, l) == 10

    l = [476969,515559,131530,405309]
    primeupper = 10**6
    assert DeRNGed(primeupper, l) == 333709

"""
Fence, min num of boards to build fence with exact l
DP[non-decreasing seq]. for each new state i, max(k=i-1...0)(tab[k]).
"""
def Fence(fencel, boardsl, curboard, out):
    # exit condition first
    if curboard >= len(boardsl) or fencel < boardsl[curboard]:
        return None, 0

    if fencel % boardsl[curboard] == 0:
        out[curboard] = fencel/boardsl[curboard]
        return True, fencel/boardsl[curboard]

    for i in xrange(fencel/boardsl[curboard], -1, -1):
        found, amt = Fence(fencel-i*boardsl[curboard], boardsl, curboard+1, out)
        if found:
            out[curboard] = i
            return True, i
    return None, 0

def testFence():
    fencel = 10000000001
    boardsl = [23,51,100]
    boardsl.sort(reverse=True)
    out = [0]*len(boardsl)
    Fence(fencel, boardsl, 0, out)
    print sum(out)

"""
diffsum, a permutation issue
enumerate all equations, given a sum N with base B
The solution involved continuously matrix transform
for each column in sum N, e.g., 5, horizontally, it can be from summation of 5, 15, 25, ..
vertically, the max carry from prev column is B, so the sum can be 15, 14, 13,...2,1
for each col, calculate the num of sets for each digits.
"""
def diffSum(N, B):
    def digits(N, B):
        l = []
        n = N
        while n > 0:
            l.append(n%B)
            n = n/B
        l.reverse()
        print l
        return l

    # get the B representation
    digitsl = digits(N,B)

    # loop thru each digit
    for i in xrange(len(digitsl)):
        for u in xrange(B):
            for v in xrange(B):  # vertical consider max B carry from prev col, digitsl[i]-v
                for carry in xrange(u):
                    for w in xrange(((digitsl[i]-v+B)%B), B*B, B):  # horizontal
                        pass

def testdiffSum():
    N = 50
    B = 4
    assert diffSum(N,B) == 4


"""
Energy balance...you inject 2 balls at pos x, energy is 2x, after popping,(x-1), (x+1), still 2x
Based on that, we know where the ball will be after injection.
"""
def hotdogStand(corner):
    nhotdogs = sum(v for k,v in corner.items())
    '''keep this sorted corner in sync with corner dict!!!'''
    sortedcornerl = sorted([[k,v] for k,v in corner.items()], key=lambda x: x[1])
    cornerl = [0]*2000000

    def getCornerl(k, cornerl):
        return cornerl[k+1000000]
    def setCornerl(k,v,cornerl):
        cornerl[k+1000000] = v

    for k,v in corner.items():
        setCornerl(k,v,cornerl)
    print corner
    print sortedcornerl

    def insert(l, k, v):
        idx = bisect.bisect_left([e[1] for e in l], v)
        l.insert(idx, [k, v])

    def delete(l, k, v):
        idx = bisect.bisect_left([e[1] for e in l], v)
        while l[idx][0] != k:
            idx += 1
        assert l[idx][0] == k
        assert l[idx][1] == v
        del l[idx]

    def moveToNB(e, corner, sortedcornerl):
        k,v = e
        if v <= 1:
            return 0

        moves = (v-1)/2 if v%2 else v/2
        corner[k] -= 2*moves
        #e[1] = corner[k]
        #setCornerl(k, corner[k], cornerl)
        #if corner[k] == 0:
        #   del corner[k]

        try:
            v = corner[k+1]
            corner[k+1] += moves
            if v > 1: # if v less than 1, means it already been popped last time.
                delete(sortedcornerl, k+1, corner[k+1]-moves)
        except KeyError:
            corner.update({k+1:moves})
        #setcornerl(k+1, corner[k+1], cornerl)
        insert(sortedcornerl, k+1, corner[k+1])

        try:
            v = corner[k-1]
            corner[k-1] += moves
            if v > 1:
                delete(sortedcornerl, k-1, corner[k-1]-moves)
        except KeyError:
            corner.update({k-1:moves})
        #setcornerl(k-1, corner[k-1], cornerl)
        insert(sortedcornerl, k-1, corner[k-1])

        return moves

    print '-----before move ---------'
    print sortedcornerl
    counts = 0
    while len(sortedcornerl) > 0:
        e = sortedcornerl.pop()
        print 'moving :', e
        counts += moveToNB(e, corner, sortedcornerl)
    print '-----after move ---------'
    print corner
    print 'min moves:', counts

def testHotdogStand():
    ncorners = 3
    corner = {}
    corner[-1] = 2
    corner[0] = 1
    corner[1] = 2
    hotdogStand(corner)

    corner.clear()
    l = [651593, 457, 652566, 1221, 653472, 119, 653642, 205, 654665, 1806, 655880, 46, 655932, 56, 656320, 1243, 658894, 2393, 660116, 126, 660213, 42, 660526, 436, 661112, 238, 661517, 556, 662397, 801, 663137, 47, 663385, 825, 664205, 170, 664895, 698, 665372, 46, 665780, 308, 666259, 303, 666623, 269, 666999, 158, 667158, 14, 667169, 9, 667197, 27, 667556, 411, 668088, 128, 668112, 165, 668172, 200, 668614, 361, 668910, 146, 668966, 158, 669306, 22, 669345, 21, 669379, 94, 669771, 205, 669945, 111, 670305, 485, 670866, 498, 671244, 11, 671863, 1311, 673301, 1099, 674169, 213, 674404, 721, 674730, 785, 677233, 1792, 677974, 31, 677995, 12, 678563, 631, 679493, 770, 680186, 116, 680303, 121, 681238, 1700, 682338, 13, 682352, 8, 682361, 7, 682373, 10, 682905, 612, 683469, 379, 684204, 485, 684426, 148, 684970, 182, 685584, 1037, 686284, 168, 686377, 293, 687004, 500, 687923, 981, 688696, 374, 689163, 35, 689370, 25, 689383, 8, 689397, 9, 689585, 66, 689794, 390, 690300, 59, 690351, 25, 690639, 763, 691700, 590, 691976, 363, 692241, 78, 693059, 311, 693149, 991, 694331, 483, 694748, 202, 695137, 78, 695193, 64, 695397, 98, 695477, 238, 696810, 2018, 697957, 143, 698070, 151, 698281, 253, 699235, 902, 699559, 363, 700200, 57, 700211, 51, 700214, 51, 700243, 47, 700527, 250, 700967, 218, 701272, 336, 701973, 568, 702527, 43, 702583, 44, 703064, 577, 703714, 330, 704452, 388, 704571, 311, 705170, 166, 705986, 1003, 706705, 15, 706727, 154, 707067, 490, 707349, 662, 708685, 802, 708992, 87, 709177, 223, 709469, 162, 709545, 20, 709798, 130, 710362, 539, 710954, 204, 711000, 63, 711233, 19, 711250, 24, 711300, 23, 711330, 124, 711575, 159, 711801, 296, 712391, 307, 713193, 1130, 713576, 108, 714063, 99, 714358, 514, 714887, 206, 715704, 1176, 716720, 331, 717016, 156, 717271, 326, 717587, 197, 717712, 19, 717743, 36, 717807, 30, 717850, 24, 718401, 655, 719526, 999, 720751, 981, 721550, 389, 722123, 574, 722634, 174, 723090, 548, 723964, 648, 724815, 989, 725643, 172, 725792, 104, 725942, 20, 725960, 71, 726378, 601, 726907, 87, 726965, 89, 727930, 1360, 728888, 205, 728910, 179, 729162, 210]
    for i in xrange(0, len(l), 2):
        corner[l[i]] = l[i+1]
    print corner
    hotdogStand(corner)

"""
bacteria, pushing down towards southeast.
repeat extends with a dfs. Using global maxx, maxy to record the global max.
"""
def Bacteria(l):
    maxx = maxy = -1 # one per tree, set before tree traverse
    minxy = 1E10
    visited = [0]*len(l)

    def overlap(ax1,ay1,ax2,ay2,bx1,by1,bx2,by2): # the smaller one larger than the bigger one
        return True if not (ax1 > bx2 or bx1 > ax2 or ay1 > by2 or by1 > ay2) else False
    def overlapByExtension(ax1,ay1,ax2,ay2,bx1,by1,bx2,by2):
        return overlap(ax1,ay1,ax2,ay2,bx1,by1,bx2,by2) and (ax2 != bx1 or ay2 != by1) and (ax1 != bx2 or ay1 != by2 )

    def dfs(l, box, visited):
        if visited[box]:
            return
        visited[box] = True
        minxy = min(minxy, l[box].x1+l[box].y1)
        maxx = max(maxx, l[box].x2)
        maxy = max(maxy, l[box].y2)

        for i in len(l):
            if overlapByExtension(box.x1,box.y1, box.x2, box.y2, l[i].x1, l[i].y1, l[i].x2, l[i].y2):
                dfs(l, i, visited)


    sortedl = sorted(l, key=lambda x: x[2], reverse=True)
    print sortl

    maxtime = 0
    for i in xrange(len(l)):
        if not visited(i):  # this is a new tree, init max
            maxx = maxy = -1 # one per tree, set before tree traverse
            minxy = 1E10
            dfs(l, i, visited)
            maxtime = max(maxtime, maxx+maxy-minxy+1)

def testBacteria():
    # X1 Y1 X2 Y2, [x1-x2][Y1-Y2]
    rectangles = [[5,1,5,1],[2,2,4,2],[2,3,2,4]]
    Bacteria(rectangles)

"""
Travel plan, build matrix of dist between each pair. find max(dist[u,v], ...)
the num of roundtrip over a seg is min(Seg_idx, max-seg_idx/2+1)
the adj segs can only be +1,0,-1 three choices. N segs, search space is 3^N.
kind of brutal force.
"""
def TravelPlan(l, F):
    # create segments lists
    segs = []
    size = len(l)
    TravelPlan.tmpmax = 0    # function.class_static_var
    for i in xrange(1,size,1):
        segs.append(l[i]-l[i-1])

    def segMaxRT(i, segs):  # restraint by the min offset to any of the ends.
        return i+1 if i+1<=len(segs)/2 else (len(segs)-(i+1))+1

    def DP(segs, prevsegRT, cursegidx, curtot):
        if cursegidx == len(segs)-1:
            tot = curtot + 2*segs[cursegidx]
            if tot < F and tot > TravelPlan.tmpmax:
                TravelPlan.tmpmax = tot
                print 'Fuel cost: ', tot
                return tot
        else:
            maxRT = segMaxRT(cursegidx, segs)
            if prevsegRT + 1 <= maxRT:
                space = [ prevsegRT+1, prevsegRT, prevsegRT-1]
            else:
                space = [maxRT, maxRT-1]

            for curRT in space:
                curtot += 2*curRT*segs[cursegidx]
                DP(segs, curRT, cursegidx+1, curtot)

    ''' start from the first seg, recursive down to last seg'''
    DP(segs, 1, 1, 2*segs[0])

def testTravelPlan():
    l = [0,1,2,3,4,5]
    F = 13
    #l = [0, -608608325, 164783780, -412357447, 870213391, 707507710, 367416524, -486243523, 798236571, -953791685, 393251122, -652564233, -631613140, 951970489, -746722987, -521852085, 633774659, -459886235, -78618105, 270483090, 103999737, -402213368, 674287741, 599673309, -564458930, 293979142, -363225457, 561420312, -671674482]
    #F = 4786364515
    TravelPlan(l, F)


"""
candyStore, coin change problem, with just greedy algorithm
a matrix, fixed col of K diff items, C rows with each row for one person.
when adding a person, ensure C types are available, by sum/C-unit
"""
def CandyStore(K,C):
    total = 0
    l = []
    for i in xrange(1,C+1,1):  # test each type has the num of coins.
        num = K - (total/i)
        total += i*num
        if num:
            l.append(num)
    print sum(l), l

def testCandyStore():
    K = 2
    C = 50
    CandyStore(K,C)

"""
letter stamper, palindrom, simplified with a stack.
"""
def Stamper(s):
    def suffix(s):
        l = []
        for i in xrange(len(s)):
            l.append(s[i:])
        l.sort()
        print l
def testStamper():
    s='ABCCBA'
    Stamper(s)

"""
wifi tower, a selection problem with P projects Q equips.
Let Pexcl be the set of projects _NOT_ selected and Q be the set of equipments purchased.
max = sum(P) - sum(Pexcl) - Q = sum(P) - min(sum(Pexcl) + Q)

Max-flow equals minimum cut of edges with capacity in edge.

Min-cut means cut edges with smallest sum of ResC
then we know the max possible flow out of one subgraph, into cut subgraph.

First, convert src->project edge caps with profit, equip->sink edge caps with cost.
MaxProfit is sum(all_project) - minCut(src-profile-cost-sink).
find the min-cut of the flow, and the max profit is sum(profit(pi)) - minCut.
http://en.wikipedia.org/wiki/Max-flow_min-cut_theorem

EdmondsKarp_algorithm: many iterations of BFS, each BFS saturate one edge. it is a min-cut edge.
1. residual_Cap of an Edge(u->v) = tot_cap(u->v) - flow_already_used(u,v).
2. for each directed edge(u->v), init residual Cap = tot_Cap(u->v).
3. each BFS find a path from src-sink, with one edge saturated. ResC of this edge is 0, it is a min-cut edge.
4. the path-flow of this BFS is the in-bound flow to sink, as the flow is throttled at each node and carry on down to the sink.
5. reduce the residual cap of all the edges along the path with path-flow of this BFS iteration.
6. ResidualCap(u->v): the residualCap(u->v) reduced by the flow cap.
7. ResidualCap(v->u): At the same time, this flow makes room for a possible flow back from v->u, hence, resC(v->u) += this flow cap.  ResC(v->u): undo the flow sent from u->v
8. Augment path find by BFS with ResC graph augs the max flow.
9. augment_path: path in resc graph, check resc bottleneck. BFS when(not colored and ResC[u->v] > 0)
9. while existing aug path, BFS find aug path, maxflow += min(aug_path)
10. after BFS done, find out all edges in Residual graph where ResC=0, those edges are min-cut edges.
11. start from src, after rm min-cut edges with 0 ResC, left over edges are max flow out edges!!
"""
def WifiTower(G):
    numnodes = len(G)
    src, sink = 0, numnodes-1
    mincut = []

    def maxFlow(G):
        # repeatly call bfs to find path from src to sink, record 
        color = [0]*numnodes
        Path = [0]*numnodes      # parent backtracking tab
        ResC = copy.deepcopy(G)  # global state updated by each bfs run.
        pathMinCapacity = [0]*numnodes

        ''' each call of BFS use a deq and node color to find a path from src to sink
            update the maxflow cap data structur(passed in) during bfs to find the min.
        '''
        def BFS(src, sink, G, pathMinCapacity):
            for i in xrange(numnodes):  # clean up bookkeeping before a round of BFS
                pathMinCapacity[i] = sys.maxint   # set path min cap to max when start
                color[i] = 0
                Path[i] = -1   # init parent pointer, global, bad!!

            Q = collections.deque()
            Q.append(src)
            while len(Q) :
                curnode = Q.popleft()
                for v in xrange(numnodes):    # for each nb of cur node reachable in Residual Graph
                    if ResC[curnode][v] > 0:  # Aug path is a path in Residual Graph.
                        # aug path in ResG exists, aug flow with ResC >0 residual cap means reachable
                        if color[v] == 0 and ResC[curnode][v] > 0 :
                            color[v] = 1
                            Path[v] = curnode  # curnode is the parent of child v
                            # node cap throttled by the parent's cap, or the in-bound edges cap
                            pathMinCapacity[v] = min(pathMinCapacity[curnode], ResC[curnode][v])
                            Q.append(v)
                            print curnode, v, ResC[curnode][v], Path
                            if v == sink:  # find a path finally to the end.
                                return True  # break once a path is found.
            return False

        # global data passed in as arg to be updated during BFS
        maxflow = 0
        while BFS(src, sink, G, pathMinCapacity):
            maxflow += pathMinCapacity[sink]   # path min is the sink's cap.
            print Path, pathMinCapacity[sink]
            v = sink   # from sink, find path parent back to src, deduct capacity of this path
            while v != src:
                parent = Path[v]
                ResC[parent][v] -= pathMinCapacity[sink]   # ResC[parent-v] reduced by the cap this flow took
                ResC[v][parent] += pathMinCapacity[sink]   # ResC[v->parent] undo the flow. incred by the cap this flow took
                print "path from", Path[v], " to ", v, " pathmin: ", pathMinCapacity[sink]
                # when resid capacity is 0, must be a min cut. 
                if ResC[parent][v] == 0 and G[parent][parent] > 0:
                    mincut.append([parent, v])
                v = parent

        print "maxflow =", maxflow
        return maxflow

    maxprofit = 0
    for v in xrange(numnodes):
        maxprofit += G[0][v]
    print "sum of all profit =", maxprofit

    mincut = maxFlow(G)
    print 'Best profit=', maxprofit - mincut

def testWifiTower():
    G = [[0]*7 for i in xrange(7)] # 5 nodes, 0 is src, 6 is sink
    G[0][1] = G[0][2] = G[0][4] = 10 # src to node, profit 10
    G[3][6] = 15
    G[5][6] = 20
    G[1][3] = G[2][3] = G[4][3] = G[4][5] = 1<<30  # project deps on equip cost
    WifiTower(G)

"""
DiceGame, count # of ways throwing n dice with n diff sides.
Permutation with repetition. C(n, n1,n2,..nk) = n!/n1!*n2!*n3!
Set with repetition: sort dice with # of sides, the val of next can only be > prev
tab[dice][faceval] = no. of ways for adding this new dice at faceval, after adding this new dice.
is the sum of prev dice, faceval -> 1, sum(tab[i-1][k=1...j])
http://www.topcoder.com/tc?module=Static&d1=tutorials&d2=combinatorics
"""
def DiceGames(l):
    l.sort()
    size = len(l)
    maxval = l[-1]
    tab = [[0]*(maxval) for i in xrange(size)]

    # init the first dice's to be 1
    for i in xrange(l[0]):
        print i
        tab[0][i] = 1   # tab[i,j] is the possible counts contributes by ith dice, with jth faces.

    for i in xrange(1, size, 1):   # for each dice
        for j in xrange(l[i]):     # for each val in side of that dice
            for k in xrange(j+1):  # the new value is the sum of all prev values. Not a simple max or min.
                tab[i][j] += tab[i-1][k]  # sum of all possible pre-k values. based only pre row(i-1)
    print tab

    # total is the summation of the last line(n dices) with all faces.
    sum = 0
    for i in xrange(maxval):
        sum += tab[size-1][i]   # all the possible contributions count sum.
    print 'DiceGames:', sum

def testDiceGames():
    l = [3, 4]  # dice array with ith dice have l[i] sides
    l = [4,5,6]  # dice array with ith dice have l[i] sides
    DiceGames(l)

"""
prob : P(n+1,i) = sum(P(n, i+1), P(n,i+2), ...)
http://www.topcoder.com/tc?module=Static&d1=tutorials&d2=probabilities
"""
def NestRand(depth):
    def fn(N):
        baseprob = 1.0/N
        l = [baseprob]*N
        for i in xrange(1, depth, 1):
            for j in xrange(N-i):
                l[j] = 0  # reset to 0 before taking sum of probs.
                for k in xrange(j+1, N-i+1, 1):
                    l[j] += l[k]*(1.0/k)
        print l
    return fn

def testNestRand():
    fn = NestRand(2)  # 2 level nest
    fn(4)

"""
subsetSum, subsetMax, BadNeighbors
http://www.topcoder.com/stat?c=problem_statement&pm=2402&rd=5009
The trick here is that the last one also subject to constraint during DP.
we need compute two values, (with first, exclude last), and (with last, exclude first
"""
def SubsetSumMax(l):
    size = len(l)
    tab = [[0]*size for i in xrange(size)]
    sol = [0]*size

    def DP(l, tab, pos, sol):
        if pos < 2:
            tab[pos] = max(l[0], l[1])
            #tab[pos] = l[pos]
            sol.append(l[0]) if l[0] > l[1] else sol.append(l[1])
        tab[pos] = max(tab[pos-2]+l[pos], tab[pos-1])
        if tab[pos-2]+l[pos] > tab[pos-1]:
            sol.append(l[pos])

    ''' tab[start][end], the max sum from start to end'''
    def DP2(l, tab, start, pos, sol):
        for i in xrange(size):
            sol[i] = 0
        tab[start][start] = sol[start] = l[start]  # special treat to first two
        tab[start][start+1] = sol[start+1] = l[start+1]

        for i in xrange(start+2, pos+1):  # max value includes ith or not
            tab[start][i] = max(tab[start][i-2] + l[i], tab[start][i-1])
            sol[i] = max(sol[i-2] + l[i], sol[i-1])
        return sol[i]  # this is the bottom up final max solution at last iteration.

    start = DP2(l, tab, 0, size-2, sol)
    end = DP2(l, tab, 1, size-1, sol)

    print "include first :" ,start
    print "include last :" ,end

def testSubsetSumMax():
    #l = [10, 3, 2, 5, 7, 8]
    l = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]  # ans: 16
    # ans: 2926
    l = [94, 40, 49, 65, 21, 21, 106, 80, 92, 81, 679, 4, 61, 6, 237, 12, 72, 74, 29, 95, 265, 35, 47, 1, 61, 397, 52, 72, 37, 51, 1, 81, 45, 435, 7, 36, 57, 86, 81, 72 ]
    SubsetSumMax(l)

"""
Give a string of numbers, add minimum + in between so sum of a given number.
Classic backtracking problem. Given you have a target sum, top down.(100->0)
At each stage, extend solution, iterate solution space, recursion over each branch.
http://www.cse.ohio-state.edu/~gurari/course/cis680/cis680Ch19.html#QQ1-51-128
Example:"382834" => 38+28+34 = 100
http://www.topcoder.com/stat?c=problem_statement&pm=2829&rd=5072
"""
def QuickSum(l, sum):
    #l = list(l)
    rows = len(l)
    table = [[-1]*(sum+1) for i in xrange(rows)]

    def decisionDP(l, idx, sum, table):
        #print "DP=", idx, l[idx], sum
        if int(l[idx]) > sum:
            return False, None

        if table[idx][sum] >= 0:
            print l[0:idx], sum, table[idx][sum]
            return True, table[idx][sum]

        for start in xrange(idx,-1,-1): # idx the last idx, range need +1, anything after : need offset one more
            if int(l[start:idx+1]) == sum:
                table[idx][sum] = 0
                print l[start:idx+1], sum, table[idx][sum]
                return True, 0
            elif int(l[start:idx+1]) < sum:
                #print 'Recusion:', l[start:idx+1], sum, sum-int(l[start:idx+1])
                ret, signs = decisionDP(l, start-1, sum-int(l[start:idx+1]), table)
                if ret:
                    table[idx][sum] += (signs+1)
                    print 'sum:', l[start:idx+1], sum, table[idx][sum]
        return False, None

    def dp(l, stop, sum, table):
        #print 'dp:', stop, sum, l
        if sum == 0 and stop <= 0:
            print 'found:  idx:', stop, 'num:',l[:stop]
            return 0, 0
        elif stop <= 0:
            return -1, -1
        elif int(l[:stop]) == sum:
            print 'found:  idx:', i, 'num:',l[:stop]
            return int(l[:stop]), i
        for i in xrange(stop, -1, -1):
            curnum = int(l[i:stop+1])  # end is open.
            #print 'cur :', curnum, 'start', i, 'stop:', stop, 'sum', sum
            if curnum <= sum:
                subval, subidx = dp(l, i-1, sum-curnum, table)
                if subval >= 0:
                    print 'stop:', stop, 'sum:', sum, 'idx:', i, 'num:',curnum, 'subval', subval
                    return curnum, i
        return -1, -1

    print '--- starting---', l, sum
    dp(l, len(l)-1, sum, table)
    #dp(l, len(l)-1, sum, table)
    decisionDP(l, len(l)-1, sum, table)


def testQuickSum():
    s = '382834'
    sum = 100
    print '--- testing QuickSum with: ', s, sum
    QuickSum(s,sum)

"""
http://www.topcoder.com/stat?c=problem_statement&pm=1861&rd=4630
make base into a palindrome by adding min letters
for pivot in each pos piv from 1->N, left_str: piv->0, right_str: piv->n,
the lcs(left_str, right_str) is the num of letters to be inserted.
First, do lcs(str, reverse(str)), then lcs(left_str, right_str)
forms a right bottom box in the origin tab, and lcs of that box
is the num of a[i]=reverse(a)[j] increase point in the box.
"""
def ShortPalindromes(s):
    def lcs(row, col):
        tab = [[0]*len(col) for i in xrange(len(row))]
        for i in xrange(len(row)):
            for j in xrange(len(col)):
                prerow = i-1 if i-1 >= 0 else 0
                precol = j-1 if j-1 >= 0 else 0
                if row[i] == col[j]:
                    if i == 0 or j == 0:  # for first row,col, max is 1
                        tab[i][j] = 1
                    else:
                        tab[i][j] = tab[prerow][precol] + 1  # lcs[i,j]=lcs[i-1,j-1]+1
                else:
                    tab[i][j] = max(tab[prerow][j], tab[i][precol]) #lcs[i,j]=max(l[i-1,j],l[i,j-1]
        return tab

    rs = s[::-1]
    tab = lcs(s, rs)
    print tab

def testShortPalindromes():
    s = 'ALRCAGOEUAOEURGCOEUOOIGFA' # return AFLRCAGIOEOUAEOCEGRURGECOEAUOEOIGACRLFA
    s = 'ROPCODER'
    ShortPalindromes(s)

"""
miniPaint, min strokes
http://www.topcoder.com/stat?c=problem_statement&pm=1996&rd=4710
for each row, for each color, find the max window this color can cover with one strike.
this is the same as max sum seq problem.
for strike from 1 to n, mispaint = (len-1th_max_color_win)
round i: mispaint -= (ith_max_col_win - hole)
"""
def MiniPaint(tab):
    pass
def testMiniPaint():
    tab = ""
    MiniPaint(tab)

"""
http://code.google.com/codejam/contest/dashboard?c=32010#s=p1
min cost for full cover. the section without overlap must be present.
one dimen X-axis line sweep to find the overlap.
Priqueue storing events to entering/leaving lines.
BSTree to store current ActiveSet<seg> sorted by Y-coord
"""
import priqueue   # both under my $HOME/wip/pydev
import rbtree
def PaintingFence(l):
    size = 10000
    lines = len(l)
    q = priqueue.PriQueue()
    tree = rbtree.RBTree()
    lines = len(l)
    sol = []

    def enteringLineEvent(event):  # when key is start point, the event is entering event.
        return True if event.key == event.value[1] else False

    '''populate the event priqueue'''
    for i in xrange(lines):
        seg = l[i]
        q.insert(priqueue.PQNode(l[i][1], seg)) #  key is start point, val is seg itself.
        q.insert(priqueue.PQNode(l[i][2], seg)) #  key is end point, val is seg itself

    '''extract each event'''
    event = q.extract()
    while event :
        #print event.key, event.value
        if enteringLineEvent(event):
            ''' prev seg has no overlap'''
            if not tree.has_key(event.key) and len(tree) == 1:
                prevseg = tree.values()[0]  # only one seg left
                sol.append(prevseg)
            tree.setdefault(event.key, event.value)  # calls tree.insert(key,val)
        else:
            ''' current seg has no overlap'''
            tree.pop(event.value[1])   # when inserted, the key was start.
            if tree.is_empty():
                sol.append(event.value)
        event = q.extract()

    print sol

def testPaintingFence():
    offer = [ ['BLUE',1,6000], ['RED',2000,8000], ['WHITE', 7000, 10000] ]
    offer = [ ['BLUE',1,6000], ['RED',4000,10000], ['WHITE', 3000, 8000] ]
    PaintingFence(offer)

"""
http://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
listing all crossings of a set of line segments.
A bin search tree of ActiveSet Tree<segs>, A PriQueue<Event> of future event.
"""
import priqueue   # both under my $HOME/wip/pydev
import rbtree
def BentleyOttmann(l):
    q = priqueue.PriQueue()
    activeset = rbtree.RBTree()
    lines = len(l)
    sol = []

    def enteringLineEvent(event):  # whether event is entering line event
        return True if event.key == event.value[1] else False

    '''populate the event priqueue'''
    for i in xrange(lines):
        seg = l[i]
        q.insert(priqueue.PQNode(l[i][1], seg)) #  the end point of each line.
        q.insert(priqueue.PQNode(l[i][2], seg)) #  the end point of each line.

    '''extract each event'''
    event = q.extract()
    while event :
        print event.key
        if enteringLineEvent(event):
            pass
        else:
            pass
        n = q.extract()

def testBentleyOttmann():
    offer = [ ['BLUE',1,6000], ['RED',2000,8000], ['WHITE', 7000, 10000] ]
    BentleyOttmann(offer)

"""
min strokes for a stripe pattern
http://www.topcoder.com/stat?c=problem_statement&pm=1215&rd=4555
for each subsection, find the min, the global min is the min of each position.
"""
def StripePainter(l):
    tab = [[0]*len(l) for i in xrange(len(l))]

    def segments(start, end, ch):  # all index are inclusive
        segs = []  # an array of all the segments
        cur = start

        while cur < end:
            while cur < end and l[cur] == ch:
                cur += 1
            next = cur
            while next < end and l[next] != ch:
                next += 1
            if next > cur:
                segs.append((cur, next-1))
            cur = next
        return segs

    def DP(start, end, tab):  # inclusive indices
        if end < 0 or start >= len(l):
            return 0

        if start == end:  # boundary condi
            tab[start][end] = 1
            return 1      # no stroke

        if tab[start][end] > 0:
            return tab[start][end]

        ''' search space is fixed, search sequentially'''
        tab[start][end] += 1
        segs = segments(start,end,l[end])
        for seg in segs:
            print 'seg', start, end, l[end], seg
            tab[start][end] += DP(seg[0],seg[1], tab)
        print 'tab', start, end, tab[start][end]
        return tab[start][end]

    ''' partial sol ending ASC, for each partial sol, [0, ith] pos is the search space'''
    for i in xrange(1,len(l),1):
        segs = []
        ch = l[i]

        sol = 1
        segs = segments(0,i,ch)
        for seg in segs:
            sol += DP(seg[0], seg[1], tab)

        print "sol.i=",i, l[i], sol

    print "best solution: ", sol
    return sol

def testStripePainter():
    #l = 'AABC'
    #l = "ABACADA"
    l = "AABBCCDDCCBBAABBCCDD"  # sol=7
    l = "BECBBDDEEBABDCADEAAEABCACBDBEECDEDEACACCBEDABEDADD" # sol=26
    StripePainter(l)

"""
Perfect Matching of bipartite graph
http://www.wikihow.com/Use-a-Bipartite-Graph-to-Find-the-Maximal-Matching
create a path from an unmatched vertex from left, to an unmatched dot on the right side.
consider all paired edge travel from right to left, while all unpaired edge travel left to right.
"""
def BipartiteByTwoColorBFS(G):
    for v in G.items():
        if color[v] is uncolored:
            color[v] = white
            bfs(G, v)
    def bfs(G, v):
        q.enqueue(v)
        while len(q) > 0:
            v = q.deque()
            v.processed = True
            for nb in G[v].neighbor:
                nb.discovered = True
                if color[nb] is color[v]:
                    bipartite = False
                    break
                else:
                    color[nb] = ~color[v]
                q.enqueue(nb)

    ''' to compare with dfs'''
    def dfs(G, v, parent, level):
        color[v] = discovered
        v.level = level
        for nb in v.neighbor:
            if nb.discovered == False and nb.processed == False:
                nb.parent = v
                v.children.append(nb)
                dfs(G, nb, v, level+1)
        v.processed = True

''' perfect match with augment path: dfs until stop at any unpaired rhs
    every dfs scan will add one augment path. done when no more augment
'''
def PerfectMatching(G):
    rG = {}
    match = []
    # build reverse G, index from 1-A
    for k, v in G.items():
        for rhk, rhv in v.items():
            # k = A rhk=1 rhv=0
            rG.setdefault(rhk, {}).update({k:rhv})

    ''' find all edges from lhs to rhs, or from rhs to lhs.
        return (lhs vertex, rhs vertex, rhs vertex value attr)'''
    def findAllEdges(lhs, rhs):  # edge always denoted by (lhs, rhs)
        if lhs is not None:
            for k, v in G[lhs].items():
                yield lhs, k, v
        elif rhs is not None:
            for k, v in rG[rhs].items():
                yield k, rhs, v

    def vertexPaired(vertex):
        if vertex in G.keys():
            for k,v in G[vertex].items():
                if v == 1:
                    return True
        if vertex in rG.keys():
            for k,v in rG[vertex].items():
                if v == 1:
                    return True
        return False

    ''' dfs search from lhs and stop at any unpaired rhs lhs is the lhs vertex.
        update partial solution edges/color list during dfs.
        color is discovered, processed. Here it just remember which vertex has been visited.
    '''
    def DFS(lhs, rhs, edges, color):
        print 'DFS searching : start ', lhs, ' rhs: ', rhs, ' path edges:', edges, color
        ''' find all vertex reached from lhs vertex'''
        if lhs is not None:
            # for each edge from lhs vertex, (lhs vertex, rhs vertex, rhs vertex attr)
            for lc, rc, v in findAllEdges(lhs, None):
                if not vertexPaired(rc):
                    edges.append((lc, rc))
                    print edges
                    return True    # dfs find a free node in rhs, done !
                else:  # recursive dfs; this time from rhs, find all lhs vertex
                    if not rc in color:
                        color.append(rc)
                        if DFS(None, rc, edges, color):  # if recursive found
                            edges.append((lhs, rc))
                            print edges
                            return True

        ''' find all vertex reached from rhs vertex to lhs'''
        if rhs is not None:
            for lc, rc, v in findAllEdges(None, rhs):
                if not lc in color:
                    color.append(lc)
                    if DFS(lc, None, edges, color):
                        edges.append((lc, rhs))
                        print edges
                        return True
        return False

    ''' first, init a mapping with the first edge of each node, augment path will fix it'''
    for k, v in G.items():
        for rhk, rhv in v.items():  # v = {'1': 0, '2': 0, '4': 0}
            # k=A rhk=1 rhv=0 False
            print k, v, rhk, rhv, vertexPaired(rhk)
            if rhv == 0 and not vertexPaired(rhk):
                G[k][rhk] = 1   # G[A] = {'1':0, '2': 0, '4': 0}
                rG[rhk][k] = 1
                match.append((k,rhk))
                break
    print G
    print rG
    print match

    ''' for each unpaired vertex lhs, find a path to another unpair vertex in rhs '''
    for k,v in G.items():
        if not vertexPaired(k):  # find an unpaired vertex in lhs
            edges = []
            color = []
            if DFS(k, None, edges, color):  # find an path from unpaired lhs settle down at unpaired rhs
                print 'match:', match
                print 'unpaired path:', edges
                for e in edges:
                    if e in match:
                        match.remove(e)
                    else:
                        match.append(e)
                        G[e[0]][e[1]] = 1
                        rG[e[1]][e[0]] = 1
                print match

def testPerfectMatching():
    # why the fuck I do not have document ?
    # G, lhs [A B C D E], rhs [1 2 3 4 5]
    G = {
        'A':{'1':0, '2':0, '4':0},
        'B':{'1':0 },
        'C':{'2':0, '3':0},
        'D':{'4':0, '5':0},
        'E':{'3':0 }
        }
    PerfectMatching(G)

"""
BFS search the space with backtracking
http://www.topcoder.com/stat?c=problem_statement&pm=3935&rd=6532
"""
def SmartWordToy(start, end, l):
    text = 'abcdefghijklmnopqrstuvwxyz'
    forbid = []
    SmartWordToy.min = sys.maxint

    def constraint(l):
        if len(l) == 0:
            yield ""
            return
        for e in l[0]:
            for word in constraint(l[1:]):
                yield e+word

    def getAdjChar(ch):
        pos = text.find(ch)
        return text[pos-1], text[(pos+1)%len(text)]

    def generateWord(word):
        for pos in xrange(len(word)):
            if word[pos] == end[pos]:  # stop here...otherwise, infinitely loop
                continue
            for ch in getAdjChar(word[pos]):
                yield word[0:pos]+ch+word[pos+1:]

    def debug(word, move, nextword=None):
        if word in ['bzaa', 'bzzz', 'bzza' ]:
            print 'from:', word, ' moves:', move, 'to:', nextword

    def BFS(startword, count):
        Q = collections.deque()
        Q.append((startword, count))

        while len(Q):
            word, moves = Q.popleft()
            #debug(word, moves)
            if word == end:
                if moves < SmartWordToy.min:
                    SmartWordToy.min = moves
                    return moves

            for nextword in generateWord(word):  #need color to prevent circle
                if not nextword in forbid:
                    ''' moves += 1  !!! You can not cumulate !!!! '''
                    Q.append((nextword, moves+1))
                    #debug(word, moves, nextword)

    ''' first, populate forbid words'''
    for row in l:
        for word in constraint(row):
            forbid.append(word)
    ''' BFS search '''
    BFS(start, 0)
    print "min moves:", SmartWordToy.min
    return SmartWordToy.min

def testSmartWordToy():
    start = 'aaaa'
    end = 'zzzz'
    l = []
    constraint = ["a a a z", "a a z a", "a z a a", "z a a a", "a z z z", "z a z z", "z z a z", "z z z a"]
    constraint = ['cdefghijklmnopqrstuvwxyz a a a',
 "a cdefghijklmnopqrstuvwxyz a a",
 "a a cdefghijklmnopqrstuvwxyz a",
 "a a a cdefghijklmnopqrstuvwxyz" ]

    for e in constraint:
        fe = filter(lambda x: x!= ' ', e)
        fe = e.split()
        l.append(fe)

    print l
    assert SmartWordToy(start, end, l) == 6


"""
http://www.topcoder.com/stat?c=problem_statement&pm=3117&rd=5865
int entree_idx = favorites[p][i] contains the entree indexes that person p likes.
BFS search: enumerate turning 1,2,..n to the left, right, backtracking the cost of each.
"""
def TurntableService(F):
    npeople = len(F)
    Q = collections.deque()
    TurntableService.mincost = sys.maxint
    color = [1]*npeople
    cost = 0

    ''' Q contains entree after rotation, entree at Q[i] match ith ppl favor '''
    def rotateMatchFavorate(F, Q):
        print 'Fav:', F
        print 'Ent:', Q
        print 'Col:', color
        count = 0
        #pdb.set_trace()
        for i in xrange(npeople):
            print Q[i]
            print F[i]
            if color[i] and str(Q[i]) in F[i]:
                color[i] = 0
                print 'after writing', color
                count += 1
        return count

    for p in xrange(npeople):
        Q.append(p)  # !=[0,1,2,3,4]

    ''' the match at the start, 0'''
    if rotateMatchFavorate(F, Q):
        cost += 15

    while color != [0]*npeople:
        steps = 0
        while not rotateMatchFavorate(F, Q):
            Q.appendleft(Q.pop())
            steps += 1

        if steps > npeople/2:
            steps = npeople - steps
        cost += 15+steps+1

    print "total min cost:", cost

def testTurntableService():
    favorate = ["0 004", "2 3", "0 01", "1 2 3 4", "1 1"]
    #favorate = ["4", "1", "2", "3", "0"]
    f = []
    for e in favorate:
        f.append(e.split())
    print f
    TurntableService(f)

"""
http://en.wikipedia.org/wiki/Simplex_algorithm
"""
def Simplex():
    pass

"""
http://www.topcoder.com/stat?c=problem_statement&pm=3080&rd=6518
try each bus stop and select the best.
"""
def BusToCafeteria(Schedule, Walk, Driving):
    buses = len(Schedule)
    BusToCafeteria.hh = 0
    BusToCafeteria.mm = 0
    end = 14*60+30  # 2:30

    def PrevSchedule(offset, time):
        minutes = time%60
        if minutes < offset:
            return -1, 60-offset
        else:
            return 0, int((minutes-offset)/10)*10+offset

    for i in xrange(buses):
        starttime = end - Driving[i]
        hh, mm = PrevSchedule(Schedule[i], starttime)
        hh = starttime/60-hh
        if mm < Walk[i]:
            hh -= 1
            mm += 60
        mm -= Walk[i]
        if BusToCafeteria.hh < hh or (BusToCafeteria.hh == hh and BusToCafeteria.mm < mm):
            BusToCafeteria.hh = hh
            BusToCafeteria.mm = mm
        print 'hh:mm', hh, mm

    return str(BusToCafeteria.hh)+':'+str(BusToCafeteria.mm)

def testBusToCafeteria():
    S = [0,1,2,3,4,5,6,7,8,9]
    W = [11,11,11,11,11,11,11,11,11,11]
    D = [190,190,190,190,190,190,190,190,190,190]
    assert BusToCafeteria(S, W, D) == '11:9'

"""
http://code.google.com/codejam/contest/dashboard?c=32010#s=p3
iterate each stop, there are n ways to put a bus there, branch out n paths if:
1. must take the leftmost bus if dist=P, 2. can not repeat if within dest-k
for each bus that can be placed there forms a path. solution is the total paths at the end.
"""
def BusStops(N,K,P):
    tab = [[0]*K for i in xrange(N)] # tab[i][j] is the no. of way in ith stop where currently bus j stops.
    count = [0]*N
    lastwinidx = N-K

    ''' calculate tab[currentstop][bus 1->k]'''
    def reverseListIndex(l, k):
        for i in xrange(len(l)-1, -1, -1):
            if l[i] == k:
                return i
        return -1

    def DP(path, currentstop, tab):
        #print 'path=', path, 'level=', currentstop
        if currentstop == N-1:
            count[currentstop] += 1   # sum every path reach the leave
            count[currentstop] %= 30031
            print 'level done: path', path, 'count=', count[currentstop]
            return 1  # only one option at last level

        prevP = currentstop - P  # a P slot window
        print path, currentstop, prevP, path[prevP]
        if prevP >= 0 and reverseListIndex(path, path[prevP]) == prevP :
            count[currentstop] += 1  # only one choice
            nextpath = path[:]
            nextpath.append(path[prevP])
            DP(nextpath, currentstop+1, tab)
        else:
            ''' special treatment for slots within the last K window'''
            print 'curstop=',currentstop, 'n-k=', N-K, path
            for bus in xrange(K):
                if currentstop >= lastwinidx: # compare cnt, not idx, hence use N
                    if bus in path[lastwinidx:]:
                        continue
                nextpath = path[:]
                nextpath.append(bus)
                count[currentstop] += 1
                DP(nextpath, currentstop+1, tab)

        print 'path=', path, 'level=', currentstop, 'total=', count[currentstop]
        return count[currentstop]

    for i in xrange(K):  # k bus at first k stop
        tab[i][i] = 1
    initpath = [i for i in xrange(K)]

    DP(initpath, K, tab)
    print 'level=', N-1, ' count=', count[N-1]
    return count[N-1]

def testBusStops():
    l = [5, 2, 3]
    assert BusStops(l[0], l[1], l[2]) == 3
    l = [40, 4, 8]
    assert BusStops(l[0], l[1], l[2]) == 7380


"""
http://code.google.com/codejam/contest/dashboard?c=32011#
constraint: sum of max of 3 fraction < 10k
powerset of all combinations, and check to find the one comb with max.
or, sort fraction A, forEach, forEach fract B, N**2
"""
def Juice(l):
    pass
def testJuice():
    l = [[0 ,1250 ,0],[3000, 0, 3000],[1000, 1000, 1000],[2000, 1000, 2000],[1000, 3000, 2000]]
    Juice(l)

"""
http://www.topcoder.com/stat?c=problem_statement&pm=1259&rd=4493
DP[longest non-decreasing seq], at each stop, loop each prev stop, find the max
tab down each stop, the max leng, and the stop is reached via negative, or positive.
void ZigZag(){
        //int[] a = { 1, 17, 5, 10, 13, 15, 10, 5, 16, 8 };   // result = 7
        //int[] a = { 70, 55, 13, 2, 99, 2, 80, 80, 80, 80, 100, 19, 7, 5, 5, 5, 1000, 32, 32 };  // result=8
        int[] a = { 374, 40, 854, 203, 203, 156, 362, 279, 812, 955,
                600, 947, 978, 46, 100, 953, 670, 862, 568, 188,
                67, 669, 810, 704, 52, 861, 49, 640, 370, 908,
                477, 245, 413, 109, 659, 401, 483, 308, 609, 120,
                249, 22, 176, 279, 23, 22, 617, 462, 459, 244 };   // result = 36

        int[] path = new int[a.length];
        int[][] tab = new int[a.length][2];  // tab[stop][0] > 1, reach stop via asc,
        int maxlen = 0;
        int maxlenprev = 0;
        tab[0][0] = tab[0][1] = 1;  // first stop can be either positive or negative.
        for(int i=1;i<a.length;i++){
            maxlen = 0;
            maxlenprev = 0;
            for(int j=0;j<i;j++){
                if(tab[j][0] > 0){ // idx 0 flag set, reached stop j via ascending
                    if(a[i] < a[j]){  // consider current only when it is less
                        if(tab[j][0] + 1 > maxlen){
                            maxlen = tab[j][0] + 1;
                            maxlenprev = j;
                        }
                    }
                }
                if(tab[j][1] > 0){  // idx 1 flag set, reach stop j via descending
                    if(a[i] > a[j]){  // consider current only when it is large
                        if(tab[j][1] + 1 > maxlen){
                            maxlen = tab[j][1] + 1;
                            maxlenprev = j;
                        }
                    }
                }
            }
            if(a[i] > a[maxlenprev]){ // reach stop i via asc
                tab[i][0] = maxlen;
                path[i] = maxlenprev;
                System.out.println("zigzag: stop:" + i + ":" + a[i] + " taking Prev:" + maxlenprev + ":" + a[maxlenprev]);
            }else{
                tab[i][1] = maxlen;  // reach stop i via desc
                path[i] = maxlenprev;
                System.out.println("zigzag: stop:" + i + ":" + a[i] + " taking Prev:" + maxlenprev + ":" + a[maxlenprev]);
            }
        }
        for(int e : path){
            System.out.println("zigzag:" + a[e]);
        }
        System.out.println("final result:" + tab[a.length-1][0] + " or " + tab[a.length-1][1]);

    }
"""
def ZigZag(l):
    pass
def testZigZag():
    l = [ 1, 17, 5, 10, 13, 15, 10, 5, 16, 8 ]
    l = [ 70, 55, 13, 2, 99, 2, 80, 80, 80, 80, 100, 19, 7, 5, 5, 5, 1000, 32, 32]
    l = [374, 40, 854, 203, 203, 156, 362, 279, 812, 955, 600, 947, 978, 46, 100, 953, 670, 862, 568, 188, 67, 669, 810, 704, 52, 861, 49, 640, 370, 908, 477, 245, 413, 109, 659, 401, 483, 308, 609, 120, 249, 22, 176, 279, 23, 22, 617, 462, 459, 244 ]
    ZigZag(l)

"""
http://people.csail.mit.edu/bdean/6.046/dp/
DP[no-descreasing seq] examples
DP decomposite stages, stage might be item in an array, can also be enumeration of each possible value in the search space.
For example, in knapsack, enumerate all possible wights up to W. in balanced partition, enumerate 1 to max n*k value space.
1. find the search space.
2. enumerate i=1..n inside search space each value.
3. calculate the
"""
def BalancedPartition(l):
    size = len(l)
    left = []
    right = []
    sumleft = sumright = 0
    kickout = 0

    ''' BAD. not a good solution'''
    def adjust(left, ln, right, rn):  # list insert n, and return
        if ln:
            left.append(ln)
        if rn:
            right.append(rn)
        #sumleft = reduce(lambda x,y: x+y, left)
        #sumright = reduce(lambda x,y: x+y, right)
        delta = (sum(left) - sum(right))/2
        if delta > 0:
            idx = bisect.bisect_right(left, delta)
            for i in xrange(idx, -1, -1):
                if left[i] < delta:
                    delta -= left[i]
                    right.append(left.pop(i))
            right.sort()
        elif delta < 0:
            idx = bisect.bisect_right(right, delta)
            for i in xrange(idx, -1, -1):
                if right[i] < delta:
                    delta -= right[i]
                    left.append(right.pop(i))
            left.sort()

    for n in l:
        if sum(left) > sum(right):
            adjust(left, 0, right, n)
        else:
            adjust(left, n, right, 0)

    print 'sum of left:', sum(left), left
    print 'sum of right:', sum(right), right

    ''' DP version, I index into the array[0...i], J index into enumerated value space, value j.
        tab[i,j] == 1 means exists _some_ subset of l[0..i] has sum value of j
        for each item in l, its search value space(possible sum) is max_value * N items.
    '''
    tab = [[0]*len(l)*max(l) for i in xrange(len(l))]
    for i in xrange(len(l)):
        for s in xrange(len(l)*max(l)):  # enum each value point, focus on result, not steps!!
            if l[i] == s:
                tab[i][s] = 1
            elif l[i] < s:  # carry the prev max of either not include l[i], or include l[i]
                tab[i][s] = max(tab[i-1][s], tab[i-1][s-l[i]])
            else:
                tab[i][s] = tab[i-1][s]

    ''' if tab[n][mid] set, means there are some subset has sum mid before ending at idx'''
    mid = sum(l)/2
    if tab[len(l)-1][mid] == 1:
        print "exact partition:", mid
    else:
        for i in xrange(mid):
            print tab[len(l)-1]
            if tab[len(l)-1][mid-i] == 1:
                print "not exact partition:", mid-i
                break
            if tab[len(l)-1][mid+i] == 1:
                print "not exact partition:", mid+i
                break


def testBalancedPartition():
    l = [ 70, 55, 13, 2, 99, 2, 80, 80, 80, 80, 100, 19, 7, 5, 5, 5, 1000, 32, 32]
    l = [ 2, 17, 5, 10, 13, 15, 10, 5, 16, 4 ]
    BalancedPartition(l)

"""
traveling salesman problem with DP.
5 nodes, start from 0, 4 branches out, at node 1 when reaching from 0,(0,1), can branch out to 2,3,4.
DP at each node, BFS all childrens not in path, choose the min child as the next step.
DP return the cost of min next step, or upon path contains all nodes, form the hamilton cycle.
DP tab, parentpath as key, cons nextminpath as value, should be a full path.
final process all paths from start and select the min as the cycle.
"""
def TravelingSalesmanProblem(matrix):
    ''' tab: key=path-prefix, val=[cost, min path to ROW; parent+ROW=full
        e.g, path: 012, min path: 3456
    '''

    ''' cons node into path, return consed path'''
    def cons(a, path):
        path = path+str(a)
        return path

    ''' recursively found and note down child minpath.
        should return minpath directly, not as an output arg.
        the returned nextmin can only hold one return result, which is the best.
    '''
    tab = {} #parentpath as key, cons minpath as value should be full path
    def DP(matrix, src, cur, minpath, parentpath):
        ''' when called, cur node already in parentpath and minpath '''

        '''path as key means, nodes in path already visited with a result,
           cons that result to the result of exploring the ROW
           the result of ROW might already known in the tab keyed by the path
           Note that current node must be the last node in the key!!!
           the routes before reaching cur node does not matter, we are looking
           for the min path starting from the current node
           if cur node is 4, 1234 = 1324 != 2341
        '''
        sortedcurpath = list(parentpath[:-1]) # cur is at the end of parentpath
        sortedcurpath.sort()  # to cur parent path as key must be sorted,
        sortedcurpath.append(cur)
        curpathkey = str(sortedcurpath) # so 123-4 and 132-4 can be reused.
        print "current path:", curpathkey

        # ask 2 questions, empty list(search space), cons result of head into recursive results.
        if(len(parentpath) == len(matrix)):
            tab[curpathkey] = [0]*2
            tab[curpathkey][0] = matrix[cur][src] # what if no direct path from cur->src ?
            # one recursion of cur done, cons found next into the result.
            #minpath.append(cur)   # min path is from cur -> src with cost
            minpath.append(src)  # min path head is the cur node, the path reach here.
            tab[curpathkey][1] = list(minpath)
            print 'full path:', parentpath, ' minpath:', minpath
            return matrix[cur][src]  # close hamilton cycle with final cost back to src

        ''' DP has the result, [cost, [min-path]]'''
        if tab.has_key(curpathkey):
            #pass
            minpath.extend(tab[curpathkey][1])
            print 'tab found parentpath:', parentpath, 'cur:', cur , ' key:', curpathkey, ' return minpath:', minpath
            return tab[curpathkey][0]

        totalcost = 0
        min = sys.maxint
        nextnode = -1
        nextminpath = []
        for i in xrange(len(matrix)-1, -1, -1): # for each node
        #for i in xrange(len(matrix)):
            if i in parentpath:   # prevent circle
                continue

            childminpath = []
            # when into this recursion func, parent path already contains cur head
            parentpath.append(i)
            cost = DP(matrix, src, i, childminpath, parentpath) # recursion child next min path
            parentpath.pop()

            totalcost = matrix[cur][i] + cost #from head to child, plus the min value from child
            if(totalcost < min):
                nextnode = i
                min = totalcost
                nextminpath = list(childminpath)

        '''done explore all children, get the min next path'''
        tab[curpathkey] = [0]*2
        tab[curpathke][0] = min
        # one recursion of cur done, cons next node and its min into the result.
        minpath.append(nextnode)  # cons the found next into result path
        minpath.extend(nextminpath)
        tab[curpathkey][1] = list(minpath)   # need to do a copy when storing into dict
        print 'head:', cur, ' path:', curpathkey, 'nextminpath:', nextminpath, ' minpath:', minpath, 'DP:',  tab[curpathkey]

        return min

    ''' calling DP, will cause BFS DP BFS recursively'''
    path = []
    minpath = []
    start = 5
    path.append(start)
    mincost = DP(matrix, start, start, minpath, path)
    print 'cycle cost:', mincost, minpath

def testTravelingSalesmanProblem():
    matrix = [
        [9999, 27, 81, 36, 80, 50, 16],
        [54, 9999, 21, 33, 77, 53, 99],
        [87, 97, 9999, 74, 75, 87, 55],
        [11, 36, 92, 9999, 58, 27, 101],
        [71, 82, 24, 90, 9999, 47, 2],
        [27, 17, 79, 33, 54, 9999, 11],
        [100, 94, 7, 83, 41, 56, 9999]
    ]
    # ans: [0 6 2 4 5 1 3 0], 206
    assert matrix[0][3] == 36
    TravelingSalesmanProblem(matrix)

''' hackercup 2012 billboard
    http://notes.tweakblogs.net/blog/7524/facebook-hacker-cup-qualification-round-problem-analysis.html
'''
def BillBoard(width, height, s):
    print ' ==== BillBoard ===== '
    words = s.split()
    wordlen  = [len(w) for w in words]
    print words, wordlen

    # with linesize, how many lines can we consolidate to ?
    def calLinesByLen(linesize, wordlen):  # always unit test with single ele list
        lines = 0  # tot lines
        i = 0
        while i < len(wordlen):  # enum all words starts from a new line
            curlinelen = wordlen[i]
            j = i+1  # j starts from i's next
            while j < len(wordlen) and curlinelen + wordlen[j] + 1 < linesize :  # merge to one line
                curlinelen += wordlen[j]  # incr current line sz, move j cursor
                j += 1
            lines += 1
            i = j  # one line break
        return lines

    # enum the length of line, calc how many lines, and get a font size.
    # for each line length, we got a font size, select the max of font size
    maxfontsz = 0
    maxlinelen = 0
    maxlines = 0
    for linelen in xrange(max(wordlen), len(s)):
        totlines = calLinesByLen(linelen, wordlen)
        fontsz = min(height/totlines, width/linelen)
        if fontsz > maxfontsz:
            maxfontsz = fontsz
            maxlinelen = linelen
            maxlines = totlines

    print 'font ', maxfontsz, ' linesize ', maxlinelen, ' lines ', maxlines

def testBillBoard():
    print BillBoard(100,20, 'Hack your way to the cup')
    print BillBoard(55,25,'Can you hack')
    print BillBoard(100 ,20, 'hacker cup 2013')
    print BillBoard(20 ,6, 'hacker cup')
    print BillBoard(10, 20, 'MUST BE ABLE TO HACK')
    print BillBoard(350, 100, "Facebook Hacker Cup 2013")


"""
find the strong components in a DAG.
1. back-edge form a cycle, all nodes inside the cycle belongs to one strong components.
2. cross-edge(end-vertex visited earlier than start vertex) can merge two back-edge formed cycles.
3. DFS find back-edges and cross-edges.
4. 3 aux properties for each vertex.
    v.parent   : who is this vertex's parent during DFS
    v.strong_component_idx  : which strong component this vertex belongs to.
    v.earliest_anecestor_level : when process back-edge, the update start's earliest to end's level.
5. encounter back-edge, update start's earliest level to end's.
   encounter cross-edge, if old cycle not belong to any strong component, merge start in. start.earliest_level = end.
6. when done a DFS node, sync back earliest anecest level to parent. if v.parent's earliest = v's earliest
7. a strong component is vertex whose earliest level is itself. When encounter this node, pop stack, assign all node's strong_component_idx.
"""

if __name__ == '__main__':
    #testDeRNGed()
    #testFence()
    #testdiffSum()
    #testHotdogStand()
    #testBacteria()
    #testTravelPlan()
    #testCandyStore()
    #testStamper()
    testWifiTower()
    #testDiceGames()
    #testNestRand()
    #testSubsetSumMax()
    #testQuickSum()
    #testShortPalindromes()
    #testMiniPaint()
    #testStripePainter()
    #testPerfectMatching()
    #testSmartWordToy()
    #testTurntableService()
    #testBusToCafeteria()
    #testPaintingFence()
    #testBentleyOttmann()
    #testPaintingFence()
    #testBusStops()
    #testZigZag()
    #testBalancedPartition()
    #testTravelingSalesmanProblem()
    #testBillBoard()
