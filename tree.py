#!/usr/bin/env python

import sys
import re, types
from collections import defaultdict

"""
defaultdict takes one arg(function) so all the default value for the key
got set to the result of calling that function.

defaultdict(str), defaultdict(list), defaultdict(int)

DFS : when dfs into a node, the node itself does not know its level. it deps on
      its parent to tell its level. When recursion, tell the sub-recursion its level.
      DFS(child, child-level)

Min vertex cover of a tree: DFS + greedy.
when DFS of all the children of a node, we have traversed all the edges of the node.
For min vertex cover problem, for each parent-son edge, we need to select one. Select parent.
"""

''' one line to create a tree'''
def tree():
    return defaultdict(tree)

def tokenize(s, removePunctuation = True):
    if removePunctuation:
        p = re.compile('[\.,!?()\[\]{}:;"\'<>/ \n\r\t]')
        return [el for el in p.split(s) if el]
    else:
        t = []
        s = re.split('\s', s)
        p = re.compile('(\W)')
        for phrase in s:
            words = p.split(phrase)
            for word in words:
                t.append(word)
        return [el for el in t if el]

''' each node has 26 children in the next dict '''
class Node(object):
    def __init__(self, name=None):
        self.name = name
        self.color = 0
        self.freq = 0
        self.next = defaultdict(int)  # 26 branches out, dict key=[a-z]
        self.indent = 0
        self.children = []
        self.indentstr = '=> '
        self.op = ''

    def log(self, *k, **kw):
        print k

    ''' print out the node with its indent '''
    def toString(self):
        ind = ''
        for i in xrange(self.indent):
            ind += self.indentstr
        #print ind, self.name
        return ind + self.name

    def buildNode(self, strval):
        l = strval.split()
        self.indent = len(l)-1
        self.name = l[-1]

    def addChild(self, child):
        self.children.append(child)

    def visit(self):
        self.color = 1
        self.log(self)

    ''' DFS pre-order, recursive, carry parent's indent, add my on top of it.
        update the total idx/cnt offset after each recursion
        idx just record total # of nodes traversed. Every traverse incr idx.
    '''
    def traverse(self, indent, idx):
        ''' recursion into this node, parent tells its level '''
        self.indent += indent
        #self.log('traverse one node : idx :', idx)
        self.log(self.toString())   # pre-order visit
        for c in self.children:   # wont accumulate idx in callback
            # recursion on cur child, DFS
            idx = c.traverse(indent+1, idx+1)  # recursion into child, tell child's level
        return idx

    ''' connect sibling http://programmers.stackexchange.com/questions/133437/how-to-find-siblings-of-a-tree
        just need a separate list for book keeping to rem the optimal result so far
    '''
    def walker(self, node, level):
        if head[level]:
            node.sib = head[level]
        head[level] = node
        for c in self.node.child:
            self.walker(c, level+1)

''' recursive traverse op node to evaluate 2+3*4 '''
class OpNode(Node):
    def __init__(self, name, op):
        super(OpNode, self).__init__(name)
        self.op = op

    def traverse(self):
        if not self.op:
            return self.toString()
        ''' op node must have lchild and rchild '''
        return "( " + self.children[0].traverse() + " op=" + self.op + " " + self.children[1].traverse() + " )"

class Tree():
    def __init__(self, root=None):
        self.root = root
        self.stack = []

    def log(self, *k, **kw):
        print k

    ''' create a test tree '''
    def createTestTree(self):
        self.root = Node('root')
        l1a = Node('l1a')
        l1b = Node('l1b')
        l1c = Node('l1c')

        l2aa = Node('l2aa')
        l2ab = Node('l2ab')
        l2ca = Node('l2ca')
        l2cb = Node('l2cb')

        l3aaa = Node('l3aaa')
        l3aab = Node('l3aab')
        l3cba = Node('l3cba')

        self.root.addChild(l1a)
        self.root.addChild(l1b)
        self.root.addChild(l1c)

        l1a.addChild(l2aa)
        l1a.addChild(l2ab)

        l1c.addChild(l2ca)
        l1c.addChild(l2cb)

        l2aa.addChild(l3aaa)
        l2aa.addChild(l3aab)
        l2cb.addChild(l3cba)

    def traverse(self):
        tot = self.root.traverse(1, 1)
        print 'total nodes: ', tot

    def iterative_inorder(self):
        stack = []
        lc = self.root
        while lc:
            stack.append(lc)
            lc = lc.children[0] # lchild

        while len(stack):
            top = stack.pop()
            top.visit()
            rc = top.children[1]
            while rc:   # trick is here we need a while loop
                stack.append(rc)
                rc = rc.children[0] # push cur node's lc

    def testSer(self):
        self.createTestTree()
        self.traverse()

    ''' not recursion, use a stack instead '''
    def testDeSer(self):
        for l in file('/Users/e51141/tmp/serde.txt'):
            #self.log(l.strip())
            n = Node(l)
            n.buildNode(l)

            if not self.stack:  # this is root
                self.stack.append(n)
                continue

            # now check the top of stack
            top = self.stack[-1]
            # case 1: cur indent bigger, push cur to stack
            if top.indent < n.indent:
                top.addChild(n)    # cur node is the child of parent node at stack top
                self.stack.append(n)
                continue
            # case 2: cur indent equal, prev has no child, pop, push cur
            if top.indent == n.indent:
                self.stack.pop()
                self.stack[-1].addChild(n)   # pop until stack top is the parent of cur node.
                self.stack.append(n)
                continue
            # case 3: cur indent less, while until prev and pre and prev parent done
            if top.indent > n.indent:
                while top.indent >= n.indent:   # keep popping until <, (>=)
                    self.stack.pop()
                    top = self.stack[-1]
                top.addChild(n)
                self.stack.append(n)

        # done, the end of stack is the root
        self.root = self.stack[0]
        self.traverse()

"""
BST serialize and deserialize, dfs recursion, fill # to the left/rite of leaf nodes.
"""
def serializeBST(root):
    if root is None:
        print '#'
        return
    serializeBST root.left
    serializeBST root.rite

def deserializeBST(fstream, parent, asleft):
    node = fstream.nextToken()
    if asleft:
        parent.left = node
    else:
        parent.rite = node

    if node is '#':
        return

    deserializeBST(fstream, node, True)
    deserializeBST(fstream, node, False)


def readBST(minv, maxv, curv, parent, fstream, asleft):
    if curv < maxv and curv > minv:
        node = Node(curv)
        if asleft:
            parent.left = node
        else:
            parent.rite = node

        if childv = fstream.nextToken:
            readBST(minv, curv, childv, node, fstream, True)
            readBST(curv, maxv, childv, node, fstream, False)


# For left tree, we do pre-order print. For rite tree, we do post-order.
def printLeaveClock(root, preorder, shallprint):
    ''' at each subtree, it's always print left, rite order '''
    if preorder:
        if shallprint or root.left is None and root.rite is None:
            print root
        printLeaveClock root.left, True, shallprint if root.left
        printLeaveClock root.rite, True, False if root.rite
    else:
        printLeaveClock root.left, False, False if root.left
        printLeaveClock root.rite, False, shallprint if root.rite
        if shallprint or root.left is None and root.rite is None:
            print root

# to print the boundary, regardless of left edge or rite edge.
def printLeaveClock(root, preorder, shallprint):
    ''' at each subtree, it's always print left, rite order '''
    if preorder:
        if shallprint or root.left is None and root.rite is None:
            print root
        printLeaveClock root.left, True, shallprint if root.left
        printLeaveClock root.rite, True, (shallprint and root.left ? False : True) if root.rite
    else:
        printLeaveClock root.left, False, (shallprint and root.rite ? False : True) if root.left
        printLeaveClock root.rite, False, shallprint if root.rite
        if shallprint or root.left is None and root.rite is None:
            print root


"""
The key is, need to process each node AND edge.
when dfs process each node, for each edge, process, and recursion dfs process
the node connected by the edge. After recursion for all the children of the 
current node, we know the optimal solution for sub problem of current node.
Apply solution of subproblem to all nodes in upper level.

The flow is : process node, at each node, process edges, recursion to all children, process_end.
when setting discovered, means we are putting the node in cur stack, or processing it now in cur stack.
   bfs : enqueu. dfs - dfs(cur-node)
when setting processed, means all children processed. the cur node is off stack.
   dfs : after all children processed.
set node's parent when enq, or recursive.

def initSearch():
    discovered[n] = False
    processed[n] = False

def dfs(g, v):
    discovered[v] = True
    rank[v] = ++ time

    process_vertex_early(v)   # recursive called in, process node early.

    for(nb : neighbor):
        if-not discovered[nb]:
            process_edge(v, nb)    # tree edge, process it.
            parent[nb] = v         # set nb's parent before dfs into nb, in not discovered case.
            dfs(g, nb)

        # if the other node processed = black. mean all its children edges processed. do not process this unidirect edge twice.
        else-if-not processed(nb) || g.isDirected:   # always process edge in directed graph.
            process_edge(v, nb)    # the other end is not processed(off the stack), process this back edge.

    process_vertex_late(v)
    processed[v] = True

def bfs(g, v):
    Q.enqueue(v)
    discovered[v] = processed[v] = True

    while len(Q):
        v = Q.deq()
        process_vertex_early(v)
        processed[v] = True       # when taking off from cur stack, done.
        for (nb : v.neighbor):
            if-not processed(nb) || g.isDirected :   # always process directed edge, and careful not process twice unidirected edge.
                process_edge(v, nb)    # first, process the edge leads to the children.

            if-not discovered[nb]:
                discovered[nb] = True  # mark child as discovered, it is in current active stack.
                parent[nb] = v         # set parent before enqueue the child
                Q.enq(nb)

        process_vertex_late(v)
"""

''' select min vertice that covers all the edges
    max indepent set is the complement of min vertex cover
'''
class MinVertexCover(Node):
    ''' Min vertex cover of a tree: DFS + greedy.
        when DFS of all the children of a node, we have traversed all
        the edges of the node. To get min vertex cover set, for each
        parent-son edge, we need to select one. Select parent.
    '''
    def __init__(self, root=None):
        self.root = root
        self.minvertex = []

    # pass in parent during dfs cursion
    def DFS(self, n, parent):
        # for leaf node, return to select parent.
        if n.children is None:
            return
        # process_node_earlier()
        for c in n.children:
            # process_edge(n, c)
            self.DFS(c, n)

        # process_node_late(), as a child, and done all my paths, update my parent is children's responsible.
        # after done n, traversed all n's children. this parent-child edge, decide which node
        # we select to reprent this parent-child edge. Greedy on parent.
        if not n in self.minvertex and not parent in self.minvertex:
                self.minvertex.append(parent)

    # A node has 3 array, parent, discovered(grey), and processed(black)
    # DFS : process node early : entering time, rank.
    # process node late when done exploring my path. update my parent
    # process edge: (me, my_child), collect info from my reachable child.
    # edge classification: parent/child defined by entering time. pos in stack.
    # tree-edge: to not discovered child.
    # back-edge: to discovered ancestor that is in current stack.
    # forward-edge : to processed children.
    # cross-edge: to a processed anecestor that is not in current stack.
    def dfs(self, n, parent):
        process_node_earlier(n)
        for c in n.children:
            if not c.discovered:    # tree edge
                process_edge(n,c)   # collect info from child, i.e., earliest reachable node from me.
                dfs(c, n)
            if not c.processed:  #  back edge, or forward edge, or cross edge
                process_edge(n,c)   # classify edges, collect info from child, e.g, earliest reachable node from me.

        # with full info from children, calc statics, update parent info if I find min/max
        process_node_late(n, parent)

''' compute the strongly connected components of a DAG.
    each node has color, lowerest parent, and scc #. Color indicates whether the node is in current active stack.
    dfs, explore node's children, back-edge makes all nodes in the loop one strongly connected component(SCC).
    done nodes children, collected all info by edges, select min/max, update parent's min/max.
'''
class StronglyConnectedComponents(Node):
    def __init__:(self, root=None):
        self.time = 0
        self.root = root
        self.earliest = []   # each node, who is the earliest node reachable
        self.scc = []    # each node, which scc the node belong to
        self.sccName = 0  # the first scc, the second scc
        self.activeStack = []

    def dfs(self, n, parent):
        process_node_early(n)
        for c in n.children:
            if not c.discovered:    # tree edge
                process_edge(n,c)   # collect info from child, i.e., earliest reachable node from me.
                dfs(c, n)
            if not c.processed:  #  back edge, or forward edge, or cross edge
                process_edge(n,c)   # classify edges, collect info from child, e.g, earliest reachable node from me.

        # with full info from children, calc statics, update parent info if I find min/max
        process_node_late(n, parent)

    def process_node_early(self, n):
        n.time = self.time++
        self.activeStack.append(n)

    ''' collect info from child at the edge end'''
    def process_edge(self, n, child):
        edgeType = self.classifyEdge(n,child)
        if edgeType is 'back-edge':
            if self.earliest[child].time < self.earliest[n].time:
                self.earliest[n] = child
        # if cross-edge, the other component must not be scc done
        # If it is scc done, means it does not have a path to me.
        if edgeType is 'cross-edge' and scc[child] is -1:
            if self.earliest[child].time < self.earliest[n].time:
                self.earliest[n] = child

    def process_node_late(self, n):
        if self.earliest[n] is n:
            self.popComponent(n)

        if self.low[n].time < self.low[parent]:
            self.low[parent] = self.low[n]

    def popComponent(self, n):
        self.sccName += 1
        self.scc[n] = self.sccName
        while cur = self.activeStack.pop() is not n:
            self.scc[cur] = self.sccName

if __name__ == '__main__':
    tree = Tree()
    tree.testSer()
    tree.testDeSer()
