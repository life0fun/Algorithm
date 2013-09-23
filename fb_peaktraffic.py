"""
FB peak traffic puzzle
http://www.facebook.com/careers/puzzles.php?puzzle_id=8
find clusters with full connections among all the peers in the cluster.

   Haijin Yan, 2011, Jan
"""

import sys
import pdb

class FriendsCluster():
    def __init__(self):
        self.users = {}     # user as key, a friend set as value 
        self.actions = []   # actions, sender -> recver
        self.clusters = []  # friends clusters 

    # given a list of user, get all relationship pairs(ab and ba) among them
    def getAllPermuPairs(self, l):
        if not l:
            return

        head = l[0]
        for e in l[1:]:
            yield head+e
            yield e+head

        for k in self.getAllPermuPairs(l[1:]):
            yield k

    # test cases
    def prepareTestData(self):
        labc = ['ab','ba','ac','ca','bc','cb']
        ldef = ['de','ed','df','fd','ef']
        lwxyz = []

        for e in self.getAllPermuPairs(['w','x','y','z']):
            lwxyz.append(e)
        for e in self.getAllPermuPairs(['x','y','z']):
            lwxyz.append(e)
        for e in self.getAllPermuPairs(['a','b','c', 'd']):
            lwxyz.append(e)

        #self.actions = lwxyz+labc+ldef
        self.actions = lwxyz+labc
        print self.actions
        #print self.actions.pop()

    # generator expression to read data
    def readInData(self, file):
        logf = open(file)
        actions = ( line.split() for line in logf )
        for action in actions:
            self.actions.append(actions[1].strip()+actions[2].strip())


    # build user relation matrix
    def buildUserActionMatrix(self):
        if not self.actions:
            return
        
        for action in self.actions:
            s, r = [ x.strip() for x in action ] 

            try:
                self.users[s].add(r)
            except KeyError:
                self.users[s] = set()
                self.users[s].add(r)

        for k, v in self.users.iteritems():
            #print 'user:friends:',  k, v
            pass


    # go thru one user's friends matrix, find cluster with size 2, 3, ... to len(friends)
    def findOneUsersClusters(self, u):
        friends = list(self.users[u])
        if not friends:
            print 'user has no friends:', u
            return False

        nfriends = len(friends)
        if nfriends < 2:   # cluster size must > 3
            print 'user has less than 2 friends:', u, friends
            return

        print ' user friends:', u, friends
        # cluster size of 2, 3, ... etc
        for step in xrange(2, nfriends+1):
            for idx in xrange(0, nfriends-step+1):
                cluster = friends[idx:idx+step]
                cluster.insert(0, u)

                for e in self.getAllPermuPairs(cluster):
                    s,r = [ x.strip() for x in e] 
                    if not r in self.users[s]:   # s->r non exist
                        print 'No Matching...:'+e
                        return

         # upon here, friend cluster is a valid 
        self.clusters.append(cluster)

    def findAllUsersClusters(self):
        self.buildUserActionMatrix()
        for k, v in self.users.iteritems():
            self.findOneUsersClusters(k)

        print '---finding all clusters---'
        print self.clusters

    # consolidate all the found clusters
    def consolidateClusters(self):
        nclusters = len(self.clusters)
        for i in xrange(nclusters):
            c = self.clusters[i]
            c.sort()
            print c

            # list is not hashable, can not be the key to dict and set
            try:
                self.clusters.index(set(c))
            except ValueError:
                self.clusters.append(set(c))

        # now reduce cluster subset to super set
        self.clusters = self.clusters[nclusters:]
        print self.clusters

        consolidatedl = []
        while self.clusters:
            top = self.clusters.pop(0)
            skiptop = False
            #for t in self.clusters[self.clusters.index(e)+1:]:
            for e in self.clusters:
                if top < e:  # e is super set
                    skiptop = True
                    break
                if top > e:  # top is super set, remove e
                    self.clusters.remove(e)
            if not skiptop:
                consolidatedl.append(top)

        self.clusters = consolidatedl
        print self.clusters

if __name__ == '__main__':
    fc = FriendsCluster()
    fc.prepareTestData()
    fc.findAllUsersClusters()
    fc.consolidateClusters()
