#!/usr/bin/env python

import os,sys
import operator
import pdb

"""
  consolidate add/del to four types of file operation
  1. add, del, rename, copy
  2. rename: if there is a del and add for the same file within 10 ticks, then this is a rename.
             if the path diff more than folders, then it is a rename of folder.
  3. take a sliding window to consolidate del/add by rename, 
     rename always happens with a del followed by an add
  4. copy : find thru file content

First, reduce rename, second, reduce copy....of [Directory or File]
If target is Directory, see a set of file, begin with top folder, follow by files
Anything between 5 ticks consider reducible
    
    Haijin Yan, Feb 2011
"""

class FileEvent():
    def __init__(self):
        self.loge = []
        self.oute = []
        self.adde = []
        self.dele = []
        self.cpe = []
        self.mve = []
        self.logeiter = None
        self.curoff = 0
        
	def log(self, *k):
		print(k)
		#pass

    def getEvents(self, file):
        log = open(file)
        events = (line.split() for line in log)
        for e in events:
            if len(e) > 1:  # skip the one number header
                self.insertEvent(e[0].strip(), e[1].strip(), e[2].strip(), e[3].strip())

        self.logeiter = iter(self.loge)

    def insertEvent(self, action, tm, name, content):
        t = [action, tm, name, content, None, None]
        self.loge.append(t)

    def reduce(self):
        start = 0
        debounce =[]

        while start < len(self.loge):
            event, time, path, content, reduceto, tbd = self.loge[start]
            #print start, '-->', self.loge[start]

            if not self.nextWithinTwoTicks(start):
                self.reduceSingle(start)
                start += 1  # skip this entry if not mergable to next
            else:   # now two events to buf with two appends
                while self.nextWithinTwoTicks(start):
                    debounce.append(start)
                    start += 1
                debounce.append(start)
                start += 1
                
            if len(debounce) > 1:
                self.reduceBlock(debounce)
                debounce[:] = []  # empty the list afterwards

    def reduceSingle(self, offset):
        # reduce a single file event, check only for single file cp by content hash.
        event, time, path, content, reduceto, tbd = self.loge[offset]
        if event == 'DEL':  # only add can possibly be reduced to a copy
            self.oute.append('Deleted file %s.' % (path))
            return

        i = self.reverseFindFile(offset)
        if i >= 0 :
            #print 'Copied file %s -> %s' % (self.loge[i][2], path)
            self.oute.append('Copied file %s -> %s.' % (self.loge[i][2], path))
        elif content == '-':
            self.oute.append('Added dir %s.' % (path))
        else:
            self.oute.append('Added file %s.' % (path))

    def reduceBlock(self, block):
        ndels = nadds = 0
        for offset in block:
            event, time, path, content, reduceto, tbd = self.loge[offset]
            ndels = ndels+1 if event == 'DEL' else ndels  # del dir or rename dir
            nadds = nadds+1 if event == 'ADD' else nadds  # cp dir or rename dir
        
        # the top one in block decides whether they are dir op or file op
        dirOp = True if self.loge[block[0]][3] == '-' else False
        if dirOp:
            oldname = self.loge[block[0]][2]
            newname = self.loge[block[ndels]][2]  # invalid if only add, i.e. copy files
            if ndels == nadds:
                op = 'Rename dir %s -> %s.' %(oldname,newname)
            elif ndels > 0:
                op = 'Deleted dir %s.' %(oldname)
            else:
                op = 'Copied dir %s.' %(oldname)

            self.oute.append(op)
        else:
            for offset in block:
                reduceSingle(offset)

        #print 'debounce offset array:', block
        #print ndels, nadds
        #print op

    # stop at event where follower is a descrete one
    def nextWithinTwoTicks(self, offset):
        try:
            return True if int(self.loge[offset+1][1]) - int(self.loge[offset][1]) <= 2 else False
        except IndexError:
            return False

    def findEventsByWindow(self, offset, starttime, endtime):
        events = []
        for e in self.loge[offset:]:
            if int(e[1]) >= starttime and int(e[1]) <= endtime:
                events.append(e)
            else:
                break
        return events

    # traverse reverse to find a file match the pass in file
    def reverseFindFile(self, offset):
        i = offset
        while i >= 0:
            i -= 1
            if self.loge[offset][3] == self.loge[i][3]:
                #print 'Matching file', self.loge[i], ' == ', self.loge[offset] 
                return i
        return -1 

    def findEventsByKey(self, content):
        for e in self.loge:
            try:
                i = e.index(content, 0)
                yield e
            except ValueError:
                pass

    def test(self):
        self.reduce()

    def output(self):
        #print '------ output ------'
        for op in self.oute:
            print op

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: drpbx_fileevent.py fs.log'
        sys.exit(1)

    fe = FileEvent()
    fe.getEvents(sys.argv[1])
    fe.reduce()
    fe.output()
