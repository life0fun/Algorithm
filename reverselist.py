#!/usr/bin/env python

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

def reorderList(head):
    print "head ", head.val
    if not head.next or not head.next.next:
        return head
        
    slow,fast = head, head
    while fast.next:
        slow = slow.next
        fast = fast.next
        if fast.next:
            fast = fast.next
        else:
            break
    print slow.val, fast.val

    mid = slow
    # reverse second half
    revprev = slow
    revcur = slow.next
    while revcur is not fast:
        revnext = revcur.next
        revcur.next = revprev
        revpre = revcur
        revcur = revnext
    revcur.next = revprev

    print revcur.val, fast.val, revcur.next.val
    
    # # mid is the end now
    mid.next = None
    start = head
    while start != mid:
        startnext = start.next
        revnext = revcur.next

        print "----", start.val, start.next.val, revcur.val

        start.next = revcur
        revcur.next = startnext
        start = startnext
        revcur = revnext
    
    print head.val, head.next.val, head.next.next.val, head.next.next.val
    return head


if __name__ == '__main__':
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n4 = Node(4)
    n1.next = n2
    n2.next = n3
    n3.next = n4

    hd = reorderList(n1)
    print "reordered list ------"
    # while hd:
    #     print hd.val
    #     hd = hd.next