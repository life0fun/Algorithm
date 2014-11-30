#!/usr/bin/env python

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None


# iterate with 3 ptr, pre, cur, next, move while next.
def rlist(l):
    if not l or not l.next:
        return l

    pre, cur, next = None, l, l.next
    while next:
        cur.next = pre
        pre, cur, next = cur, next, next.next
    return cur

# recursion with head, and pass in last iteration of head's prev for head's next.
def rlist(pre, head):
    if not head or not head.next:
        return head

    hnext = head.next
    head.next = pre
    return rlist(head, hnext)

# reverse list by inject to head
# swap cur's next to head, sift down cur, repeat.
# swap 3 ptrs, hd.next where injected, prev and next of swapped item(cur's next).
def reverse-inject-head(root):
    hd = new Node()
    hd.next = root
    if not root or not root.next:
        return root

    while root.next:
        swap = root.next  # swap node is root's next.
        swap_next = swap.next
        # inject swap to be hd
        swap.next = hd.next
        hd.next = swap.next
        # for next iteration, inj swap_next
        root.next = swap_next

    return hd.next

# look at head cur. if head cur next is same, reset head cur next to next's next.
def remove-dup(root):
    if not root or not root.next:
        return root

    cur = root
    while cur.next:
        nxt = cur.next
        if nxt.value != cur.value:
            cur = cur.next
        else:
            cur.next = nxt.next

    return root


# while loop to skip head and its dup. if head has dup, skip it.
# recur with prev, next. else, update prev's next to head. recur head, head.next.
def remove-all-dup(prev, head):
    if not head:
        return head

    cur = head
    while cur.next and cur.next.val == cur.val:
        cur = cur.next
    
    # if head has dup, recur with prev and next
    if head != cur:
        return remove-all-dup(prev, cur.next)
    else:
        # update prev's next here, and recur.
        prev.next = head
        return remove-all-dup(head, head.next)

def remove-all-dup(head):
    if not head or not head.next:
        return head

    cur = head
    while cur.next and cur.next.val == head.val:
        cur = cur.next

    if cur != head:
        return remove-all-dup(cur.next)
    else:
        head.next = remove-all-dup(head.next)
        return head

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