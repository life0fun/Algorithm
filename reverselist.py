#!/usr/bin/env python

class Node:
    def __init__(self, x):
        self.val = x
        self.next = None


# iterate with 3 ptr, pre, cur, next, move while next.
# 1 > 2 > 3 > 4  ==> 4 > 3 > 2 > 1
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
    if not head:
        return pre

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

# if head is unique, set head next = recur(head.next). 
# otherwise, directly recur(first node that's node head)
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


# rotate, not reverse, 1,2,3,4,5 => 4,5,1,2,3
def rotate-at-k(head, k):
    if not head: return head
    tail = head
    sz = 1
    while tail.next:
        tail = tail.next
        sz += 1
    p = tail
    for i in xrange(k%sz):
        p = p.next
    rhead = p.next
    p.next = None
    return rhead


# remove nth from end, 2 ptrs, p, q. q move n step at first. then both move to end.
def drop-tail-kth(head, k):
    if not head: return head
    p, q = head
    for i in xrange(k):   # cross k edges, move q to kth node, [0..k]
        q = q.next
    while q.next:
        q = q.next
        p = p.next
    p.next = p.next.next
    return head


# swap every 2 adj node in pair, 1,2,3,4 => 2,1,4,3
def swap-pair(head):
    if not head or head.next:
        return head
    headnnxt = head.next.next
    nhead = head.next
    nhead.next = head
    head.next = swap-pair(headnnxt)
    return nhead


# reverse nodes in k-group, k=3, 1,2,3,4,5 => 3,2,1,4,5
def reverse-k-group(head, k):
    tail = head
    for i in xrange(k-1): # cross k-1 edges
        if not tail:
            return head
        tail = tail.next

    nexthead = tail.next

    dummyhead = new Node()
    dummyhead = head
    cur = head
    while cur.next != nexthead:
        swp = cur.next
        swapnxt = swp.next
        
        swp.next = dummyhead.next
        dummyhead.next = swp
        cur.next = swpnxt

    cur.next = reverse-k-group(nexthead, k)
    return dummyhead.next

# cp list with random edge.
def cpList(head):
    if not head:
        return head

    cur = head
    # clone
    while cur is not None:
        newcur = new Node(cur.value)
        newcur.next = cur.next
        cur.next = newcur
        cur = newcur.next
    # set random
    cur = head
    while cur is not None:
        cur.next.random = cur.random.next
        cur = cur.next.next
    # split
    cur = head
    newhead = cur.next
    newtail = newhead
    while cur is not None:
        newcur = cur.next
        cur.next = newcur.next
        newtail.next = newcur
    return newhead

# reverse the second half, ln->ln-1->ln-2, then interleave with l1->l2->l3
def reorderList(head):
    def reverse(pre, head):
        if not head:
            return pre
        hdnext = head.next
        head.next = pre
        return reverse(head, ndnext)
    
    if not head or not head.next or head.next.next:
        return head

    slow,fast = head, head
    while fast.next:
        slow, fast = slow.next, fast.next
        fast = fast.next if fast.next else break
    mid = slow
    rhead = reverse(None, mid)
    cur, rcur = head, rhead
    while cur and rcur:
        p,q = cur, rcur
        p.next = q
        q.next = cur.next
        cur = cur.next
        rcur = rcur.next


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