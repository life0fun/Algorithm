#! /usr/bin/python

"""
binary tree node implementation
Using generator. Treat tree as a collection.
Tree traverse using generator, yield, take the top ele produced on top of stack.
"""

class Node:
	def __init__(self, value, left=None, right=None):
		self.value=value;
        self.left=left;
        self.right=right
	def children(self):
		return [self.left, self.right]

""" Result: 5 3 1 2 4 """
def treeWalker(node):
	lifo=[]
	while True:
		print node.value
        if not node.left:
			lifo.append(node)
			node=node.left
		else:
			try:
				node=lifo.pop()
			except:
				return None
			node=node.right

""" recursive generator, generator is a os managed list """
def inorder(node, level):  # you can carry extra info here down the recursion
	if node.left:
		for x in inorder(node.left): 
			yield x

	yield node.value

	if node.right:
		for x in inorder(node.right): 
			yield x

def dfs(node):
	if node.left:	
		for x in dfs(node.left):
			yield x
	if node.right:
		for x in dfs(node.right):
			yield x
	yield node.value
	

def bfs(node):
	q=[]
	q.append(node)

	while len(q):
		n = q.pop()

		yield n.value

		for child in n.children():
			if child:
				q.append(child)

if __name__ == "__main__":
	n1=Node(1)
	n2=Node(2)
	n3=Node(3,n1,n2)
	n4=Node(4)
	n5=Node(5,n3,n4)
	treeWalker(n5)

	print 'Inorder traverse....'
	g=inorder(n5)
	for node in g:
		print node

	print 'Breath fs.......'
	for node in bfs(n5):
		print node

	print 'Deep fs.......'
	for node in dfs(n5):
		print node
