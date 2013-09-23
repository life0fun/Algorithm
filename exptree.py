#!/usr/bin/env python

import sys

class Node(object):
	"""docstring for Node"""
	def __init__(self, val, l, r):
		super(Node, self).__init__()
		self.val = val
		self.lnode = l
		self.rnode = r
		
class ExpTree(object):
	def __init__(self, exp):
		super(ExpTree, self).__init__()
		self.exp = exp
		self.numl = []
		self.symboll = []
		self.operator = '+-*/%()'
		self.preced = ['()', '+-','*/%'] #() means eval separately, lowest precedence.

	def log(self, *k, **kw):
		print k

	def toString(self):
		pass

	def comparePreced(self, cur, pre):
		for i in xrange(len(self.preced)):
			if cur in self.preced[i]:
				curprio = i
			if pre in self.preced[i]:
				preprio = i

		self.log(cur, '?', pre, ' = ', curprio, ':', preprio)
		return curprio-preprio

	def infix2postfix(self):
		# when eval, from left to right, see operator, pop two num, push result
		postfix = [] # final postfix string
		stack = []  # operator stack
		for c in self.exp:
			if not c.strip():
				continue
			# output operands
			if not c in self.operator:
				postfix.append(c)
				continue
			if c == '(':
				stack.append(c)
				continue
			if c == ')':
				while len(stack):
					op = stack.pop()
					if op != '(':  #output everything between ()
						postfix.append(op)
					else:
						break
				continue
			if not len(stack):
				stack.append(c)
				continue
			else:
				while len(stack):
					# ensure top stack that has higher precedence be outputed first
					# as postfix evaled from left to right.
					if self.comparePreced(c, stack[len(stack)-1]) <= 0:
						postfix.append(stack.pop())
					else:
						break  # break loop back on operator stack until less
				stack.append(c)
				continue
		# final output
		print stack
		while len(stack):
			postfix.append(stack.pop())
		print postfix
		return postfix

	def eval(self, postfix):
		stack = []
		for c in postfix:  # eval from left to right
			if not c in self.operator:
				stack.append(c)
			else:
				stack.append((stack.pop(), c, stack.pop())) 
		print stack

if __name__ == '__main__':
	exp = 'a*(b+c/d)'
	#exp = 'x-y*a+b/c'
	t = ExpTree(exp)
	t.toString()
	t.eval(t.infix2postfix())
