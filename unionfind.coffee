#!/usr/bin/env coffee
#

class UnionFind
	@ALPHA = 'abcdefghijklmnopqrstuvwxyz'
	_root = ( -1 for i in [1..50])

	constructor: ->
		console.log 'UnionFind constructor...'
		console.log _root

	insert: (e) ->
		idx = UnionFind.ALPHA.indexOf(e)
		console.log 'inserting : ' + e + ' at idx:' + idx
		if idx >= 0
			_root[idx] = -1

	# recursive, path compression by resetting node's parent during recursion, -1 is the end
	findParent = (idx) ->
		if _root[idx] < 0
			return idx
		_root[idx] = findParent _root[idx]  # recursive path compression
		return _root[idx]  # ret idx's parent after path compression recursion done.

	find: (e) ->
		idx = UnionFind.ALPHA.indexOf(e)
		pidx = findParent idx
		console.log 'find: ' + e + ' with root idx:' + pidx
		return pidx

	unionSet: (a, b) ->
		pa = @find a
		pb = @find b

		if _root[pa] < _root[pa]  # a tree deeper
			_root[pb] = pa   # b tree point to a tree
		else
			if _root[pa] is _root[pb]
				_root[pb] -= 1
			_root[pa] = pb  # b tree deeper, a tree points to b tree


		console.log 'unionSet: ' + a + ':' + b + ' idx ' + pa + ':'+pb+' root ' + _root[pa] + ':'  + _root[pb]

	test: ->
		@insert 'a'  # need @ prefix to refer to func defined in prototype
		@insert 'd'
		@insert 'e'
		@insert 'f'
		@unionSet 'a', 'd'
		@unionSet 'e', 'f'
		@unionSet 'a', 'e'
		@find 'a'
		@find 'd'
		@find 'e'
		@find 'f'

f = new UnionFind
f.test()

# 
# export the class, usage:
# _UF = require('unionfind')
# UnionFind = _UF.UnionFind
# f = new UnionFind
# f.test()
#
exports.UnionFind = UnionFind

