#!/usr/bin/env python
import sys

"""
print out chrismas tree
the parameter is tree height, the number of * at each level is 2*height+1
so the start offset is (2*height+1) - sz / 2
"""
def tree(height):
  max_stars = 2*(height-1)+1  # offset starts from 0

  for i in xrange(height):
    cur_stars = i*2 + 1
    offset = (max_stars - cur_stars)/2
    for k in xrange(offset):
      sys.stdout.write(" ")
    for k in xrange(cur_stars):
      sys.stdout.write("*")
    sys.stdout.write("\n")
  
  for i in xrange(3):
    offset = (max_stars - 1) /2
    for k in xrange(offset):
      sys.stdout.write(" ")
    sys.stdout.write("*")
    sys.stdout.write("\n")


"""
calculates every palindrome between 1 and 10,000
idea is to recursively call with smaller width, width-2, then 
filter out number with leading 0.
"""

def palindrom(n):

  def palindrom_width(w, inner_column):
    ret = []
    if w == 1:
      return [str(i) for i in xrange(10)]
    if w == 2:
      if not inner_column:
        return [str(11*i) for i in xrange(1,10)]
      else:
        return ["00"] + [str(11*i) for i in xrange(1,10)]
    else:
      for i in xrange(10):
        for n in palindrom_width(w-2, True):
          strn = str(i) + n + str(i)
          ret.append(strn)
      return ret


  tot = []
  for w in xrange(1, len(str(n))):
    p = palindrom_width(w, False)
    tot += p
    #print "width ", w, " tot ", len(p), " = ", p

  tot = filter(lambda x: x[0] != '0' or len(x) == 1, tot)
  print "total palindrom below " , n, " is " , len(tot)


if __name__ == "__main__":
  palindrom(10)
  palindrom(100)
  palindrom(1000)
  palindrom(100000)

  tree(7)
