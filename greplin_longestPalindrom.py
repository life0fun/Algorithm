#!/usr/bin/env python

import copy

''' the common length of two strings
'''
def textdiff(s, t):
    minlen = len(t) if len(t) < len(s) else len(s)
    for i in xrange(minlen):
        if s[i] != t[i]:
            break
    return i

''' reverse the text, build suffix array from both strings
'''
def longestPalindrom(text):
    lenx = len(text)
    textl = list(text)
    rtextl = copy.deepcopy(textl)
    rtextl.reverse()

    textsuffixl = []
    rtextsuffixl = []

    for i in xrange(lenx):
        textsuffixl.append(textl[i:])
        rtextsuffixl.append(rtextl[i:])

    textsuffixl.sort()
    rtextsuffixl.sort()

    palindromlen = 0
    palindromidx = 0

    for i in xrange(lenx):
        s = textsuffixl[i]
        for t in [e for e in rtextsuffixl if e[0] == s[0]]:
            commonsize = textdiff(s,t)
            if commonsize > palindromlen:
                palindromlen = commonsize
                palindromidx = i

    print 'longest palindrom:', textsuffixl[palindromidx][:palindromlen]
    print 'longest palindrom:', ''.join(textsuffixl[palindromidx][:palindromlen])

if __name__ == '__main__':

    #text = "I like racecars that go fast"
    text = 'FourscoreandsevenyearsagoourfaathersbroughtforthonthiscontainentanewnationconceivedinzLibertyanddedicatedtothepropositionthatallmenarecreatedequalNowweareengagedinagreahtcivilwartestingwhetherthatnaptionoranynartionsoconceivedandsodedicatedcanlongendureWeareqmetonagreatbattlefiemldoftzhatwarWehavecometodedicpateaportionofthatfieldasafinalrestingplaceforthosewhoheregavetheirlivesthatthatnationmightliveItisaltogetherfangandproperthatweshoulddothisButinalargersensewecannotdedicatewecannotconsecratewecannothallowthisgroundThebravelmenlivinganddeadwhostruggledherehaveconsecrateditfaraboveourpoorponwertoaddordetractTgheworldadswfilllittlenotlenorlongrememberwhatwesayherebutitcanneverforgetwhattheydidhereItisforusthelivingrathertobededicatedheretotheulnfinishedworkwhichtheywhofoughtherehavethusfarsonoblyadvancedItisratherforustobeherededicatedtothegreattdafskremainingbeforeusthatfromthesehonoreddeadwetakeincreaseddevotiontothatcauseforwhichtheygavethelastpfullmeasureofdevotionthatweherehighlyresolvethatthesedeadshallnothavediedinvainthatthisnationunsderGodshallhaveanewbirthoffreedomandthatgovernmentofthepeoplebythepeopleforthepeopleshallnotperishfromtheearth'

    longestPalindrom(text)
