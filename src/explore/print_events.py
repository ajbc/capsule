#!/usr/bin/python

# printtopics.py: Prints the words that are most prominent in a set of
# topics.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys, os, re, random, math, urllib2, time, cPickle
import numpy as np
from collections import defaultdict


def main():
    """
    Displays topics fit by onlineldavb.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    vocab = [f.strip() for f in file(sys.argv[1]).readlines()]
    print len(vocab)
    pi = np.loadtxt(sys.argv[2])[:,1:] #pi
    strength = np.loadtxt(sys.argv[3]) #psi



    frequency = {}
    exclusivity = defaultdict(dict)
    V = pi.shape[1]
    #print V, "vocab size"

    for k in range(0, len(pi)):
        lambdak = pi[k, :]
        lambdak = lambdak / sum(lambdak)
        lambdak = np.array(lambdak)
        lambdak[lambdak==np.inf] = 0
        frequency[k] = lambdak
    for v in range(V):
        total = 0.
        for k in range(len(frequency)):
            total += frequency[k][v]
        for k in range(len(frequency)):
            if total == 0:
                exclusivity[k][v] = 0
                #print vocab[v]
            else:
                exclusivity[k][v] = frequency[k][v] / total

    Fcdf = {}
    Ecdf = {}
    for k in range(0, len(pi)):
        temp = zip(list(frequency[k]), range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        Fcdf[k] = {}
        cv = 0
        for val,idx in temp:
            cv += val
            Fcdf[k][idx] = cv

        ex = [exclusivity[k][v] for v in range(V)]
        temp = zip(ex, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        Ecdf[k] = {}
        cv = 0
        for val,idx in temp:
            cv += val
            Ecdf[k][idx] = cv

    for k in range(0, len(pi)):
        #temp = sorted(range(V), key = lambda v: 1./((0.01/Ecdf[k][v]) + (0.99/Fcdf[k][v])), reverse=False)
        #temp = sorted(range(V), key = lambda v: 1./((0.99/Ecdf[k][v]) + (0.01/Fcdf[k][v])), reverse=False)
        #temp = sorted(range(V), key = lambda v: Fcdf[k][v], reverse=False)
        #temp = sorted(range(V), key = lambda v: Ecdf[k][v], reverse=False)
        #temp = sorted(range(V), key = lambda v: 1./((0.9/Ecdf[k][v]) + (0.1/Fcdf[k][v])), reverse=False)
        temp = sorted(range(V), key = lambda v: 1./((0.5/Ecdf[k][v]) + (0.5/Fcdf[k][v])), reverse=False)
        terms = ', '.join(['%s' % vocab[temp[i]] for i in range(int(sys.argv[4]))])
        #terms = '\n'.join(['%s (%f, %f, %f)' % (vocab[temp[i]], Fcdf[k][temp[i]], Ecdf[k][temp[i]], 1./((0.01/Ecdf[k][temp[i]]) + (0.99/Fcdf[k][temp[i]]))) for i in range(int(sys.argv[3]))])
        print '%d\t%f\t%s' % (k, strength[k], terms)#, 1./((0.5/Ecdf[k][v]) + (0.5/Fcdf[k][v])))
        #print '    topic %d: %s' % (k, terms)


if __name__ == '__main__':
    main()
