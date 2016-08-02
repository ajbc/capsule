import numpy as np
from scipy.special import digamma

'''a = np.random.gamma(1.0,5.0, 10)
print "raw"
print a

print"\nnormalized"
na = a / sum(a)
print na

print "\n******\n\nlog raw"
print digamma(a) - digamma(sum(a))

print "\nlog normalized"
print digamma(na) - digamma(sum(na))

print digamma(na*sum(a)) - digamma(sum(na)*sum(a))'''

import sys, sqlite3
import numpy as np
from collections import defaultdict

stem = "may23/weekly_bodyandsubj1976/capsule_dur0/"
stem = "may23/weekly_bodyandsubj1976/exp3/"
label = 'final'

pi = np.loadtxt(stem + 'pi-' + label + '.dat')[:,1:]
psi = np.loadtxt(stem + 'psi-' + label + '.dat')
epsilon = np.loadtxt(stem + 'epsilon-' + label + '.dat')
theta = np.loadtxt(stem + 'theta-' + label + '.dat')[:,1:]
beta = np.loadtxt(stem + 'beta-' + label + '.dat')[:,1:]
docmap = {}
for line in open("/home/statler/achaney/dat/DE/cable_data/pro/may23/weekly_bodyandsubj1976/doc_ids.tsv"):
    doc, idx = line.strip().split('\t')
    docmap[int(doc)] = idx
entity = {}
date = {}
for line in open("/home/statler/achaney/dat/DE/cable_data/pro/may23/weekly_bodyandsubj1976/meta.tsv"):
    doc, author, time = line.strip().split('\t')
    entity[int(doc)] = int(author)
    date[int(doc)] = int(time)


scores = defaultdict(lambda: defaultdict(float))
counts = defaultdict(lambda: defaultdict(int))
for line in open("/home/statler/achaney/dat/DE/cable_data/pro/may23/weekly_bodyandsubj1976/train.tsv"):
    doc, term, count = [int(v) for v in line.strip().split('\t')]
    scores[date[doc]][doc] += pi[date[doc], term] * count
    counts[date[doc]][doc] += count



for week in range(53):
    sc = 0.0
    c = 0
    for s in scores[week]:
        sc += scores[week][s] / counts[week][s]
        c +=1
    print week, sc, c, (sc/c)
    #scores[week], counts[week], (scores[week]/counts[week])
