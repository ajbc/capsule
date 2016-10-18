import sys, os
import numpy as np
from collections import defaultdict

dat_dir = sys.argv[1]
fit_dir = sys.argv[2]

if sys.argv[3] == 'final':
    iter_id = 'final'
else:
    iter_id = '%04d' % int(sys.argv[3])

epsilon = np.loadtxt(os.path.join(fit_dir, 'epsilon-%s.dat' % iter_id))
theta = np.loadtxt(os.path.join(fit_dir, 'theta-%s.dat' % iter_id))[:,1:]
zeta = np.loadtxt(os.path.join(fit_dir, 'zeta-%s.dat' % iter_id))[:,1:]

Nwords = defaultdict(int)
for line in open(os.path.join(dat_dir, "train.tsv")):
    doc, term, count = [int(t) for t in line.strip().split('\t')]
    Nwords[doc] += count

#cumulative = np.zeros(max(Nwords.keys())+1) # event only
cumulative = np.sum(theta,1) + np.sum(zeta,1)
docs = defaultdict(set)
for doc, time, eps, feps in epsilon:
    cumulative[int(doc)] += feps
    if feps == eps:
        docs[time].add(doc)


eventness = defaultdict(float)
eventness2 = defaultdict(float)
sumfeps = defaultdict(float)
sumeps = defaultdict(float)
dc = defaultdict(float)
cd = defaultdict(float)
Nw = defaultdict(float)
Nwf = defaultdict(float)
print "time", "num.docs", "fweights", "current.metric", "sum.feps", "fweighted.sum.feps", "sum.eps", "sum.feps.divdoc", "sum.eps.divdoc", "sumeps.divNwords", "sumfeps.divNfwords"
for doc, time, eps, feps in epsilon:
    eventness[int(time)] += feps / cumulative[int(doc)]
    eventness2[int(time)] += (feps/eps) * feps / cumulative[int(doc)]
    sumfeps[int(time)] += feps
    sumeps[int(time)] += eps
    dc[int(time)] += feps / eps
    cd[int(time)] += 1
    Nw[int(time)] += Nwords[int(doc)]
    Nwf[int(time)] += Nwords[int(doc)] * (feps / eps)


for time in eventness:
    print time, len(docs[time]), dc[time], (eventness[time]/dc[time]), sumfeps[time], (sumfeps[time]/dc[time]), sumeps[time], (sumfeps[time]/cd[time]), (sumeps[time]/cd[time]), (sumeps[time]/Nw[time]), (sumfeps[time]/Nwf[time]), (eventness2[time]/dc[time])
