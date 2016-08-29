import sys, sqlite3, os
import numpy as np
from collections import defaultdict
import scipy
from scipy.stats import rankdata

data_dir = sys.argv[1]
fit_dir = sys.argv[2]

if sys.argv[3] == 'final':
    iter_id = 'final'
else:
    iter_id = '%04d' % int(sys.argv[3])

def ECDF(vals):
    v = np.cumsum(sorted(vals))[[int(i-1) for i in rankdata(vals)]] #4
    return v*vals/sum(v)
    #return np.cumsum(sorted(vals))[[int(i-1) for i in rankdata(vals)]] #3
    #return rankdata(vals) / len(vals) #1

#theta = np.sum(np.loadtxt(sys.argv[2])[:,1:], axis=1)
docmap = {}
for line in open(os.path.join(data_dir, 'doc_map.tsv')): # doc map
    doc, idx = line.strip().split('\t')
    docmap[int(doc)] = idx

docs = {}
for general_topics in np.loadtxt(os.path.join(fit_dir, 'theta-%s.dat' % iter_id)):
    doc = general_topics[0]
    theta = general_topics[1:]
    docs[int(doc)] = list(theta)

for doc, val in np.loadtxt(os.path.join(fit_dir, 'zeta-%s.dat' % iter_id)):
    docs[int(doc)].append(val)

ddidx = {}
for line in open(os.path.join(fit_dir, 'epsilon-%s.dat' % iter_id)): # epsilon
    doc, date, val, fval = line.split('\t')[:4]
    if float(val) ==3e-1:
        continue
    ddidx[(int(doc), int(date))] = len(docs[int(doc)])
    docs[int(doc)].append(float(fval))

ecdf_docs = {}
for doc in docs:
    ecdf_docs[doc] = ECDF(docs[doc])

scores = defaultdict(dict)
for doc, date in ddidx:
    scores[date][doc] = ecdf_docs[doc][ddidx[(doc, date)]] #3 and #4
    #scores[date][doc] = docs[doc][ddidx[(doc, date)]]/ sum(docs[doc]) #2
    #scores[date][doc] = ecdf_docs[doc][ddidx[(doc, date)]] #1 meh...too many tied with value of 1.0

cables_db = "/dat/Dropbox/cables/raw/history-lab-source.sqlite"
con = sqlite3.connect(cables_db)
cur = con.cursor()

for week in sorted(scores):
    print "date", week
    for doc in sorted(scores[week], key=lambda x: -scores[week][x])[:20]:
        #print doc, scores[week][doc], docmap[doc]
        for row in cur.execute("SELECT date, \"from\", \"to\", subject FROM docs where id='%s'" % docmap[doc]):
            print docmap[doc], scores[week][doc], "\t", row[0], '\t', row[1], '\t', row[2], '\t', row[3]
    print ""

scores = defaultdict(dict)
for doc in ecdf_docs:
    for k in range(100):
        scores[k][doc] = ecdf_docs[doc][k]

for k in range(100):
    print "general topic", k
    for doc in sorted(scores[k], key=lambda x: -scores[k][x])[:20]:
        #print doc, scores[week][doc], docmap[doc]
        for row in cur.execute("SELECT date, \"from\", \"to\", subject FROM docs where id='%s'" % docmap[doc]):
            print docmap[doc], scores[k][doc], "\t", row[0], '\t', row[1], '\t', row[2], '\t', row[3]
    print ""

