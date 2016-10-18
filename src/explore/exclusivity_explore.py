import sys, os, time
from datetime import date, timedelta
import sqlite3
from collections import defaultdict
import numpy as np
import scipy
from scipy.stats import rankdata

doc_db = sys.argv[1]
data_dir = sys.argv[2]
fit_dir = sys.argv[3]

if sys.argv[4] == 'final':
    iter_id = 'final'
else:
    iter_id = '%04d' % int(sys.argv[4])

N_terms = 10
N_docs = 10

#import scipy
#from scipy.stats import rankdata
def ECDF(vals):
    return rankdata(vals) / len(vals)

# load vocab
vocab = [line.strip() for line in \
    open(os.path.join(data_dir, 'vocab.dat'))]
# exclusivity comparison
ctopics = np.zeros(len(vocab))
#for line in open(os.path.join(fit_dir, "eta-%s.dat" % iter_id)):
#    entity, terms = line.strip().split('\t', 1)
#    ctopics += np.array([float(t) for t in terms.split('\t')])
#ctopics = np.zeros(len(vocab))
for line in open(os.path.join(fit_dir, "pi-%s.dat" % iter_id)):
    time, terms = line.strip().split('\t', 1)
    time = int(time)
    ctopics += np.array([float(t) for t in terms.split('\t')])
beta = np.loadtxt(os.path.join(fit_dir, "beta-%s.dat" % iter_id))[:,1:].T
#for k in range(len(beta)):
#    ctopics += beta[k]

## Entities
# entity names
entity_names = {}
for line in open(os.path.join(data_dir, "entities.tsv")):
    idx, name = line.split('\t')
    name = name.strip()
    if idx == 'id':
        continue
    entity_names[int(idx)] = name

# entity topics
fout = open(os.path.join(fit_dir, "topics_entity_ex%s.dat" % iter_id), 'w+')
for line in open(os.path.join(fit_dir, "eta-%s.dat" % iter_id)):
    entity, terms = line.strip().split('\t', 1)

    tt = np.array([float(t) for t in terms.split('\t')])
    #terms = dict(zip(vocab, 1.0/((0.5/tt) + (0.5/(tt/ctopics)))))
    #terms = dict(zip(vocab, np.array([float(t) for t in terms.split('\t')])/ctopics))
    terms = dict(zip(vocab, tt))
    topterms = sorted(terms, key=lambda t: -terms[t])[:N_terms]
    fout.write("%s\t%s\n" % (entity_names[int(entity)], ' / '.join(topterms)))
fout.close()

## Events
# map to real dates
date_map = {}
for line in open(os.path.join(data_dir, 'dates.tsv')):
    fit_id, dt = line.strip().split('\t')
    if fit_id == 'id':
        continue
    #tv =dt.split('-')
    #date_map[int(fit_id)] = date(int(tv[0]), int(tv[1]),  int(tv[2]))
    date_map[int(fit_id)] = dt #weekly

# WARNING: this requires daily time intervals
def get_date(time):
    if time not in date_map:
        date_map[time] = "date %d not found" % time
        #date_map[time] = get_date(time-1) + timedelta(days=1)
        #date_map[time] = get_date(time-1) + timedelta(days=7)
    return date_map[time]

# event topics
fout = open(os.path.join(fit_dir, "topics_events_ex%s.dat" % iter_id), 'w+')
for line in open(os.path.join(fit_dir, "pi-%s.dat" % iter_id)):
    time, terms = line.strip().split('\t', 1)
    time = int(time)

    dt = get_date(time)

    tt = np.array([float(t) for t in terms.split('\t')])
    vv = 0.5
    freq_cdf = ECDF(tt)
    excl_cdf = ECDF(tt/ctopics)
    terms = dict(zip(vocab, 1.0/(((1.0-vv)/freq_cdf) + vv/excl_cdf)))
    topterms = sorted(terms, key=lambda t: -terms[t])[:N_terms]
    fout.write("%s\t%s\n" % (dt, ' / '.join(topterms)))
fout.close()

# general topics
#beta = np.loadtxt(os.path.join(fit_dir, "beta-%s.dat" % iter_id))[:,1:].T
fout = open(os.path.join(fit_dir, "topics_general_ex%s.dat" % iter_id), 'w+')
for k in range(len(beta)):
    #terms = dict(zip(vocab, 1.0/((0.5/beta[k]) + (0.5/(beta[k]/ctopics)))))
    #terms = dict(zip(vocab, beta[k]/ctopics))
    terms = dict(zip(vocab, beta[k]))
    topterms = sorted(terms, key=lambda t: -terms[t])[:N_terms]
    top3 = ' / '.join(topterms[:3])
    fout.write("general #%d\t%s\n" % (k, ' / '.join(topterms)))
fout.close()


# get top docs for each event
docmap = {}
#for line in open(os.path.join(data_dir, 'doc_map.tsv')):
for line in open(os.path.join(data_dir, 'doc_ids.tsv')):
    doc, idx = line.strip().split('\t')
    docmap[int(doc)] = idx

epsilon = np.loadtxt(os.path.join(fit_dir, 'epsilon-%s.dat' % iter_id))
theta = np.loadtxt(os.path.join(fit_dir, 'theta-%s.dat' % iter_id))[:,1:]
zeta = np.loadtxt(os.path.join(fit_dir, 'zeta-%s.dat' % iter_id))[:,1:]

cumulative = np.sum(theta,1) + np.sum(zeta,1)
for doc, time, eps, feps in epsilon:
    cumulative[int(doc)] += feps

Nwords = defaultdict(int)
for line in open(os.path.join(data_dir, "train.tsv")):
    doc, term, count = [int(t) for t in line.strip().split('\t')]
    Nwords[doc] += count

scores = defaultdict(dict)
scores2 = defaultdict(dict)
scores3 = defaultdict(dict)
for doc, time, eps, feps in epsilon:
    scores[int(time)][int(doc)] = feps / cumulative[int(doc)]
    scores2[int(time)][int(doc)] = feps
    scores3[int(time)][int(doc)] = feps * Nwords[int(doc)] / cumulative[int(doc)]

#con = sqlite3.connect(doc_db)
#cur = con.cursor()

fout = open(os.path.join(fit_dir, "top_docs_unw_%s.dat" % iter_id), 'w+')
#fout2 = open(os.path.join(fit_dir, "top_docs_raw_%s.dat" % iter_id), 'w+')
#fout3 = open(os.path.join(fit_dir, "top_docs_wgh_%s.dat" % iter_id), 'w+')
for week in sorted(scores):
    fout.write("date %d, %s\n" % (week, date_map[week]))
    #fout2.write("date %d\n" % week)
    #fout3.write("date %d\n" % week)
    for doc in sorted(scores[week], key=lambda x: -scores[week][x])[:20]:
        fout.write("%s\n" % docmap[doc])
        #for row in cur.execute("SELECT date, \"from\", \"to\", subject FROM docs where id='%s'" % docmap[doc]):
        #    fout.write("%s\t%f\t%s\t%s -> %s :: %s\n" % \
        #        (docmap[doc], scores[week][doc], row[0], row[1], row[2], row[3]))
    #for doc in sorted(scores2[week], key=lambda x: -scores2[week][x])[:20]:
    #    for row in cur.execute("SELECT date, \"from\", \"to\", subject FROM docs where id='%s'" % docmap[doc]):
    #        fout2.write("%s\t%f\t%s\t%s -> %s :: %s\n" % \
    #            (docmap[doc], scores2[week][doc], row[0], row[1], row[2], row[3]))
    #for doc in sorted(scores3[week], key=lambda x: -scores3[week][x])[:20]:
    #    for row in cur.execute("SELECT date, \"from\", \"to\", subject FROM docs where id='%s'" % docmap[doc]):
    #        fout3.write("%s\t%f\t%s\t%s -> %s :: %s\n" % \
    #            (docmap[doc], scores3[week][doc], row[0], row[1], row[2], row[3]))
    fout.write("\n")
    #fout2.write("\n")
    #fout3.write("\n")
fout.close()
#fout2.close()
#fout3.close()
