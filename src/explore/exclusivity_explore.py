import sys, os, time
from datetime import date, timedelta
import sqlite3
from collections import defaultdict
import numpy as np

doc_db = sys.argv[1]
data_dir = sys.argv[2]
fit_dir = sys.argv[3]

if sys.argv[4] == 'final':
    iter_id = 'final'
else:
    iter_id = '%04d' % int(sys.argv[4])

N_terms = 10
N_docs = 10

con = sqlite3.connect(doc_db)
cur = con.cursor()

# load vocab
vocab = [line.strip() for line in \
    open(os.path.join(data_dir, 'vocab.dat'))]
# exclusivity comparison
ctopics = np.zeros(len(vocab))
#for line in open(os.path.join(fit_dir, "eta-%s.dat" % iter_id)):
#    entity, terms = line.strip().split('\t', 1)
#    ctopics += np.array([float(t) for t in terms.split('\t')])
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
        #date_map[time] = get_date(time-1) + timedelta(days=1)
        date_map[time] = get_date(time-1) + timedelta(days=7)
    return date_map[time]

# event topics
fout = open(os.path.join(fit_dir, "topics_events_ex%s.dat" % iter_id), 'w+')
for line in open(os.path.join(fit_dir, "pi-%s.dat" % iter_id)):
    time, terms = line.strip().split('\t', 1)
    time = int(time)

    dt = get_date(time)

    tt = np.array([float(t) for t in terms.split('\t')])
    terms = dict(zip(vocab, -1.0/((0.5/np.log(tt)) + (0.5/(tt/ctopics)))))
    #terms = dict(zip(vocab, -tt/ctopics))
    #terms = dict(zip(vocab, np.array([float(t) for t in terms.split('\t')])/ctopics))
    topterms = sorted(terms, key=lambda t: terms[t])[:N_terms]
    fout.write("%s\t%s\n" % (date_map[time], ' / '.join(topterms)))
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

