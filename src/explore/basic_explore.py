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
fout = open(os.path.join(fit_dir, "topics_entity_%s.dat" % iter_id), 'w+')
for line in open(os.path.join(fit_dir, "eta-%s.dat" % iter_id)):
    entity, terms = line.strip().split('\t', 1)

    terms = dict(zip(vocab, [float(t) for t in terms.split('\t')]))
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
    tv =dt.split('-')
    date_map[int(fit_id)] = date(int(tv[0]), int(tv[1]),  int(tv[2]))
    #date_map[int(fit_id)] = dt #weekly

# WARNING: this requires daily time intervals
def get_date(time):
    if time not in date_map:
        date_map[time] = get_date(time-1) + timedelta(days=1)
    return date_map[time]

# event topics
fout = open(os.path.join(fit_dir, "topics_events_%s.dat" % iter_id), 'w+')
for line in open(os.path.join(fit_dir, "pi-%s.dat" % iter_id)):
    time, terms = line.strip().split('\t', 1)
    time = int(time)

    dt = get_date(time)

    terms = dict(zip(vocab, [float(t) for t in terms.split('\t')]))
    topterms = sorted(terms, key=lambda t: -terms[t])[:N_terms]
    fout.write("%s\t%s\n" % (date_map[time], ' / '.join(topterms)))
fout.close()

# general topics
beta = np.loadtxt(os.path.join(fit_dir, "beta-%s.dat" % iter_id))[:,1:].T
fout = open(os.path.join(fit_dir, "topics_general_%s.dat" % iter_id), 'w+')
for k in range(len(beta)):
    terms = dict(zip(vocab, beta[k]))
    topterms = sorted(terms, key=lambda t: -terms[t])[:N_terms]
    top3 = ' / '.join(topterms[:3])
    fout.write("general #%d\t%s\n" % (k, ' / '.join(topterms)))
fout.close()


#TODO: documents and eventness
'''
### DOCUMENTS
# for each type of component, pick top N_docs documents
# metadata
doc_entity = {}
for line in open(os.path.join(data_dir, 'meta.tsv')):
    doc, entity, time = [int(t) for t in line.strip().split('\t')]
    doc_entity[doc] = entity

# select entity docs
entity_docs = defaultdict(dict)
for line in open(os.path.join(fit_dir, 'zeta-%04d.dat' % iter_id)):
    doc, val = line.strip().split('\t')
    doc = int(doc)
    val = float(val)
    entity = doc_entity[doc]
    if entity not in emap:
        continue
    if len(entity_docs[entity]) < N_docs:
        entity_docs[entity][doc] = val
    elif val > max(entity_docs[entity].values()):
        # rm lowest val doc if needed
        del entity_docs[entity][min(entity_docs[entity], \
            key=entity_docs[entity].get)]

        # add in more relevant doc
        entity_docs[entity][doc] = val

# select event docs
event_docs = defaultdict(dict)
for line in open(os.path.join(fit_dir, 'epsilon-%04d.dat' % iter_id)):
    doc, time, val, decayedval = line.strip().split('\t')
    doc = int(doc)
    time = int(time)
    val = float(decayedval)
    if len(event_docs[time]) < N_docs:
        event_docs[time][doc] = val
    elif val > max(event_docs[time].values()):
        # rm lowest val doc if needed
        del event_docs[time][min(event_docs[time], \
            key=event_docs[time].get)]

        # add in more relevant doc
        event_docs[time][doc] = val

# select general docs
general_docs = defaultdict(dict)
for line in open(os.path.join(fit_dir, 'theta-%04d.dat' % iter_id)):
    doc, vals = line.strip().split('\t', 1)
    doc = int(doc)
    vals = [float(v) for v in vals.split('\t')]
    if len(general_docs[0]) < N_docs:
        for k in range(len(vals)):
            general_docs[k][doc] = vals[k]
    else:
        for k in range(len(vals)):
            if vals[k] > max(general_docs[k].values()):
                # rm lowest val doc if needed
                del general_docs[k][min(general_docs[k], \
                    key=general_docs[k].get)]

                # add in more relevant doc
                general_docs[k][doc] = val

# aggregate all relevant documents
docs = set()
for entity in entity_docs:
    docs = docs | set(entity_docs[entity].keys())
for time in event_docs:
    docs = docs | set(event_docs[time].keys())
for k in general_docs:
    docs = docs | set(general_docs[k].keys())

# find sqlite doc ids
doc_idx_map = {}
for line in open(os.path.join(data_dir, 'doc_map.tsv')):
    fit_id, sqlite_id = line.strip().split('\t')
    if int(fit_id) in docs:
        doc_idx_map[int(fit_id)] = sqlite_id

# create each doc object
dmap = {}
emap_names = dict(zip([e.name for e in emap.values()], emap.values()))
for fit_id in docs:
    # find key from data_dir
    idx = doc_idx_map[fit_id]

    entity = doc_entity[fit_id]
    # special case: needs a name, but no link in UI
    if entity not in emap:
        ent = Entity(name=entity_names[entity], doc_count=0)
        ent.save()
        emap[entity] = ent

    # grab misc info from db
    cur.execute('SELECT date, \"to\", subject, body FROM docs WHERE id=\"%s\"' % idx)
    print idx
    dt, recipients, subj, body = cur.fetchone()

    dt = [int(t) for t in dt.split('-')]
    doc = Document(id=idx, sender=emap[entity], date=date(dt[0], dt[1], dt[2]), \
        subject=subj, message=body)
    recipients = recipients.replace('\t', '\n')
    recipients = recipients.replace('  ', '\n')
    for r in recipients.split('\t'):
        r = r.strip()
        if r not in emap_names:
            ent = Entity(name=r, doc_count=0)
            ent.save()
            emap_names[r] = ent
        doc.recipients.add(emap_names[r])

    doc.save()
    dmap[fit_id] = doc

# gather doc params for each document
for line in open(os.path.join(fit_dir, 'zeta-%04d.dat' % iter_id)):
    doc, val = line.strip().split('\t')
    doc = int(doc)
    if doc not in docs:
        continue
    val = float(val)
    if doc_entity[doc] not in ent_topics:
        continue
    topic = ent_topics[doc_entity[doc]].topic
    dt = DocTopic(doc=dmap[doc], topic=topic, value=val)
    dt.save()

for line in open(os.path.join(fit_dir, 'epsilon-%04d.dat' % iter_id)):
    doc, date, val = line.strip().split('\t')[:3]
    doc = int(doc)
    if doc not in docs:
        continue
    dt = DocTopic(doc=dmap[doc], topic=event_topics[int(date)], \
        value=float(val))
    dt.save()

for line in open(os.path.join(fit_dir, 'theta-%04d.dat' % iter_id)):
    doc, vals = line.strip().split('\t',1)
    doc = int(doc)
    if doc not in docs:
        continue
    vals = [float(v) for v in vals.split('\t')]
    for t in range(len(vals)):
        dt = DocTopic(doc=dmap[doc], topic=general_topics[t], \
            value=vals[t])
        dt.save()



'''
