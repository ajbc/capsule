import sys
import os
from os.path import join
import numpy as np

truth = sys.argv[1]
fit = sys.argv[2]

### load ground truth
entity_file = open(join(truth, "simulated_entities.tsv"))
entities = {}
for line in entity_file:
    if "entity" in line:
        continue
    tok = [float(t) for t in line.split('\t')]
    entities[int(tok[0])] = np.array(tok[1:])

events_file = open(join(truth, "simulated_events.tsv"))
events = {}
for line in events_file:
    if "time" in line:
        continue
    tok = [float(t) for t in line.split('\t')]
    events[int(tok[0])] = np.array(tok[1:])


### for each logged iteration, compute RMSE
fout = open(join(fit,"RMSE.dat"), 'w+')
iters = set()
for f in os.listdir(fit):
    if "_" in f and "." in f and 'final' not in f:
        iters.add(int(f.split('.')[0].split('_')[1]))

fout.write("iteration\tentity.RMSE\teoccur.RMSE\tevents.RMSE\n")
K = 6
for i in sorted(iters):
    entRMSE = 0.0
    entity_count = 0
    for line in open(join(fit, "entities_%04d.tsv" % i)):
        tok = np.array([float(t) for t in line.split('\t')])
        try:
            entRMSE += ((entities[entity_count] - tok)**2).sum()
            entity_count += 1
        except:
            continue
    entRMSE = np.sqrt(entRMSE/(K*entity_count))

    eocRMSE = 0.0
    evtRMSE = 0.0
    event_count = 0
    eoccur = [float(eoc.strip()) for eoc in open(join(fit, "eoccur_%04d.tsv" % i)).readlines()]
    for line in open(join(fit, "events_%04d.tsv" % i)):
        if event_count in events:
            eocRMSE += (1-eoccur[event_count])**2
            tok = np.array([float(t) for t in line.split('\t')])
            evtRMSE += ((events[event_count] - tok)**2).sum()
        else:
            eocRMSE += (0-eoccur[event_count])**2
        event_count += 1
    eocRMSE = np.sqrt(eocRMSE/event_count)
    evtRMSE = np.sqrt(evtRMSE/(K*len(events)))

    fout.write("%d\t%f\t%f\t%f\n" % (i, entRMSE, eocRMSE, evtRMSE))
