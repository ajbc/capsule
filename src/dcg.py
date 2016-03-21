import sys
import numpy as np

truth = sys.argv[1]
fit = sys.argv[2]

#head fit/full_04/events_final.tsv
#rank    day event.strength
#1   0   0.000320
#2   48  0.000271
#3   67  0.000241
#4   42  0.000240


#head ~/cables/src/poisson_event_detection/dat/sim/simulated_eoccur.tsv
#0   0.148044
#1   0.023989

true_events = {}
for t in open(truth):
    day, val = t.strip().split('\t')
    true_events[int(day)] = float(val)

ndcg = 0
ndcr = 0
i = 1
for v in sorted(true_events.values(), reverse=True):
    if i == 1:
        ndcg += v
        ndcr += v
    else:
        ndcg += v / (np.log(i) /np.log(2))
        ndcr += v / i
    i += 1

header = True
dcg = 0
dcr = 0
for line in open(fit):
    if header:
        header = False
        continue
    rank, day, strength = line.strip().split('\t')
    rank = int(rank)
    day = int(day)
    if rank == 1:
        dcg += true_events[day]
        dcr += true_events[day]
    else:
        dcg += true_events[day] / (np.log(rank) /np.log(2))
        dcr += true_events[day] / rank

print "DCG:", dcg, (dcg/ndcg)
print "DCR", dcr, (dcr/ndcr)
