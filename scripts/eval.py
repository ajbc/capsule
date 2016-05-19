import sys
import numpy as np

truth = {}
for line in open(sys.argv[1]):
    a,b = line.strip().split('\t')
    if a == "day":
        continue
    truth[int(a)] = 1.0/float(b)

fite = {}
inv = sys.argv[3] == "inv"
for line in open(sys.argv[2]):
    v = float(line.strip())
    fite[len(fite)] = 1.0/v if inv else v

t_sorted = sorted(truth.keys(), key=lambda x: truth[x])
f_sorted = sorted(fite.keys(), key=lambda x: fite[x])
AUC = 0.0
ideal_AUC = 0.0
for d in range(len(t_sorted)):
    inter = len(set(t_sorted[:d+1]) & set(f_sorted[:d+1]))
    ideal_AUC += d
    AUC += inter

print AUC / ideal_AUC
