import sys
from collections import defaultdict

v = defaultdict(int)
for line in open(sys.argv[1]):
    v[sum([float(x) != 0.01 for x in line.strip().split('\t')])] += 1

for val in sorted(v):
    #print "%d : %s" % (val, 'X'*v[val])
    print "%d : %d" % (val, v[val])
