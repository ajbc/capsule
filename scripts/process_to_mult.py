import sys
from collections import defaultdict

fout = open(sys.argv[2], 'w+')
docs = defaultdict(dict)
for line in open(sys.argv[1]):
    doc, term, count = [int(t) for t in line.strip().split('\t')]
    docs[doc][term] = count

for doc in range(max(docs)+1):
    line = "%d" % len(docs[doc])
    for term in sorted(docs[doc]):
        line += " %d:%d" % (term, docs[doc][term])
    fout.write(line + '\n')
fout.close()
