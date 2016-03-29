import sys

vocab = [line.strip() for line in open(sys.argv[1]).readlines()]
events = open(sys.argv[2]).readlines()
event = int(sys.argv[3])

terms = [float(v) for v in events[event].strip().split('\t')[2:]]

event_terms = sorted(zip(terms, vocab), reverse=True)

for t in event_terms[:100]:
    print t



