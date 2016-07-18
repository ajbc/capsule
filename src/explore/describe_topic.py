import sys

vocab = [line.strip() for line in open(sys.argv[1]).readlines()]
topics = open(sys.argv[2]).readlines()
topic = int(sys.argv[3])


topic = [float(v) for v in [topics[i].strip().split('\t')[topic+2] for i in range(len(vocab))]]

topic_terms = sorted(zip(topic, vocab), reverse=True)

for t in topic_terms[:20]:
    print t



