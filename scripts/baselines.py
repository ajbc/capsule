import sys
import numpy as np
from collections import defaultdict
from scipy.stats import multivariate_normal

# read in args
data = sys.argv[1]
out = sys.argv[2]

class TT:
    def __init__(self):
        self.dict = {}
    def add(self, key, val):
        if len(self.dict) < 100:
            self.dict[key] = val
        else:
            s = sorted(self.dict.keys(), key=lambda x: self.dict[x])[0]
            if val > self.dict[s]:
                self.dict.pop(s, None)
                self.dict[key] = val
    def ts(self):
        return ','.join([str(k) for k in self.dict.keys()])


# read in meta
meta = np.loadtxt(data + "/meta.tsv")
dates = set(meta[:,2])
senddate = {}
docs = defaultdict(list)
for doc, author, date in meta:
    senddate[int(doc)] = int(date)
    docs[int(date)].append(int(doc))

# read in training data
training = np.loadtxt(data + "/train.tsv")
V = int(max(training[:,1]) + 1)
ave_wordcounts = np.zeros(V)
docsets = defaultdict(set)
for dp in training:
    doc, term, count = [int(d) for d in dp]
    ave_wordcounts[term] += count
    docsets[term].add(doc)

doc_counts = np.zeros(V)
for v in docsets:
    doc_counts[v] = len(docsets[v])
doc_counts[doc_counts == 0] = 1e-20

Ndocs = len(set(training[:,0]))
tfidf = ave_wordcounts * np.log(Ndocs) / doc_counts
ave_wordcounts /= Ndocs

### calculate baselines
print "word count based"
# random
fout = open(out + '/random.topdocs.dat', 'w+')
for date in dates:
    ds = np.random.choice(docs[date], 10, replace=False)
    fout.write(','.join([str(d) for d in ds]) + '\n')
fout.close()

# wordcount-based metrics
D = len(dates)
greatest_term_outlier = np.zeros(D)
gto_docs = defaultdict(TT)
greatest_term_outlier_tfidf = np.zeros(D)
gtot_docs = defaultdict(TT)
total_dev = np.zeros(D)
total_dev_tfidf = np.zeros(D)
sum_deviation = np.zeros(D)
total_day = np.zeros(D)
words_per_day = np.zeros(D)
doc_devs = defaultdict(float)
doc_devs_tfidf = defaultdict(float)
doc_dev_counts = defaultdict(int)
for dp in training:
    doc, term, count = [int(d) for d in dp]
    deviation = count - ave_wordcounts[term]
    doc_devs[doc] += abs(deviation)
    doc_devs_tfidf[doc] += abs(deviation * tfidf[term])
    doc_dev_counts[doc] += 1
    gto_docs[senddate[doc]].add(doc, deviation)
    gtot_docs[senddate[doc]].add(doc, deviation * tfidf[term])

    if deviation > greatest_term_outlier[senddate[doc]]:
        greatest_term_outlier[senddate[doc]] = deviation
    if deviation*tfidf[term] > greatest_term_outlier_tfidf[senddate[doc]]:
        greatest_term_outlier_tfidf[senddate[doc]] = deviation*tfidf[term]
    total_dev[senddate[doc]] += abs(deviation)
    total_dev_tfidf[senddate[doc]] += abs(deviation) * tfidf[term]
    words_per_day[senddate[doc]] += 1

max_doc_outlier = np.zeros(D)
max_doc_outlier_tfidf = np.zeros(D)
total_doc_dev = np.zeros(D)
total_doc_dev_tfidf = np.zeros(D)
datewise_doc_counts = np.zeros(D)
md_docs = defaultdict(TT)
mdt_docs = defaultdict(TT)
for doc in doc_devs:
    doc_devs[doc] /= doc_dev_counts[doc]
    doc_devs_tfidf[doc] /= doc_dev_counts[doc]

    md_docs[senddate[doc]].add(doc, doc_devs[doc])
    mdt_docs[senddate[doc]].add(doc, doc_devs_tfidf[doc])

    if doc_devs[doc] > max_doc_outlier[senddate[doc]]:
        max_doc_outlier[senddate[doc]] = doc_devs[doc]
    if doc_devs_tfidf[doc] > max_doc_outlier_tfidf[senddate[doc]]:
        max_doc_outlier_tfidf[senddate[doc]] = doc_devs_tfidf[doc]
    total_doc_dev[senddate[doc]] += doc_devs[doc]
    total_doc_dev_tfidf[senddate[doc]] += doc_devs_tfidf[doc]
    datewise_doc_counts[senddate[doc]] += 1


# greatest single term wordcount outlier for the day
fout = open(out + '/word_outlier.dat', 'w+')
foutd = open(out + '/word_outlier.docs.dat', 'w+')
for date in dates:
    fout.write('%f\n' % greatest_term_outlier[date])
    foutd.write(gto_docs[date].ts() + '\n')
fout.close()
foutd.close()

# greatest wordcount outlier for the day, weighted by tfidf
fout = open(out + '/word_outlier_tfidf.dat', 'w+')
foutd = open(out + '/word_outlier_tfidf.docs.dat', 'w+')
for date in dates:
    fout.write('%f\n' % greatest_term_outlier_tfidf[date])
    foutd.write(gtot_docs[date].ts() + '\n')
fout.close()
foutd.close()

# greatest doc wordcount outlier for the day
fout = open(out + '/doc_outlier.dat', 'w+')
foutd = open(out + '/doc_outlier.docs.dat', 'w+')
for date in dates:
    fout.write('%f\n' % max_doc_outlier[date])
    foutd.write(md_docs[date].ts() + '\n')
fout.close()
foutd.close()

# greatest doc wordcount outlier for the day
fout = open(out + '/doc_outlier_tfidf.dat', 'w+')
foutd = open(out + '/doc_outlier_tfidf.docs.dat', 'w+')
for date in dates:
    fout.write('%f\n' % max_doc_outlier_tfidf[date])
    foutd.write(mdt_docs[date].ts() + '\n')
fout.close()
foutd.close()

# total wordcount deviation for the day
fout = open(out + '/total_wc_deviation.dat', 'w+')
for date in dates:
    fout.write('%f\n' % total_dev[date])
fout.close()

# total wordcount deviation for the day, weighted by tfidf
fout = open(out + '/total_wc_deviation_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % total_dev_tfidf[date])
fout.close()

# average wordcount deviation for the day
fout = open(out + '/ave_wc_deviation.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_dev[date] / words_per_day[date]))
fout.close()

# average wordcount deviation for the day, weighted by tfidf
fout = open(out + '/ave_wc_deviation_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_dev_tfidf[date] / words_per_day[date]))
fout.close()


# total doc wc deviation for the day
fout = open(out + '/total_doc_deviation.dat', 'w+')
for date in dates:
    fout.write('%f\n' % total_doc_dev[date])
fout.close()

# total wordcount deviation for the day, weighted by tfidf
fout = open(out + '/total_doc_deviation_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % total_doc_dev_tfidf[date])
fout.close()

# average wordcount deviation for the day
fout = open(out + '/ave_doc_deviation.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_doc_dev[date] / datewise_doc_counts[date]))
fout.close()

# average wordcount deviation for the day, weighted by tfidf
fout = open(out + '/ave_doc_deviation_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_doc_dev_tfidf[date] / datewise_doc_counts[date]))
fout.close()

