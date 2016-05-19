import sys
import numpy as np
from collections import defaultdict
from scipy.stats import multivariate_normal

# read in args
data = sys.argv[1]
out = sys.argv[2]

# read in meta
meta = np.loadtxt(data + "/meta.tsv")
dates = set(meta[:,2])
entities = set(meta[:,1])
authors = {}
senddate = {}
for doc, author, date in meta:
    authors[doc] = author
    senddate[doc] = date

# read in training data
training = np.loadtxt(data + "/train.tsv")
V = int(max(training[:,1]) + 1)
N = len(entities)
ave_wordcounts = np.zeros(V)
ave_ent_wordcounts = np.zeros((N,V))
docsets = defaultdict(set)
ent_doccounts = np.zeros((N,V))
for dp in training:
    doc, term, count = dp
    ave_wordcounts[term] += count
    ave_ent_wordcounts[authors[doc], term] += count
    docsets[term].add(doc)
    ent_doccounts[authors[doc],term] += 1

doc_counts = np.zeros(V)
for v in docsets:
    doc_counts[v] = len(docsets[v])
doc_counts[doc_counts == 0] = 1e-20

tfidf = ave_wordcounts * np.log(len(set(training[:,0])) / doc_counts)
ave_wordcounts /= len(dates)
ent_doccounts[ent_doccounts == 0] = 1e-10
ave_ent_wordcounts = ave_ent_wordcounts / ent_doccounts

### calculate baselines
print "word count based"
# random
fout = open(out + '/random.dat', 'w+')
for date in dates:
    fout.write('%f\n' % np.random.random())
fout.close()

# wordcount-based metrics
D = len(dates)
greatest_outlier = np.zeros(D)
greatest_outlier_tfidf = np.zeros(D)
greatest_outlier_entities = np.zeros((D, N))
greatest_outlier_entities_tfidf = np.zeros((D, N))
total_dev = np.zeros(D)
total_dev_tfidf = np.zeros(D)
total_dev_entities = np.zeros((D, N))
total_dev_entities_tfidf = np.zeros((D, N))
sum_deviation = np.zeros(D)
total_day = np.zeros(D)
words_per_day = np.zeros(D)
for dp in training:
    doc, term, count = dp
    deviation = count - ave_wordcounts[term]
    ent_deviation = count - ave_ent_wordcounts[authors[doc], term]
    if deviation > greatest_outlier[senddate[doc]]:
        greatest_outlier[senddate[doc]] = deviation
    if deviation*tfidf[term] > greatest_outlier_tfidf[senddate[doc]]:
        greatest_outlier_tfidf[senddate[doc]] = deviation*tfidf[term]
    if ent_deviation > greatest_outlier_entities[senddate[doc], authors[doc]]:
        greatest_outlier_entities[senddate[doc], authors[doc]] = ent_deviation
    if ent_deviation*tfidf[term] > greatest_outlier_entities_tfidf[senddate[doc], authors[doc]]:
        greatest_outlier_entities_tfidf[senddate[doc], authors[doc]] = ent_deviation * tfidf[term]
    total_dev[senddate[doc]] += abs(deviation)
    total_dev_tfidf[senddate[doc]] += abs(deviation) * tfidf[term]
    total_dev_entities[senddate[doc],authors[doc]] += abs(ent_deviation)
    total_dev_entities_tfidf[senddate[doc],authors[doc]] += abs(ent_deviation) * tfidf[term]
    words_per_day[senddate[doc]] += 1

# greatest wordcount outlier for the day
fout = open(out + '/word_outlier.dat', 'w+')
for date in dates:
    fout.write('%f\n' % greatest_outlier[date])
fout.close()

# greatest wordcount outlier for the day, weighted by tfidf
fout = open(out + '/word_outlier_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % greatest_outlier_tfidf[date])
fout.close()

# total wordcount deviation for the day
fout = open(out + '/total_deviation.dat', 'w+')
for date in dates:
    fout.write('%f\n' % total_dev[date])
fout.close()

# total wordcount deviation for the day, weighted by tfidf
fout = open(out + '/total_deviation_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % total_dev_tfidf[date])
fout.close()

# average wordcount deviation for the day
fout = open(out + '/ave_deviation.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_dev[date] / words_per_day[date]))
fout.close()

# average wordcount deviation for the day, weighted by tfidf
fout = open(out + '/ave_deviation_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_dev_tfidf[date] / words_per_day[date]))
fout.close()

## ENTITIES
# entity average of greatest wordcount outlier for the day
fout = open(out + '/word_outlier_entity_ave.dat', 'w+')
fout2 = open(out + '/word_outlier_entity_ave_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % sum(greatest_outlier_entities[date,:]))
    fout2.write('%f\n' % sum(greatest_outlier_entities_tfidf[date,:]))
fout.close()
fout2.close()

# entity max of greatest wordcount outlier for the day
fout = open(out + '/word_outlier_entity_max.dat', 'w+')
fout2 = open(out + '/word_outlier_entity_max_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % max(greatest_outlier_entities[date,:]))
    fout2.write('%f\n' % max(greatest_outlier_entities_tfidf[date,:]))
fout.close()
fout2.close()

# average of entity wordcount deviations for day
fout = open(out + '/ave_deviation_entity.dat', 'w+')
fout2 = open(out + '/ave_deviation_entity_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % sum(total_dev_entities[date,:]))
    fout2.write('%f\n' % sum(total_dev_entities_tfidf[date,:]))
fout.close()
fout2.close()

# maximum of entity wordcount deviations for day
fout = open(out + '/max_deviation_entity.dat', 'w+')
fout2 = open(out + '/max_deviation_entity_tfidf.dat', 'w+')
for date in dates:
    fout.write('%f\n' % max(total_dev_entities[date,:]))
    fout2.write('%f\n' % max(total_dev_entities_tfidf[date,:]))
fout.close()
fout2.close()


## LDA
print "LDA"
lda = np.loadtxt(out + "/lda/final.gamma")
ave = np.mean(lda, 0)
entity_sum = np.zeros((N, len(ave)))
entity_counts = np.zeros(N)
for doc in range(len(lda)):
    entity_sum[authors[doc]] += lda[doc]
    entity_counts[authors[doc]] += 1
greatest_outlier = np.zeros(D)
greatest_ent_outlier = np.zeros(D)
total_dev = np.zeros(D)
total_ent_dev = np.zeros(D)
doc_count = np.zeros(D)
for doc in range(len(lda)):
    dev = np.sum(abs(lda[doc] - ave))
    ent_dev = np.sum(abs(lda[doc] - entity_sum[authors[doc]]/entity_counts[authors[doc]]))
    if dev > greatest_outlier[senddate[doc]]:
        greatest_outlier[senddate[doc]] = dev
    if ent_dev > greatest_ent_outlier[senddate[doc]]:
        greatest_ent_outlier[senddate[doc]] = ent_dev
    total_dev[senddate[doc]] += dev
    total_ent_dev[senddate[doc]] += ent_dev
    doc_count[senddate[doc]] += 1

# lda: greatest single doc deviation
fout = open(out + '/doc_outlier_lda.dat', 'w+')
for date in dates:
    fout.write('%f\n' % greatest_outlier[date])
fout.close()

# lda: ave deviation for day
fout = open(out + '/ave_dev_lda.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_dev[date] / doc_count[date]))
fout.close()

# lda: greatest single doc deviation (relative to entity ave)
fout = open(out + '/doc_ent_outlier_lda.dat', 'w+')
for date in dates:
    fout.write('%f\n' % greatest_ent_outlier[date])
fout.close()

# lda: total deviation for day (relative to entity ave)
fout = open(out + '/ave_ent_dev_lda.dat', 'w+')
for date in dates:
    fout.write('%f\n' % (total_ent_dev[date] / doc_count[date]))
fout.close()

## MULTINOMIAL GAUSSIANS
print "Multinomial Gaussians"
# use mu (mean) from previous computations
mu_wordcounts = ave_wordcounts # V vector
mu_wc_tfidf = ave_wordcounts * tfidf # V
mu_wc_ent = ave_ent_wordcounts # (N, V)
mu_wc_ent_tfidf = ave_ent_wordcounts # (N, V)
for ent in range(len(mu_wc_ent_tfidf)):
    mu_wc_ent_tfidf[ent] *= tfidf
mu_lda = ave
mu_lda_ent = entity_sum
for ent in range(len(mu_lda_ent)):
    mu_lda_ent[ent] /= entity_counts[ent]

# compute sigma
print "computing sigmas"
'''docs = np.zeros((len(set(meta[:,0])), V))
for doc, term, count in training:
    docs[doc, term] = count
sigma_wordcounts = np.zeros((V,V))
for doc in docs:
    v = np.matrix(doc - mu_wordcounts)
    sigma_wordcounts += v.T * v
sigma_wordcounts /= len(docs)'''
sigma_lda = np.zeros((10,10))
sigma_lda_ent = defaultdict(lambda: np.zeros((10,10)))
ent_count = defaultdict(int)
di = 0
for doc in lda:
    v = np.matrix(doc - mu_lda)
    sigma_lda += v.T * v
    v = np.matrix(doc - mu_lda_ent[authors[di]])
    sigma_lda_ent[authors[di]] += v.T * v
    di += 1
sigma_lda /= len(lda)
for ent in ent_count:
    sigma_lda_ent[ent] /= ent_count[ent]


# compute doc probabilities
print "computing doc probabilities"
'''wordcounts = np.zeros(D)
for doc in range(len(docs)):
    p = multivariate_normal.pdf(docs[doc], mu_wordcounts, sigma_wordcounts)
    ip_wordcounts = 1.0 / p#max(1e-10, multivariate_normal.pdf(docs[doc], mu_wordcounts, sigma_wordcounts))
    if ip_wordcounts > wordcounts[senddate[doc]]:
        wordcounts[senddate[doc]] = ip_wordcounts'''
lda_MG_max = np.zeros(D)
lda_MG_ave = np.zeros(D)
lda_ent_MG_max = np.zeros(D)
lda_ent_MG_ave = np.zeros(D)
for doc in range(len(lda)):
    v = multivariate_normal.pdf(lda[doc], mu_lda, sigma_lda)
    ip_lda = 1.0 / v if v!=0 else float('inf')
    ip_lda_ent = 1.0 / multivariate_normal.pdf(lda[doc], mu_lda_ent[authors[doc]], sigma_lda_ent[authors[doc]])
    if ip_lda > lda_MG_max[senddate[doc]]:
        lda_MG_max[senddate[doc]] = ip_lda
    if ip_lda_ent > lda_ent_MG_max[senddate[doc]]:
        lda_ent_MG_max[senddate[doc]] = ip_lda_ent
    lda_MG_ave[senddate[doc]] += ip_lda
    lda_ent_MG_ave[senddate[doc]] += ip_lda_ent


# greatest wordcount MG doc outlier for the day
'''fout = open(out + '/word_outlier_MG.dat', 'w+')
for date in dates:
    fout.write('%f\n' % wordcounts[date])
fout.close()'''

# LDA multinomial gaussian
fout = open(out + '/lda_MG_max.dat', 'w+')
fout2 = open(out + '/lda_MG_ave.dat', 'w+')
fout3 = open(out + '/lda_ent_MG_max.dat', 'w+')
fout4 = open(out + '/lda_ent_MG_ave.dat', 'w+')
for date in dates:
    fout.write('%e\n' % lda_MG_max[date])
    fout2.write('%e\n' % (lda_MG_ave[date] / doc_count[date]))
    fout3.write('%e\n' % lda_ent_MG_max[date])
    fout4.write('%e\n' % (lda_ent_MG_ave[date] / doc_count[date]))
fout.close()
fout2.close()
fout3.close()
fout4.close()


# multinomial gaussian (compute mu and sigma) for
    # wordcounts
    # entity wordcounts (one gaussian per entity)
    # tf-idf weighted wordcounts
    # tf-idf weighted entity wordcounts (one fit per entity)
    # lda topics
    # lda topics per entity
# and then for each document, compute it's probability and report
    # greatest inverse doc prob for the day
    # average inverse doc prob for the day
# so this gives us 12 baselines using multinomial gaussians

