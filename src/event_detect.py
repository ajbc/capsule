import numpy as np
import shutil, os, sys
from datetime import datetime as dt
from scipy.special import gammaln, digamma

import warnings #TODO rm
warnings.filterwarnings('error')
from collections import defaultdict #TODO rm

## helper functions

# softmax
def M(x):
    x[x > np.log(sys.float_info.max)] = np.log(sys.float_info.max)
    rv = np.log(1.0 + np.exp(x))
    rv[rv == 0] = (sys.float_info.min)**0.5
    return rv

# derivative of softmax
def dM(x):
    x[x > np.log(sys.float_info.max)] = np.log(sys.float_info.max)
    return np.exp(x) / (1.0 + np.exp(x))

# inverse of softmax
def iM(x):
    return np.log(np.exp(x) - 1.0)


# sigmoid
def S(x):
    return 1.0 / (1.0 + np.exp(-x))

# derivative of sigmoid
def dS(x):
    return - np.exp(-x) / (1.0 + np.exp(-x))**2

# inverse of sigmoid
def iS(x):
    return -1.0 *np.log(1.0 / x - 1.0)


#TODO: do we need this? can we just use gammaln
def lngamma(val):
    if isinstance(val, float):
        return gammaln(val) if val > sys.float_info.min else gammaln(sys.float_info.max)
    else:
        val[val < sys.float_info.min] = sys.float_info.min
        return gammaln(val)

def cov(a, b):
    # the below was in the figure i sent to dave
    v = (a - sum(a)/a.shape[0]) * (b - sum(b)/b.shape[0])
    return sum(v)/v.shape[0]

def var(a):
    rv = cov(a, a)
    rv[rv == 0] = sys.float_info.min #TODO: is this needed?
    return rv

def pGamma(x, a, b):
    if (x == 0).any() or (np.array(b) == 0).any():
        x[x==0] = sys.float_info.min
    return a*b*np.log(b) - lngamma(a*b) + (a*b-1.0)*np.log(x) - b*x

def qgGamma(x, a, b):
    dMb = dM(b)
    b = M(b)
    dMa = dM(a) * b
    a = M(a) * b

    g_a = dMa * (np.log(b) - digamma(a) + np.log(x))
    g_b = dMb * a * ((np.log(b) + 1.0) - digamma(a) + np.log(x) - x)

    return (pGamma(x, a, b), g_a, g_b)

def EGamma(a, b):
    num = M(a) * M(b)
    den = M(b)

    num[num > 1] = 1 #TODO: do we really need this threshold?
    return num / den

def pBernoulli(x, p):
    if (isinstance(p, float) and p==1) or (isinstance(p, np.ndarray) and p.all()==1):
        return x*np.log(p)
    return x*np.log(p) + (1.0-x)*np.log(1.0-p)

def qgBernoulli(x, p):
    dSp = dS(p)
    #print p
    p = S(p)
    #print p
    #print "---"
    #TODO: do we need this?
    p[p==1] = 0.9999
    return (pBernoulli(x, p), dSp * (x/p + (1.0-x)/(1.0-p)))


## Classes

class Document:
    def __init__(self, id, day, sparse_rep):
        self.id = id
        self.day = day
        self.rep = sparse_rep
        self.rep[self.rep == 0] = sys.float_info.min


class Corpus:
    def __init__(self, content_filename, time_filename):
        self.docs = []
        times = [int(t.strip()) for t in open(time_filename).readlines()]
        self.days = sorted(set(times))
        self.dimension = 0
        for line in open(content_filename):
            rep = np.array([float(v) for v in line.strip().split('\t')])
            if self.dimension == 0:
                self.dimension = len(rep)
            elif self.dimension != len(rep):
                print "Data malformed; document representations not of equal length"
                sys.exit(-1)
            self.docs.append(Document(len(self.docs), times.pop(0), rep))

        self.validation = set()
        while len(self.validation) < 0.05 * len(self.docs):
            self.validation.add(self.random_doc())

    def day_count(self):
        return len(self.days)

    def num_docs(self):
        return len(self.docs)

    def random_doc(self):
        return self.docs[np.random.randint(len(self.docs))]


class Parameters:
    def __init__(self, outdir, batch_size, num_samples, save_freq, \
        conv_thresh, max_iter, tau, kappa, \
        a_ent, b_ent, a_evn, b_evn, b_doc, eoc, event_duration, \
        content, time):
        self.outdir = outdir
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.save_freq = save_freq
        self.convergence_thresh = conv_thresh
        self.max_iter = max_iter

        self.tau = tau
        self.kappa = kappa

        self.a_entity = a_ent
        self.b_entity = b_ent
        self.a_events = a_evn
        self.b_events = b_evn
        self.b_docs = b_doc
        self.l_eoccur = eoc

        self.d = event_duration

        self.content = content
        self.time = time

    def save(self, seed):
        f = open(os.path.join(self.outdir, 'settings.dat'), 'w+')

        f.write("random seed:\t%d\n" % seed)
        f.write("batch size:\t%d\n" % self.batch_size)
        f.write("number of samples:\t%d\n" % self.num_samples)
        f.write("save frequency:\t%d\n" % self.save_freq)
        f.write("convergence threshold:\t%f\n" % self.convergence_thresh)
        f.write("max # of iterations:\t%d\n" % self.max_iter)
        f.write("tau:\t%d\n" % self.tau)
        f.write("kappa:\t%f\n" % self.kappa)
        f.write("a_entity:\t%f\n" % self.a_entity)
        f.write("b_entity:\t%f\n" % self.b_entity)
        f.write("a_events:\t%f\n" % self.a_events)
        f.write("b_events:\t%f\n" % self.b_events)
        f.write("b_docs:\t%f\n" % self.b_docs)
        f.write("prior on event occurance:\t%f\n" % self.l_eoccur)
        f.write("data, content:\t%s\n" % self.content)
        f.write("data, times:\t%s\n" % self.time)

        f.close()

    def f(self, a, c):
        if a > c or c >= (a+self.d):
            return 0
        return (1 - ((c-a)/self.d))


class Model:
    def __init__(self, data, params):
        self.data = data
        self.params = params

    def init(self):
        # free variational parameters
        self.a_entity = np.ones((1,self.data.dimension)) * iM(self.params.a_entity)
        self.b_entity = np.ones((1,self.data.dimension)) * iM(self.params.b_entity)
        self.a_events = np.ones((self.data.day_count(), self.data.dimension)) * \
            iM(self.params.a_events)
        self.b_events = np.ones((self.data.day_count(), self.data.dimension)) * \
            iM(self.params.b_events)
        self.l_eoccur = np.ones(self.data.day_count()) * iS(self.params.l_eoccur)

        # expected values of goal model parameters
        self.entity = EGamma(self.a_entity, self.b_entity)
        self.events = EGamma(self.a_events, self.b_events)
        self.eoccur = S(self.l_eoccur)

        self.likelihood_decreasing_count = 0

    def compute_likelihood(self):
        log_likelihood = 0
        LL = np.zeros(self.data.dimension)
        f_array = np.zeros((self.data.day_count(),1))
        for doc in self.data.validation:
            for day in range(self.data.day_count()):
                f_array[day] = self.params.f(self.data.days[day], doc.day) * self.eoccur[day]
            #print self.entity
            #print f_array*self.events
            #print f_array, self.events
            #print sum(f_array*self.events)
            doc_params = self.entity + sum(f_array*self.events)
            log_likelihood += np.sum(pGamma(doc.rep, self.params.b_docs * doc_params, self.params.b_docs))
            LL += np.sum(pGamma(doc.rep, self.params.b_docs * doc_params, self.params.b_docs), 0)
        print "LL", LL
        return log_likelihood

    def converged(self, iteration):
        if iteration == 0:
            self.likelihood = -sys.float_info.max
            flog = open(os.path.join(self.params.outdir, 'log.dat'), 'w+')
            flog.write("iteration\tlikelihood\tchange\n")
            return False

        self.old_likelihood = self.likelihood
        self.likelihood = self.compute_likelihood()
        delta = (self.likelihood - self.old_likelihood) / \
            abs(self.old_likelihood)

        flog = open(os.path.join(self.params.outdir, 'log.dat'), 'a')
        flog.write("%d\t%f\t%f\n" % (iteration, self.likelihood, delta))
        print "%d\t%f\t%f" % (iteration, self.likelihood, delta)

        if delta < 0:
            print "likelihood decreasing (bad)"
            self.likelihood_decreasing_count += 1
            if iteration > 30 and self.likelihood_decreasing_count == 3:
                print "STOP: 3 consecutive iterations of increasing likelihood"
                return True
            return False
        else:
            self.likelihood_decreasing_count = 0

        #if iteration > 20 and delta < self.params.convergence_thresh:
        if iteration > 20 and delta < self.params.convergence_thresh:
            print "STOP: model converged!"
            return True
        if iteration == self.params.max_iter:
            print "STOP: iteration cap reached"
            return True
        return False

    def save(self, tag):
        fout = open(os.path.join(self.params.outdir, "entities_%s.tsv" % tag), 'w+')
        fout.write(('\t'.join(["%f"]*len(self.entity[0]))+'\n') % tuple(self.entity[0]))
        fout.close()

        fout = open(os.path.join(self.params.outdir, "events_%s.tsv" % tag), 'w+')
        for i in range(len(self.events)):
            fout.write(('\t'.join(["%f"]*len(self.events[i])) +'\n') % tuple(self.events[i]))
        fout.close()

        fout = open(os.path.join(self.params.outdir, "eoccur_%s.tsv" % tag), 'w+')
        for i in range(len(self.eoccur)):
            fout.write("%f\n" % self.eoccur[i])
        fout.close()

    def fit(self):
        self.init()

        iteration = 0
        days_seen = np.zeros((self.data.day_count(),1))
        self.save('%04d' % iteration) #TODO: rm; this is just for visualization

        print "starting..."
        while not self.converged(iteration):
            iteration += 1

            lambda_a_events = np.zeros((self.data.day_count(), self.data.dimension))
            lambda_b_events = np.zeros((self.data.day_count(), self.data.dimension))
            lambda_a_entity = np.zeros((1, self.data.dimension))
            lambda_b_entity = np.zeros((1, self.data.dimension))
            lambda_eoccur = np.zeros(self.data.day_count())

            #for doc in self.data.docs:
            for d in range(self.params.batch_size):
                doc = self.data.random_doc()
                f_array_master = np.zeros((self.data.day_count(),1))
                for day in range(self.data.day_count()):
                    f_array_master[day] = self.params.f(self.data.days[day], doc.day)

                h_a_entity = np.zeros((self.params.num_samples, self.data.dimension))
                f_a_entity = np.zeros((self.params.num_samples, self.data.dimension))
                h_b_entity = np.zeros((self.params.num_samples, self.data.dimension))
                f_b_entity = np.zeros((self.params.num_samples, self.data.dimension))
                h_a_events = np.zeros((self.params.num_samples, \
                    self.data.day_count(), self.data.dimension))
                f_a_events = np.zeros((self.params.num_samples, \
                    self.data.day_count(), self.data.dimension))
                h_b_events = np.zeros((self.params.num_samples, \
                    self.data.day_count(), self.data.dimension))
                f_b_events = np.zeros((self.params.num_samples, \
                    self.data.day_count(), self.data.dimension))

                for s in range(self.params.num_samples):
                    # sample for each latent parameter
                    entity = np.random.gamma(M(self.a_entity) * M(self.b_entity), \
                        1.0 / M(self.b_entity), (1, self.data.dimension))
                    events = np.random.gamma(M(self.a_events) * M(self.b_events), \
                        1.0 / M(self.b_events), (self.data.day_count(), self.data.dimension))
                    eoccur = np.random.binomial(1, S(self.l_eoccur))

                    f_array = np.zeros((self.data.day_count(),1))
                    for day in range(self.data.day_count()):
                        f_array[day] = f_array_master[day] * eoccur[day]

                    # entity contributions to updates
                    p_entity = pGamma(entity, self.params.a_entity, self.params.b_entity)
                    q_entity, g_entity_a, g_entity_b = \
                        qgGamma(entity, self.a_entity, self.b_entity)

                    # event content contributions
                    p_events = pGamma(events, self.params.a_events, self.params.b_events)
                    q_events, g_events_a, g_events_b = \
                        qgGamma(events, self.a_events, self.b_events)

                    # event occurance contributions
                    p_eoccur = pBernoulli(eoccur, self.params.l_eoccur)
                    q_eoccur, g_eoccur = qgBernoulli(eoccur, self.l_eoccur)

                    # document contributions to updates
                    doc_params = entity + sum(f_array*events)
                    p_doc = pGamma(doc.rep, self.params.b_docs * doc_params, self.params.b_docs)
                    p_doc_eoccur = ((f_array != 0) * p_doc).sum()

                    h_a_entity[s] = g_entity_a
                    h_b_entity[s] = g_entity_b
                    f_a_entity[s] = g_entity_a * (p_entity + \
                        p_doc - q_entity)
                    f_b_entity[s] = g_entity_b * (p_entity + \
                        p_doc - q_entity)
                    lambda_a_entity += f_a_entity[s]
                    lambda_b_entity += f_b_entity[s]

                    h_a_events[s] = (f_array != 0) * g_events_a
                    h_b_events[s] = (f_array != 0) * g_events_b
                    f_a_events[s] = (f_array != 0) * g_events_a * \
                        (p_events + p_doc - q_events)
                    f_b_events[s] = (f_array != 0) * g_events_b * \
                        (p_events + p_doc - q_events)
                    lambda_a_events += f_a_events[s]
                    lambda_b_events += f_b_events[s]

                    f_array = f_array.flatten()
                    lambda_eoccur += (f_array != 0) * g_eoccur * \
                        (p_eoccur + p_doc_eoccur - q_eoccur)

                #print "H", sum(h_a_entity)
                #print "VAR", var(h_a_entity)
                #print "COV", cov(f_a_entity, h_a_entity)
                #print lambda_a_entity
                lambda_a_entity -= sum(h_a_entity) * cov(f_a_entity, h_a_entity) / var(h_a_entity)
                lambda_b_entity -= sum(h_b_entity) * cov(f_b_entity, h_b_entity) / var(h_b_entity)
                #print lambda_a_entity
                lambda_a_events -= sum(h_a_events) * cov(f_a_events, h_a_events) / var(h_a_events)
                lambda_b_events -= sum(h_b_events) * cov(f_b_events, h_b_events) / var(h_b_events)

            lambda_a_entity /= self.params.batch_size * self.params.num_samples
            lambda_b_entity /= self.params.batch_size * self.params.num_samples
            lambda_a_events /= self.params.batch_size * self.params.num_samples
            lambda_b_events /= self.params.batch_size * self.params.num_samples
            lambda_eoccur /= self.params.batch_size * self.params.num_samples

            rho = (iteration + self.params.tau) ** (-1.0 * self.params.kappa)
            self.a_entity += rho * lambda_a_entity
            self.b_entity += rho * lambda_b_entity

            #TESTING w/ fixed events
            self.a_events += rho * lambda_a_events
            self.b_events += rho * lambda_b_events #BNOB TODO put back
            self.l_eoccur += rho * lambda_eoccur

            min_thresh = 1e-10#sys.float_info.min**(1./4)
            max_thresh = 1e5

            self.entity = EGamma(self.a_entity, self.b_entity)
            self.events = EGamma(self.a_events, self.b_events)
            self.eoccur = S(self.l_eoccur)
            print "end of iteration"
            print "*************************************"

            if iteration % params.save_freq == 0:
                self.save('%04d' % iteration)

        # save final state
        self.save('final')



if __name__ == '__main__':
    ## Start by parsing the arguments
    import argparse

    # general script description
    parser = argparse.ArgumentParser(description = \
        'find events in a collection of documents.')

    parser.add_argument('content_filename', type=str, \
        help='a path to document content; lda-svi doc-topic output form (one doc per line, tab separated values)')
    parser.add_argument('time_filename', type=str, \
        help='a path to document times; one line per document with integer value')
    parser.add_argument('--out', dest='outdir', type=str, \
        default='out', help='output directory')

    parser.add_argument('--batch', dest='B', type=int, \
        default=1024, help = 'number of docs per batch, default 1024')
    parser.add_argument('--samples', dest='S', type=int, \
        default=64, help = 'number of approximating samples, default 64')
    parser.add_argument('--save_freq', dest='save_freq', type=int, \
        default=10, help = 'how often to save, default every 10 iterations')
    parser.add_argument('--convergence_thresh', dest='convergence_thresh', type=float, \
        default=1e-3, help = 'likelihood threshold for convergence, default 1e-3')
    parser.add_argument('--max_iter', dest='max_iter', type=int, \
        default=1000, help = 'maximum number of iterations, default 1000')
    parser.add_argument('--seed', dest='seed', type=int, \
        default=(dt.fromtimestamp(0) - dt.now()).microseconds, help = 'random seed, default from time')

    parser.add_argument('--tau', dest='tau', type=int, \
        default=1024, help = 'positive-valued learning parameter that downweights early iterations; default 1024')
    parser.add_argument('--kappa', dest='kappa', type=float, \
        default=0.7, help = 'learning rate: should be between (0.5, 1.0] to guarantee asymptotic convergence')

    parser.add_argument('--a_entities', dest='a_entities', type=float, \
        default=1.0, help = 'shape prior on entities; default 1')
    parser.add_argument('--b_entities', dest='b_entities', type=float, \
        default=1.0, help = 'rate prior on entities; default 1')
    parser.add_argument('--a_events', dest='a_events', type=float, \
        default=1.0, help = 'shape prior on events; default 1')
    parser.add_argument('--b_events', dest='b_events', type=float, \
        default=1.0, help = 'rate prior on events; default 1')
    parser.add_argument('--b_docs', dest='b_docs', type=float, \
        default=1.0, help = 'rate prior (and partial shape prior) on documents; default 1')
    parser.add_argument('--event_occur', dest='event_occurance', type=float, \
        default=0.5, help = 'prior to how often events should occur; range [0,1] and default 0.5')

    parser.add_argument('--event_dur', dest='event_duration', type=int, \
        default=7, help = 'the length of time an event can be relevant; default 7')

    # parse the arguments
    args = parser.parse_args()


    ## Other setup: input (data), output, parameters object
    # seed random number generator
    np.random.seed(args.seed)

    # read in data
    data = Corpus(args.content_filename, args.time_filename)

    # create output dir (check if exists)
    if os.path.exists(args.outdir):
        print "Output directory %s already exists.  Removing it to have a clean output directory!" % args.outdir
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    # create an object of model parameters
    #def __init__(self, outdir, batch_size, num_samples, save_freq, \
    #    conv_thresh, max_iter, tau, kappa, \
    #    a_ent, b_ent, a_evn, b_evn, b_doc, eoc, event_duration, \
    #    content, time):
    params = Parameters(args.outdir, args.B, args.S, args.save_freq, \
        args.convergence_thresh, args.max_iter, args.tau, args.kappa, \
        args.a_entities, args.b_entities, args.a_events, args.b_events, args.b_docs, args.event_occurance, \
        args.event_duration, args.content_filename, args.time_filename)
    params.save(args.seed)


    ## Fit model
    model = Model(data, params)
    model.fit()
