import numpy as np
import shutil, os, sys
from datetime import datetime as dt
from scipy.special import gammaln, digamma
from scipy.misc import factorial
import subprocess

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

# inverse of softmax
def iM(x):
    return np.log(np.exp(x) - 1.0)


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

def compute_variance(p, itr):
    var = p.base_variance * np.exp(-itr * p.decay_rate)
    return max(var, sys.float_info.min)

def pGaussian(x, mu, var):
    return -1.0 * (np.log(np.sqrt(var * 2 * np.pi)) + (x-mu)**2 / (2*var))

def gGaussian(x, mu, var):
    return (1.0 / var) * (x - mu)

def pGamma(x, a, b):
    if (x == 0).any() or (np.array(b) == 0).any():
        x[x==0] = sys.float_info.min
    return a*b*np.log(b) - lngamma(a*b) + (a*b-1.0)*np.log(x) - b*x

def qgGamma(x, a1, b1, a0, b0, v):
    return (pGamma(x, M(a1), M(b1)) + pGaussian(a1, a0, v) + \
        pGaussian(b1, b0, v), gGaussian(a1, a0, v), gGaussian(b1, b0, v))

def EGamma(a, b):
    num = M(a) * M(b)
    den = M(b)

    return num / den

def pPoisson(x, p):
    rv = x*np.log(p) - np.log(factorial(x)) - p
    rv[np.isinf(rv)] = -sys.float_info.max
    return rv

def qgPoisson(x, p1, p0, v):
    return (pPoisson(x, M(p1)) + pGaussian(p1, p0, v), gGaussian(p1, p0, v))

def cv_update(p, q, g):
    #f = g * (p - q - 1) #TODO: try with and without -1 (shouldnt matter...)
    f = g * (p - q)
    cv = cov(f, g) / var(g)
    return (f - cv * g).sum(0)


## Classes

class Document:
    def __init__(self, id, day, sparse_rep):
        self.id = id
        self.day = day
        self.rep = sparse_rep
        self.rep[self.rep == 0] = sys.float_info.min


class Corpus:
    def __init__(self, content_filename, time_filename, date_function):
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
        conv_thresh, max_iter, tau, kappa, base_var, decay_rate, \
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

        self.base_variance = base_var
        self.decay_rate = decay_rate

        self.a_entity = a_ent
        self.b_entity = b_ent
        self.a_events = a_evn
        self.b_events = b_evn
        self.b_docs = b_doc
        self.l_eoccur = eoc

        self.d = event_duration

        self.content = content
        self.time = time

    def save(self, seed, message):
        f = open(os.path.join(self.outdir, 'settings.dat'), 'w+')

        f.write("%s\n" % dt.now())
        f.write("%s\n\n" % message)

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

        p = subprocess.Popen(['git','rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f.write("\ncommit #%s" % out)

        p = subprocess.Popen(['git','diff', 'event_detect.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f.write("%s" % out)

        f.close()

    def f(self, a, c):
        if a > c or c >= (a+self.d):
            return 0
        return (1 - ((0.0+c-a)/self.d))

    def fdays(a):
        return range(a, a + self.d)


class Model:
    def __init__(self, data, params):
        self.data = data
        self.params = params

    def init(self):
        # free variational parameters
        self.a_entity = np.ones(self.data.dimension) * iM(self.params.a_entity)
        self.b_entity = np.ones(self.data.dimension) * iM(self.params.b_entity)
        self.l_eoccur = np.ones((self.data.day_count(), 1)) * iM(self.params.l_eoccur)
        self.a_events = np.ones((self.data.day_count(), self.data.dimension)) * \
            iM(self.params.a_events)
        self.b_events = np.ones((self.data.day_count(), self.data.dimension)) * \
            iM(self.params.b_events)

        # expected values of goal model parameters
        self.entity = EGamma(self.a_entity, self.b_entity)
        self.eoccur = M(self.l_eoccur)
        self.events = EGamma(self.a_events, self.b_events)

        self.likelihood_decreasing_count = 0

    def compute_likelihood(self):
        log_likelihood = 0
        LL = np.zeros(self.data.dimension)
        f_array = np.zeros((self.data.day_count(),1))
        for doc in self.data.validation:
            for day in range(self.data.day_count()):
                f_array[day] = self.params.f(self.data.days[day], doc.day)
            doc_params = self.entity + (f_array*self.events*self.eoccur).sum(0)
            log_likelihood += np.sum(pGamma(doc.rep, doc_params, self.params.b_docs))
            LL += np.sum(pGamma(doc.rep, doc_params, self.params.b_docs), 0)
        #print "LL", LL
        return log_likelihood

    def converged(self, iteration):
        if iteration == 0:
            self.likelihood = -sys.float_info.max
            flog = open(os.path.join(self.params.outdir, 'log.dat'), 'w+')
            flog.write("iteration\ttime\tlikelihood\tchange\n")
            return False

        self.old_likelihood = self.likelihood
        self.likelihood = self.compute_likelihood()
        delta = (self.likelihood - self.old_likelihood) / \
            abs(self.old_likelihood)

        flog = open(os.path.join(self.params.outdir, 'log.dat'), 'a')
        flog.write("%d\t%s\t%f\t%f\n" % (iteration, dt.now(), self.likelihood, delta))
        print "%d\t%s\t%f\t%f" % (iteration, dt.now(), self.likelihood, delta)

        if delta < 0:
            print "likelihood decreasing (bad)"
            self.likelihood_decreasing_count += 1
            if iteration > 50 and self.likelihood_decreasing_count >= 3:
                print "STOP: 3 consecutive iterations of increasing likelihood"
                return True
            return False
        else:
            self.likelihood_decreasing_count = 0

        if iteration > 20 and delta < self.params.convergence_thresh:
            print "STOP: model converged!"
            return True
        if iteration == self.params.max_iter:
            print "STOP: iteration cap reached"
            return True
        return False

    def save(self, tag):
        fout = open(os.path.join(self.params.outdir, "entities_%s.tsv" % tag), 'w+')
        fout.write(('\t'.join(["%f"]*len(self.entity))+'\n') % tuple(self.entity))
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
            variance = compute_variance(self.params, iteration)

            event_count = np.zeros((self.data.day_count(),self.data.dimension))

            #print "****----****"
            #print "a", self.a_entity
            #print "b", self.b_entity
            #print "M(a)", M(self.a_entity)
            #print "M(b)", M(self.b_entity)
            #print "****----****"

            # sample latent parameters
            entity_a = np.random.normal(loc=self.a_entity, scale=variance, \
                size=(self.params.num_samples, self.data.dimension))
            entity_b = np.random.normal(loc=self.b_entity, scale=variance, \
                size=(self.params.num_samples, self.data.dimension))
            entity = np.random.gamma(M(entity_a) * M(entity_b), \
                1.0 / M(entity_b))
            #print "---"
            #print "sampled a[0]", entity_a[0]
            #print "sampled a[1]", entity_a[1]
            #print "sampled a[2]", entity_a[2]
            #print "M(a[0])", M(entity_a[0])
            #print "M(a[1])", M(entity_a[1])
            #print "M(a[2])", M(entity_a[2])
            #print "sampled b[0]", entity_b[0]
            #print "sampled b[1]", entity_b[1]
            #print "sampled b[2]", entity_b[2]
            #print "M(b[0])", M(entity_b[0])
            #print "M(b[1])", M(entity_b[1])
            #print "M(b[2])", M(entity_b[2])
            #print "sampled e[0]", entity[0]
            #print "sampled e[1]", entity[1]
            #print "sampled e[2]", entity[2]
            #print "---"

            l_eoccur = np.random.normal(loc=self.l_eoccur * \
                np.ones((self.params.num_samples, self.data.day_count(), 1)),
                scale=variance)
            eoccur = np.random.poisson(M(l_eoccur))

            events_a = np.random.normal(loc=self.a_events, scale=variance, \
                size=(self.params.num_samples, self.data.day_count(), \
                self.data.dimension))
            events_b = np.random.normal(loc=self.b_events, scale=variance, \
                size=(self.params.num_samples, self.data.day_count(), \
                self.data.dimension))
            events = np.random.gamma(M(events_a) * M(events_b), \
                1.0 / M(events_b))

            ## p, q, and g for latent parameters
            # entity topics
            p_entity = pGamma(entity, self.params.a_entity, self.params.b_entity)
            q_entity, g_entity_a, g_entity_b = \
                qgGamma(entity, entity_a, entity_b, \
                self.a_entity, self.b_entity, variance)

            # event occurance
            p_eoccur = pPoisson(eoccur, self.params.l_eoccur)
            q_eoccur, g_eoccur = qgPoisson(eoccur, l_eoccur, \
                self.l_eoccur, variance)

            # event content
            p_events = pGamma(events, self.params.a_events, self.params.b_events)
            q_events, g_events_a, g_events_b = \
                qgGamma(events, events_a, events_b, \
                self.a_events, self.b_events, variance)

            #TODO: constrain event content based on occurance (e.g. probabilties above)
            incl = eoccur != 0

            for doc in self.data.docs:
                f_array = np.zeros((self.data.day_count(),1))
                relevant_days = set()
                for day in range(self.data.day_count()):
                    f_array[day] = self.params.f(self.data.days[day], doc.day)
                    relevant_days.add(day)

                # document contributions to updates
                doc_params = entity + (f_array*events*eoccur).sum(1)
                p_doc = pGamma(doc.rep, doc_params, self.params.b_docs)
                p_entity += p_doc
                for i in relevant_days:
                    p_eoccur[:,i,:] += np.transpose(p_doc.sum(1) * np.ones((1,1)))
                    p_events[:,i,:] += incl[:,i,:] * p_doc
                    event_count[i] += sum(incl[:,i,:])

            rho = (iteration + self.params.tau) ** (-1.0 * self.params.kappa)
            #print rho

            self.a_entity += (rho/self.params.num_samples) * cv_update(p_entity, q_entity, g_entity_a)
            self.b_entity += (rho/self.params.num_samples) * cv_update(p_entity, q_entity, g_entity_b)

            #self.l_eoccur += (rho/self.params.num_samples) * cv_update(p_eoccur, q_eoccur, g_eoccur)

            #incl = event_count != 0
            #event_count[event_count == 0] = 1
            #self.a_events += (incl * rho / event_count) * cv_update(p_events, q_events, g_events_a)
            #self.b_events += (incl * rho / event_count) * cv_update(p_events, q_events, g_events_b)

            self.entity = EGamma(self.a_entity, self.b_entity)
            self.eoccur = M(self.l_eoccur)
            self.events = EGamma(self.a_events, self.b_events)
            #print "end of iteration"
            #print "*************************************"

            if iteration % params.save_freq == 0:
                self.save('%04d' % iteration)

            #if iteration == 3:
            #    break

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
    parser.add_argument('--msg', dest='message', type=str, \
        default='', help='log message')

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

    parser.add_argument('--base_var', dest='base_var', type=float, \
        default=10, help = 'base variance for first level sampling (Gaussian); default 10')
    parser.add_argument('--decay_rate', dest='decay_rate', type=float, \
        default=0.01, help = 'decay rate of variance for first level of sampling; default 0.01')

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

    # create output dir (check if exists)
    if os.path.exists(args.outdir):
        print "Output directory %s already exists.  Removing it to have a clean output directory!" % args.outdir
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    # create an object of model parameters
    params = Parameters(args.outdir, args.B, args.S, args.save_freq, \
        args.convergence_thresh, args.max_iter, args.tau, args.kappa, \
        args.base_var, args.decay_rate, \
        args.a_entities, args.b_entities, args.a_events, args.b_events, args.b_docs, args.event_occurance, \
        args.event_duration, args.content_filename, args.time_filename)
    params.save(args.seed, args.message)

    # read in data
    data = Corpus(args.content_filename, args.time_filename, params.f)

    ## Fit model
    model = Model(data, params)
    model.fit()
