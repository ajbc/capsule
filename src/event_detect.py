import numpy as np
import shutil, os, sys
from datetime import datetime as dt
from scipy.special import gammaln, digamma
from scipy.misc import factorial
from collections import defaultdict
import subprocess, time
from multiprocessing import Process, Lock
import multiprocessing as mp

# suppress scientific notation when printing
np.set_printoptions(suppress=True)

import warnings #TODO rm
warnings.filterwarnings('error')

## helper functions

# softplus
def M(x):
    x[x > np.log(sys.float_info.max)] = np.log(sys.float_info.max)
    rv = np.log(1.0 + np.exp(x))
    rv[rv == 0] = (sys.float_info.min)**0.5
    return rv

# derivative of softplus
def dM(x):
    #x[x > np.log(sys.float_info.max)] = np.log(sys.float_info.max)
    #return np.exp(x) / (1.0 + np.exp(x))
    x[-x > np.log(sys.float_info.max)] = -np.log(sys.float_info.max)
    return 1.0 / (1.0 + np.exp(-x))

# inverse of softplus
def iM(x):
    return np.log(np.exp(x) - 1.0)

# sigmoid
def S(x):
    x[x < -np.log(sys.float_info.max)] = -np.log(sys.float_info.max)
    return 1.0 / (1.0 + np.exp(-x))

#derivative of sigmoid
def dS(x):
    return S(x) * (1-S(x))

# inverse of sigmoid
def iS(x):
    return -np.log(1.0/x - 1)


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
    if isinstance(rv, float):
        rv = sys.float_info.min if rv == 0 else rv
    else:
        rv[rv == 0] = sys.float_info.min
    return rv

def pGamma(x, a, b):
    if (x == 0).any() or (np.array(b) == 0).any():
        x[x==0] = sys.float_info.min
    return a*b*np.log(b) - lngamma(a*b) + (a*b-1.0)*np.log(x) - b*x

def qgGamma(x, a, b):
    dMb = dM(b)
    b = M(b)
    dMa = dM(a)
    a = M(a)

    g_a = dMa * b * (np.log(b) - digamma(a*b) + np.log(x))
    g_b = dMb * (a*(np.log(b)+1.0) - a * digamma(a*b) + a*np.log(x) - x)

    return (pGamma(x, a, b), g_a, g_b)

def EGamma(a, b):
    num = M(a) * M(b)
    den = M(b)

    return num / den

def pExponential(x, rate):
    return rate * np.exp(-rate*x)

def qgExponential(x, rate, b):
    return (pExponential(x, M(rate)), dM(rate) * (1.0 / M(rate) - x), np.zeros(b.shape))

def EExponential(rate):
    return 1.0 / M(rate)

def pLogNormal(x, loc, scale):
    if isinstance(loc, float):
        loc = loc if loc < np.log(sys.float_info.max)/2 else np.log(sys.float_info.max)/2
    else:
        loc[loc > np.log(sys.float_info.max)/2] = np.log(sys.float_info.max)/2
    x[x < np.sqrt(sys.float_info.min)] = np.sqrt(sys.float_info.min)
    x[x > np.sqrt(sys.float_info.max)] = np.sqrt(sys.float_info.max)
    return -np.log(x * scale * np.sqrt(2 * np.pi)) - \
        (np.log(x)-loc)**2 / (2 * scale**2)

def qgLogNormal(x, loc, scale):
    dMs = dM(scale)
    scale = M(scale)
    return (pLogNormal(x, loc, scale), \
        (scale**-2 * (np.log(x) - loc)), \
        dMs * (scale**-3 * (np.log(x) - loc)**2 - scale**-2))

def ELogNormal(loc, scale):
    #scale[scale > iM(np.sqrt(np.log(sys.float_info.max)))] = iM(np.sqrt(np.log(sys.float_info.max)))
    loc[loc > np.log(sys.float_info.max)/2] = np.log(sys.float_info.max)/2
    return np.exp(loc + M(scale)**2/2)

def pTopics(dist, x, a, b):
    if dist == "Gamma":
        return pGamma(x, a, b)
    elif dist == "LogNormal":
        return pLogNormal(x, a, b)
    elif dist == "Exponential":
        return pExponential(x, a)
    else:
        print "Invalid topic distribution.  Options: Gamma, LogNormal, Exponential.  Given:", dist
        sys.exit(-1)

def qgTopics(dist, x, a, b):
    if dist == "Gamma":
        return qgGamma(x, a, b)
    elif dist == "LogNormal":
        return qgLogNormal(x, a, b)
    elif dist == "Exponential":
        return qgExponential(x, a, b)
    else:
        print "Invalid topic distribution.  Options: Gamma, LogNormal, Exponential.  Given:", dist
        sys.exit(-1)

def ETopics(dist, a, b):
    if dist == "Gamma":
        return EGamma(a, b)
    elif dist == "LogNormal":
        return ELogNormal(a, b)
    elif dist == "Exponential":
        return EExponential(a)
    else:
        print "Invalid topic distribution.  Options: Gamma, LogNormal, Exponential.  Given:", dist
        sys.exit(-1)

def pBernoulli(x, p):
    return p**x * (1-p)**(1-x)

def qgBernoulli(x, p):
    return (pBernoulli(x, S(p)), dS(p)**x * (-dS(p))**(1-x))

def pPoisson(x, p):
    rv = x*np.log(p) - np.log(factorial(x)) - p
    rv[np.isinf(rv)] = -sys.float_info.max
    return rv

def qgPoisson(x, p):
    dMp = dM(p)
    p = M(p)
    return (pPoisson(x, p), dMp * (x/p - 1))

def cv_update(p, q, g, pr=False):
    f = g * (p - q)
    cv = cov(f, g) / var(g)
    if pr:
        return (f - cv *g)
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
        self.dated_docs = defaultdict(list)
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
            doc = Document(len(self.docs), times.pop(0), rep)
            self.docs.append(doc)
            self.dated_docs[doc.day].append(doc)

        self.validation = set()
        while len(self.validation) < 0.05 * len(self.docs):
            self.validation.add(self.random_doc())

    def day_count(self):
        return len(self.days)

    def num_docs(self):
        return len(self.docs)

    def num_docs_by_day(self, day):
        return len(self.dated_docs[day])

    def random_doc(self):
        return self.docs[np.random.randint(len(self.docs))]

    def random_doc_by_day(self, day):
        return self.dated_docs[day][np.random.randint(len(self.dated_docs[day]))]


class Parameters:
    def __init__(self, outdir, batch_size, num_samples, save_freq, \
        conv_thresh, min_iter, max_iter, tau, kappa, \
        a_ent, b_ent, a_evn, b_evn, b_doc, eoc, \
        event_duration, event_dist, topic_dist,\
        content, time):
        self.outdir = outdir
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.save_freq = save_freq
        self.convergence_thresh = conv_thresh
        self.min_iter = min_iter
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
        self.event_dist = event_dist
        self.topic_dist = topic_dist

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
        f.write("min # of iterations:\t%d\n" % self.min_iter)
        f.write("max # of iterations:\t%d\n" % self.max_iter)
        f.write("tau:\t%d\n" % self.tau)
        f.write("kappa:\t%f\n" % self.kappa)
        f.write("a_entity:\t%f\n" % self.a_entity)
        f.write("b_entity:\t%f\n" % self.b_entity)
        f.write("a_events:\t%f\n" % self.a_events)
        f.write("b_events:\t%f\n" % self.b_events)
        f.write("b_docs:\t%f\n" % self.b_docs)
        f.write("prior on event occurance:\t%f\n" % self.l_eoccur)
        f.write("event duration:\t%d\n" % self.d)
        f.write("event dist:\t%s\n" % self.event_dist)
        f.write("topic dist:\t%s\n" % self.topic_dist)
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
        self.l_eoccur = np.ones((self.data.day_count(), 1)) * \
            (iM(self.params.l_eoccur) if self.params.event_dist == "Poisson" else iS(self.params.l_eoccur))
        self.a_events = np.ones((self.data.day_count(), self.data.dimension)) * \
            iM(self.params.a_events)
        self.b_events = np.ones((self.data.day_count(), self.data.dimension)) * \
            iM(self.params.b_events)
        if self.params.topic_dist == "LogNormal":
            self.a_entity *= 0
            self.a_events *= 0

        # expected values of goal model parameters
        self.entity = ETopics(self.params.topic_dist, self.a_entity, self.b_entity)
        self.eoccur = (M(self.l_eoccur) if self.params.event_dist == "Poisson" else S(self.l_eoccur))
        self.events = ETopics(self.params.topic_dist, self.a_events, self.b_events)

        self.likelihood_decreasing_count = 0

    def compute_ELBO(self):
        log_priors = pTopics(self.params.topic_dist, self.entity, self.params.a_entity, self.params.b_entity).sum() + \
            pTopics(self.params.topic_dist, self.events, self.params.a_events, self.params.b_events).sum()
        log_q = pTopics(self.params.topic_dist, self.entity, M(self.a_entity), M(self.b_entity)).sum() + \
            pTopics(self.params.topic_dist, self.events, M(self.a_events), M(self.b_events)).sum()
        if self.params.event_dist == "Poisson":
            log_priors += pPoisson(self.eoccur, self.params.l_eoccur).sum()
            log_q += pPoisson(self.eoccur, M(self.l_eoccur)).sum()
        else:
            log_priors += pBernoulli(self.eoccur, self.params.l_eoccur).sum()
            log_q += pBernoulli(self.eoccur, S(self.l_eoccur)).sum()
        return self.compute_likelihood(False) + log_priors - log_q

    def compute_likelihood(self, valid=True):
        log_likelihood = 0
        LL = np.zeros(self.data.dimension)
        f_array = np.zeros((self.data.day_count(),1))
        ds = self.data.validation if valid else self.data.docs
        for doc in ds:
            for day in range(self.data.day_count()):
                f_array[day] = self.params.f(self.data.days[day], doc.day)
            doc_params = self.entity + (f_array*self.events*self.eoccur).sum(0)
            log_likelihood += np.sum(pGamma(doc.rep, doc_params, self.params.b_docs))
            LL += pGamma(doc.rep, doc_params, self.params.b_docs)
        #print "LL", LL
        return log_likelihood

    def converged(self, iteration):
        if iteration == 0:
            self.likelihood = -sys.float_info.max
            self.elbo = -sys.float_info.max
            flog = open(os.path.join(self.params.outdir, 'log.dat'), 'w+')
            flog.write("iteration\ttime\tlog.likelihood\tll.change\tELBO\tELBO.change\n")

        self.old_likelihood = self.likelihood
        self.likelihood = self.compute_likelihood()
        lldelta = (self.likelihood - self.old_likelihood) / \
            abs(self.old_likelihood)

        self.old_elbo = self.elbo
        self.elbo = self.compute_ELBO()
        elbodelta = (self.elbo - self.old_elbo) / \
            abs(self.old_elbo)

        flog = open(os.path.join(self.params.outdir, 'log.dat'), 'a')
        flog.write("%d\t%s\t%f\t%f\t%f\t%f\n" % (iteration, dt.now(), self.likelihood, lldelta, self.elbo, elbodelta))
        print "%d\t%s\t%f\t%f\t%f\t%f" % (iteration, dt.now(), self.likelihood, lldelta, self.elbo, elbodelta)

        if elbodelta < 0:
            print "ELBO decreasing (bad)"
            self.elbo_decreasing_count += 1
            if iteration > self.params.min_iter and self.elbo_decreasing_count >= 3:
                print "STOP: 3 consecutive iterations of increasing ELBO"
                return True
            return False
        else:
            self.elbo_decreasing_count = 0

        if iteration > self.params.min_iter and elbodelta < self.params.convergence_thresh:
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

    '''def doc_contributions(self, locks, date, entity, eoccur, events, incl):
        p_entity_lock, p_eoccur_lock, p_events_lock = locks
        doc_scale = 1.0
        docset = []
        print "\t\t day", date, "(%d docs)" % self.data.num_docs_by_day(date)
        if self.data.num_docs_by_day(date) < self.params.batch_size:
            docset = self.data.dated_docs[date]
        else:
            doc_scale = self.data.num_docs_by_day(date) * 1.0 / self.params.batch_size
            docset = [self.data.random_doc_by_day(date) for d in range(self.params.batch_size)]
        for doc in docset:
            f_array = np.zeros((self.data.day_count(),1))
            relevant_days = set()
            for day in range(self.data.day_count()):
                f_array[day] = self.params.f(self.data.days[day], doc.day)
                relevant_days.add(day)

            # document contributions to updates
            doc_params = entity + (f_array*events*eoccur).sum(1)
            p_doc = pGamma(doc.rep, doc_params, self.params.b_docs)

            p_entity_lock.acquire()
            self.p_entity += p_doc * doc_scale
            p_entity_lock.release()

            for i in relevant_days:
                p_eoccur_lock.acquire()
                self.p_eoccur[:,i,:] += np.transpose(p_doc.sum(1) * np.ones((1,1))) * doc_scale
                p_eoccur_lock.release()

                p_events_lock.acquire()
                self.p_events[:,i,:] += incl[:,i,:] * p_doc * doc_scale
                p_events_lock.release()
    '''

    def fit(self):
        self.init()

        iteration = 0
        days_seen = np.zeros((self.data.day_count(),1))

        self.save('%04d' % iteration) #TODO: rm; this is just for visualization

        #print "a-ent", self.a_entity
        #print "b-ent", self.b_entity
        #print "M(b-ent)", M(self.b_entity)
        #print "entity", self.entity
        #print "*************************************"

        print "starting..."
        while not self.converged(iteration):
            print "*************************************"
            iteration += 1
            print "iteration", iteration

            event_count = np.zeros((self.data.day_count(),self.data.dimension))

            print "sampling latent parameters"
            # sample latent parameters
            entity = np.random.gamma(M(self.a_entity) * M(self.b_entity), \
                1.0 / M(self.b_entity), (self.params.num_samples, self.data.dimension))
            if self.params.event_dist == "Poisson":
                eoccur = np.random.poisson(M(self.l_eoccur) * np.ones((self.params.num_samples, self.data.day_count(), 1)))
            else:
                eoccur = np.random.binomial(1, S(self.l_eoccur) * np.ones((self.params.num_samples, self.data.day_count(), 1)))
            events = np.random.gamma(M(self.a_events) * M(self.b_events), \
                1.0 / M(self.b_events), (self.params.num_samples, self.data.day_count(), self.data.dimension))

            print "computing p, q, and g for latent parameters"
            ## p, q, and g for latent parameters
            # entity topics
            self.p_entity = pTopics(self.params.topic_dist, entity, self.params.a_entity, self.params.b_entity)
            q_entity, g_entity_a, g_entity_b = \
                qgTopics(self.params.topic_dist, entity, self.a_entity, self.b_entity)
            #print "src a", M(self.a_entity)
            #print "src b", M(self.b_entity)

            # event occurance
            if self.params.event_dist == "Poisson":
                self.p_eoccur = pPoisson(eoccur, self.params.l_eoccur)
                q_eoccur, g_eoccur = qgPoisson(eoccur, self.l_eoccur)
            else:
                self.p_eoccur = pBernoulli(eoccur, self.params.l_eoccur)
                q_eoccur, g_eoccur = qgBernoulli(eoccur, self.l_eoccur)

            # event content
            self.p_events = pTopics(self.params.topic_dist, events, self.params.a_events, self.params.b_events)
            q_events, g_events_a, g_events_b = \
                qgTopics(self.params.topic_dist, events, self.a_events, self.b_events)

            #TODO: constrain event content based on occurance (e.g. probabilties above)
            incl = eoccur != 0

            doc_scale = self.data.num_docs() / self.params.batch_size
            for d in range(self.params.batch_size):
                doc = self.data.random_doc()
                f_array = np.zeros((self.data.day_count(),1))
                relevant_days = set()
                for day in range(self.data.day_count()):
                    f_array[day] = self.params.f(self.data.days[day], doc.day)
                    relevant_days.add(day)

                # document contributions to updates
                doc_params = entity + (f_array*events*eoccur).sum(1)
                p_doc = pGamma(doc.rep, doc_params, self.params.b_docs)
                p_entity += p_doc * doc_scale
                for i in relevant_days:
                    p_eoccur[:,i,:] += np.transpose(p_doc.sum(1) * np.ones((1,1))) * doc_scale
                    p_events[:,i,:] += incl[:,i,:] * p_doc * doc_scale
            '''print "\tgoing through each day"

            max_children = 20
            locks = (Lock(), Lock(), Lock())
            for date in self.data.days:
                print "current:", len(mp.active_children())
                while len(mp.active_children()) >= max_children:
                    time.sleep(2)

                p = Process(target=self.doc_contributions, args=(locks, date, entity, eoccur, events, incl))
                p.start()

            for p in mp.active_children():
                p.join()'''

            rho = (iteration + self.params.tau) ** (-1.0 * self.params.kappa)

            aup = cv_update(p_entity, q_entity, g_entity_a, True)
            self.a_entity += (rho/self.params.num_samples) * cv_update(p_entity, q_entity, g_entity_a)
            self.b_entity += (rho/self.params.num_samples) * cv_update(p_entity, q_entity, g_entity_b)

            self.l_eoccur += (rho/self.params.num_samples) * cv_update(self.p_eoccur, q_eoccur, g_eoccur)

            es = eoccur.sum(0) + sys.float_info.min
            self.a_events += (eoccur.sum(0) != 0) * (rho / es) * cv_update(p_events, q_events, g_events_a)
            self.b_events += (eoccur.sum(0) != 0) * (rho / es) * cv_update(p_events, q_events, g_events_b)

            self.entity = ETopics(self.params.topic_dist, self.a_entity, self.b_entity)
            print "*************************************"
            #print "a-ent", self.a_entity
            #print "b-ent", self.b_entity
            #print "M(b-ent)", M(self.b_entity)
            print "entity", self.entity
            if self.params.event_dist == "Poisson":
                self.eoccur = M(self.l_eoccur)
            else:
                self.eoccur = S(self.l_eoccur)
            print "events", self.eoccur.T
            self.events = ETopics(self.params.topic_dist, self.a_events, self.b_events)
            #print "end of iteration"
            print "*************************************"

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
        default=1024, help = 'number of approximating samples, default 1024')
    parser.add_argument('--save_freq', dest='save_freq', type=int, \
        default=10, help = 'how often to save, default every 10 iterations')
    parser.add_argument('--convergence_thresh', dest='convergence_thresh', type=float, \
        default=1e-3, help = 'likelihood threshold for convergence, default 1e-3')
    parser.add_argument('--min_iter', dest='min_iter', type=int, \
        default=30, help = 'minimum number of iterations, default 30')
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
        default=0.1, help = 'rate prior (and partial shape prior) on documents; default 0.1 (*sensitive*)')
    parser.add_argument('--event_occur', dest='event_occurance', type=float, \
        default=0.5, help = 'prior to how often events should occur; range [0,1] and default 0.5')

    parser.add_argument('--event_dur', dest='event_duration', type=int, \
        default=7, help = 'the length of time an event can be relevant; default 7')
    parser.add_argument('--event_dist', dest='event_dist', type=str, \
        default="Bernoulli", help = 'what distribution used to model event occurance: \"Poisson\" or \"Bernoulli\" (default)')
    parser.add_argument('--topic_dist', dest='topic_dist', type=str, \
        default="Gamma", help = 'what distribution used to model topics: \"Gamma\" (default) or \"LogNormal\" or \"Exponential\"')

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
        args.convergence_thresh, args.min_iter, args.max_iter, args.tau, args.kappa, \
        args.a_entities, args.b_entities, args.a_events, args.b_events, args.b_docs, args.event_occurance, \
        args.event_duration, args.event_dist, args.topic_dist, \
        args.content_filename, args.time_filename)
    params.save(args.seed, args.message)

    # read in data
    data = Corpus(args.content_filename, args.time_filename, params.f)

    ## Fit model
    model = Model(data, params)
    model.fit()
