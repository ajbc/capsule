import numpy as np
import shutil, os, sys
from datetime import datetime as dt
from scipy.special import gammaln, digamma
from scipy.misc import factorial
from collections import defaultdict
import subprocess, time
import cProfile, pstats, StringIO

# suppress scientific notation when printing
np.set_printoptions(suppress=True, linewidth=100)

import warnings #TODO rm
warnings.filterwarnings('error')

## helper functions
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

# soft-plus
def SP(x):
    return np.log(1. + np.exp(x))

# derivative of soft-plus
def dSP(x):
    return np.exp(x) / (1. + np.exp(x))

# inverse of soft-plus
def iSP(x):
    return np.log(np.exp(x) - 1.)

# log probability of a gamma given sparsity a and mean m
def Gammac(x, a, m):
    return a * (np.log(a) - np.log(m) - x/m) + (a-1.) * np.log(x)
def Gamma(x, a, m):
    return a * (np.log(a) - np.log(m) - x/m) - gammaln(a) + (a-1.) * np.log(x)

# derivative of above log gamma with respect to sparsity a
def dGamma_alpha(x, a, m):
    return np.log(a) + 1. - np.log(m) - digamma(a) + np.log(x) - (x / m)

# derivative of above log gamma with respect to mean m
def dGamma_mu(x, a, m):
    #return - (a / m) + ((a * x) / m**2)
    return (a / m) * (x / m - 1.)

# log probabilty of a Normal distribution with unit variance and mean m
def Normal(x, m):
    return - 0.5 * (x - m)**2 - np.log(np.sqrt(2 * np.pi))

# covariance where each column is draws for a variable
# (returns a row of covariances)
def cv(a, b):
    # subample rows
    r = np.random.choice(a.shape[0], a.shape[0]/10, replace=False)
    a = a[r]
    b = b[r]
    # compute true cv function
    aa = (a - sum(a)/a.shape[0])
    ab = a*b
    ab = (ab - sum(ab)/ab.shape[0])
    cov = (aa * ab)
    var = (aa * aa)
    cov = sum(cov) / cov.shape[0]
    var = sum(var) / var.shape[0]
    var[var < 1e-300] = 1e-300
    return cov / var

def draw_gamma(a, m, shape):
    rv = np.random.gamma(SP(a), SP(m)/SP(a), shape)
    rv[rv < 1e-300] = 1e-300
    return rv

def draw_masked_gamma(a, m, shape, rows):
    events = np.ones(shape) * 1e-300
    for row in range(len(rows)):
        if rows[row] != 0:
            events[row] = draw_gamma(a[row], b[row], (shape[1], shape[2]))
    return events

def pBernoulli(x, p):
    return p**x * (1-p)**(1-x)

def qgBernoulli(x, p):
    return (pBernoulli(x, S(p)), dS(p)**x * (-dS(p))**(1-x))

def pPoisson(x, p):
    rv = x*np.log(p) - np.log(factorial(x)) - p
    rv[np.isinf(rv)] = -sys.float_info.max
    return rv

def qgPoisson(x, p):
    dMp = dSP(p)
    p = SP(p)
    return (pPoisson(x, p), dMp * (x/p - 1))


## Classes

class Document:
    def __init__(self, id, sender, receiver, day, sparse_rep):
        self.id = id
        self.sender = sender# - 1
        self.receiver = receiver #- 1
        self.day = day
        self.rep = sparse_rep
        self.rep[self.rep < 1e-6] = 1e-6


class Corpus:
    def __init__(self, content_filename, meta_filename, date_function):
        self.docs = []
        self.dated_docs = defaultdict(list)
        metadata = [tuple([int(tt) for tt in t.strip().split('\t')]) \
            for t in open(meta_filename).readlines()]
        self.days = sorted(set([t[2] for t in metadata]))
        self.senders = sorted(set([t[0] for t in metadata]))
        self.dated_doc_count = defaultdict(int)
        self.sender_doc_count = defaultdict(int)
        self.date_sum = defaultdict(int)
        self.sender_sum = defaultdict(int)
        self.dimension = 0
        for line in open(content_filename):
            rep = np.array([float(v) for v in line.strip().split('\t')])
            if self.dimension == 0:
                self.dimension = len(rep)
            elif self.dimension != len(rep):
                print "Data malformed; document representations not of equal length"
                sys.exit(-1)
            meta = metadata.pop(0) # sender, receiver, date, all as ints
            doc = Document(len(self.docs), meta[0], meta[1], meta[2], rep)
            self.docs.append(doc)
            self.dated_doc_count[doc.day] += 1
            self.sender_doc_count[doc.sender] += 1
            self.date_sum[doc.day] += doc.rep
            self.sender_sum[doc.sender] += doc.rep

        self.validation = set()
        while len(self.validation) < 1000:
            self.validation.add(self.random_doc())

    def day_count(self):
        return len(self.days)

    def entity_count(self):
        return len(self.senders)

    def num_docs(self):
        return len(self.docs)

    def num_docs_by_date(self, date):
        return self.dated_doc_count[date]

    def ave_day(self, day):
        if self.dated_doc_count[day] == 0:
            return np.ones(self.dimension) * 0.001
        return self.date_sum[day] / self.dated_doc_count[day]

    def num_docs_by_sender(self, sender):
        return self.sender_doc_count[sender]

    def ave_entity(self, entity):
        if self.sender_doc_count[entity] == 0:
            return np.ones(self.dimension) * 0.001
        return self.sender_sum[entity] / self.sender_doc_count[entity]

    def random_doc(self):
        return self.docs[np.random.randint(len(self.docs))]


class Parameters:
    def __init__(self, outdir, batch_size, num_samples, save_freq, \
        conv_thresh, min_iter, max_iter, tau, kappa, \
        a_ent, m_ent, a_evn, m_evn, a_dsp, m_dsp, eoc, \
        event_duration, event_dist,\
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
        self.m_entity = m_ent
        self.a_events = a_evn
        self.m_events = m_evn
        #self.a_docspar = a_dsp
        #self.m_docspar = m_dsp
        self.l_eoccur = eoc

        self.d = event_duration
        self.event_dist = event_dist

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
        f.write("m_entity:\t%f\n" % self.m_entity)
        f.write("a_events:\t%f\n" % self.a_events)
        f.write("m_events:\t%f\n" % self.m_events)
        #f.write("a_docspar:\t%f\n" % self.a_docspar)
        #f.write("m_docspar:\t%f\n" % self.m_docspar)
        f.write("prior on event occurance:\t%f\n" % self.l_eoccur)
        f.write("event duration:\t%d\n" % self.d)
        f.write("event dist:\t%s\n" % self.event_dist)
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
        self.a_entity = np.ones((self.data.entity_count(), self.data.dimension)) * 10.0
        self.m_entity = np.ones((self.data.entity_count(), self.data.dimension)) * 0.1
        #self.m_entity = np.zeros((self.data.entity_count(), self.data.dimension))
        #for entity in range(self.data.entity_count()):
        #    self.m_entity[entity] = iSP(self.data.ave_entity(entity))
        #self.a_docspar = np.ones((self.data.entity_count(), self.data.dimension)) * 10.0
        #self.m_docspar = np.ones((self.data.entity_count(), self.data.dimension)) * iSP(0.1)#-1.0
        self.l_eoccur = np.ones((self.data.day_count(), 1)) * \
            (iSP(self.params.l_eoccur) if self.params.event_dist == "Poisson" else iS(self.params.l_eoccur))
        self.a_events = np.ones((self.data.day_count(), self.data.dimension)) * 0.1
        self.m_events = np.ones((self.data.day_count(), self.data.dimension)) * 0.01
        #self.m_events = np.zeros((self.data.day_count(), self.data.dimension))
        #for day in range(self.data.day_count()):
        #    self.m_events[day] = iSP(self.data.ave_day(day))

        # expected values of goal model parameters
        self.entity = SP(self.m_entity)
        #self.docspar = SP(self.m_docspar)
        self.eoccur = (SP(self.l_eoccur) if self.params.event_dist == "Poisson" else S(self.l_eoccur))
        self.events = SP(self.m_events)

        self.likelihood_decreasing_count = 0

    def compute_ELBO(self):
        pent = Gamma(self.entity, self.params.a_entity, self.params.m_entity).sum()
        pdsp = 0#Gamma(self.docspar, self.params.a_docspar, self.params.m_docspar).sum()
        pevt = Gamma(self.events, self.params.a_events, self.params.m_events).sum()
        log_priors = pent + pevt + pdsp

        qent = Gamma(self.entity, SP(self.a_entity), SP(self.m_entity)).sum()
        qdsp = 0#Gamma(self.docspar, SP(self.a_docspar), SP(self.m_docspar)).sum()
        qevt = Gamma(self.events, SP(self.a_events), SP(self.m_events)).sum()
        log_q = qent + qevt + qdsp

        if self.params.event_dist == "Poisson":
            peoc = pPoisson(self.eoccur, self.params.l_eoccur).sum()
            log_priors += pPoisson(self.eoccur, self.params.l_eoccur).sum()
            qeoc = pPoisson(self.eoccur, SP(self.l_eoccur)).sum()
            log_q += pPoisson(self.eoccur, SP(self.l_eoccur)).sum()
        else:
            peoc = pBernoulli(self.eoccur, self.params.l_eoccur).sum()
            log_priors += pBernoulli(self.eoccur, self.params.l_eoccur).sum()
            qeoc = pBernoulli(self.eoccur, S(self.l_eoccur)).sum()
            log_q += pBernoulli(self.eoccur, S(self.l_eoccur)).sum()
        ll = self.compute_likelihood(True)
        #return ll, pent, qent, pdsp, qdsp, pevt, qevt, peoc, qeoc, ll+log_priors - log_q
        return ll, pent, qent, pevt, qevt, peoc, qeoc, ll+log_priors - log_q

    def compute_likelihood(self, valid=True):
        log_likelihood = 0
        f_array = np.zeros((self.data.day_count(),1))
        d = 0
        ds = self.data.validation if valid else self.data.docs
        mult = len(self.data.docs) / len(self.data.validation) if valid else 1.0
        for doc in ds:
            #TODO: group by day for more efficient computation?
            for day in range(self.data.day_count()):
                f_array[day] = self.params.f(self.data.days[day], doc.day)
            doc_params = self.entity[doc.sender] + (f_array*self.events*self.eoccur).sum(0)
            #print "doc %d [day %d]" % (d, doc.day)
            d += 1
            #print "\trep:", doc.rep
            #print "\tpar:", doc_params
            #print '\t LL:', Gamma(doc.rep, self.params.a_docs, doc_params)
            #log_likelihood += np.sum(Gamma(doc.rep, self.docspar[doc.sender], doc_params))
            log_likelihood += np.sum(Gamma(doc.rep, 0.1, doc_params))
        return log_likelihood * mult

    def converged(self, iteration):
        #only check every 10 iterations
        if iteration % 10 != 0:
            return False

        if iteration == 0:
            self.likelihood = -sys.float_info.max
            self.elbo = -sys.float_info.max
            flog = open(os.path.join(self.params.outdir, 'log.dat'), 'w+')
            flog.write("iteration\ttime\tlog.likelihood\tll.change\tELBO\tELBO.change\n")
            flogE = open(os.path.join(self.params.outdir, 'log.ELBO.dat'), 'w+')
            flogE.write("iteration\ttime\tlog.likelihood\tlog.p.entity\tlog.q.entity\tlog.p.events\tlog.q.events\tlog.p.eoccur\tlog.q.eoccur\tELBO.approx\n")

        self.old_likelihood = self.likelihood
        self.likelihood = self.compute_likelihood()
        lldelta = (self.likelihood - self.old_likelihood) / \
            abs(self.old_likelihood)

        self.old_elbo = self.elbo
        #self.elbo = self.compute_ELBO()
        #ll, pent, qent, pdsp, qdsp, pevt, qevt, peoc, qeoc, self.elbo = self.compute_ELBO()
        ll, pent, qent, pevt, qevt, peoc, qeoc, self.elbo = self.compute_ELBO()
        elbodelta = (self.elbo - self.old_elbo) / \
            abs(self.old_elbo)

        flog = open(os.path.join(self.params.outdir, 'log.dat'), 'a')
        flog.write("%d\t%s\t%f\t%f\t%f\t%f\n" % (iteration, dt.now(), self.likelihood, lldelta, self.elbo, elbodelta))
        print "%d\t%s\t%f\t%f\t%f\t%f" % (iteration, dt.now(), self.likelihood, lldelta, self.elbo, elbodelta)
        flogE = open(os.path.join(self.params.outdir, 'log.ELBO.dat'), 'a')
        #flogE.write("%d\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (iteration, dt.now(), ll, pent, qent, pdsp, qdsp, pevt, qevt, peoc, qeoc, self.elbo))
        flogE.write("%d\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (iteration, dt.now(), ll, pent, qent, pevt, qevt, peoc, qeoc, self.elbo))

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
        for i in range(len(self.entity)):
            fout.write(('\t'.join(["%f"]*len(self.entity[i]))+'\n') % tuple(self.entity[i]))
        fout.close()
        '''
        fout = open(os.path.join(self.params.outdir, "docspar_%s.tsv" % tag), 'w+')
        for i in range(len(self.docspar)):
            fout.write(('\t'.join(["%f"]*len(self.docspar[i]))+'\n') % tuple(self.docspar[i]))
        fout.close()
        '''
        fout = open(os.path.join(self.params.outdir, "events_%s.tsv" % tag), 'w+')
        for i in range(len(self.events)):
            fout.write(('\t'.join(["%f"]*len(self.events[i])) +'\n') % tuple(self.events[i]))
        fout.close()

        fout = open(os.path.join(self.params.outdir, "eoccur_%s.tsv" % tag), 'w+')
        for i in range(len(self.eoccur)):
            fout.write("%f\n" % self.eoccur[i])
        fout.close()

    #def doc_contributions(self, p_entity, p_docspar, p_eoccur, p_events, entity, docspar, eoccur, events, incl):
    def doc_contributions(self, p_entity, p_eoccur, p_events, entity, eoccur, events, incl):
        docset = []
        if self.data.num_docs() < self.params.batch_size:
            docset = self.data.docs
            scale = False
            entity_scale = 1.0
            event_scale = 1.0
            print len(docset), "A"
        else:
            docset = [self.data.random_doc() for d in range(self.params.batch_size)]
            scale = True
            print len(docset), "B"
        days = {}
        dc  = 0
        for doc in docset:
            #print dc
            dc +=1
            if doc.day not in days:
                f_array = np.zeros((self.data.day_count(),1))
                relevant_days = set()
                for day in range(self.data.day_count()):
                    f_array[day] = self.params.f(self.data.days[day], doc.day)
                    relevant_days.add(day)
                days[doc.day] = (f_array * events * eoccur).sum(1)

            # document contributions to updates
            doc_params = entity[:,doc.sender,:] + days[doc.day]

            #p_doc = Gamma(doc.rep, docspar[:,doc.sender,:], doc_params)
            p_doc = Gammac(doc.rep, 0.1, doc_params)

            if scale:
                entity_scale = self.data.num_docs_by_sender(doc.sender) * 1.0 / self.params.batch_size
                event_scale = self.data.num_docs_by_date(doc.day) * 1.0 / self.params.batch_size

            p_entity[:,doc.sender,:] += p_doc * entity_scale
            #p_docspar[:,doc.sender,:] += p_doc * entity_scale

            for i in relevant_days:
                p_eoccur[:,i,:] += np.transpose(p_doc.sum(1) * np.ones((1,1))) * event_scale

                p_events[:,i,:] += incl[:,i,:] * p_doc * event_scale

    def fit(self):
        self.init()

        iteration = 0
        days_seen = np.zeros((self.data.day_count(),1))

        self.save('%04d' % iteration) #TODO: rm; this is just for visualization

        MS_a_entity = np.zeros(self.data.dimension)
        MS_m_entity = np.zeros(self.data.dimension)
        MS_eoccur = np.zeros(self.data.day_count())
        MS_a_events = np.zeros((self.data.day_count(), self.data.dimension))
        MS_m_events = np.zeros((self.data.day_count(), self.data.dimension))

        print "starting..."
        while not self.converged(iteration):
            print "*************************************"
            iteration += 1
            print "iteration", iteration

            event_count = np.zeros((self.data.day_count(),self.data.dimension))

            print "sampling latent parameters"
            # sample latent parameters
            entity = draw_gamma(self.a_entity, self.m_entity, (self.params.num_samples, self.data.entity_count(), self.data.dimension))
            #docspar = draw_gamma(self.a_docspar, self.m_docspar, (self.params.num_samples, self.data.entity_count(), self.data.dimension))
            if self.params.event_dist == "Poisson":
                eoccur = np.random.poisson(SP(self.l_eoccur) * np.ones((self.params.num_samples, self.data.day_count(), 1)))
            else:
                eoccur = np.random.binomial(1, S(self.l_eoccur) * np.ones((self.params.num_samples, self.data.day_count(), 1)))
            if iteration < 200:
                eoccur = np.zeros((self.params.num_samples, self.data.day_count(), 1))
            #elif iteration < 500:
            #    eoccur = np.ones((self.params.num_samples, self.data.day_count(), 1))
            events = draw_masked_gamma(self.a_events, self.m_events, (self.params.num_samples, self.data.day_count(), self.data.dimension), mask=eoccur)

            #eoccur = np.zeros((S(self.l_eoccur) * np.ones((self.params.num_samples, self.data.day_count(), 1))).shape)
            #docspar = 0.1 * np.ones((self.params.num_samples, self.data.entity_count(), self.data.dimension))

            print "computing p, q, and g for latent parameters"
            ## p, q, and g for latent parameters
            # entity topics
            p_entity = Gamma(entity, self.params.a_entity, self.params.m_entity)
            q_entity = Gamma(entity, SP(self.a_entity), SP(self.m_entity))
            g_entity_a = dSP(self.a_entity) * dGamma_alpha(entity, SP(self.a_entity), SP(self.m_entity))
            g_entity_m = dSP(self.m_entity) * dGamma_mu(entity, SP(self.a_entity), SP(self.m_entity))

            # document sparsity
            '''p_docspar = Gamma(docspar, self.params.a_docspar, self.params.m_docspar)
            q_docspar = Gamma(docspar, SP(self.a_docspar), SP(self.m_docspar))
            g_docspar_a = dSP(self.a_docspar) * dGamma_alpha(docspar, SP(self.a_docspar), SP(self.m_docspar))
            g_docspar_m = dSP(self.m_docspar) * dGamma_mu(docspar, SP(self.a_docspar), SP(self.m_docspar))'''

            # event occurance
            if self.params.event_dist == "Poisson":
                p_eoccur = pPoisson(eoccur, self.params.l_eoccur)
                q_eoccur, g_eoccur = qgPoisson(eoccur, self.l_eoccur)
            else:
                p_eoccur = pBernoulli(eoccur, self.params.l_eoccur)
                q_eoccur, g_eoccur = qgBernoulli(eoccur, self.l_eoccur)

            # event content
            p_events = Gamma(events, self.params.a_events, self.params.m_events)
            q_events = Gamma(events, SP(self.a_events), SP(self.m_events))
            g_events_a = dSP(self.a_events) * dGamma_alpha(events, SP(self.a_events), SP(self.m_events))
            g_events_m = dSP(self.m_events) * dGamma_mu(events, SP(self.a_events), SP(self.m_events))

            #TODO: constrain event content based on occurance (e.g. probabilties above)
            incl = eoccur != 0

            self.doc_contributions(p_entity, p_eoccur, p_events, entity, eoccur, events, incl)
            #self.doc_contributions(p_entity, p_docspar, p_eoccur, p_events, entity, docspar, eoccur, events, incl)

            print "cv"
            # control variates to decrease variance of gradient; one for each variational parameter
            cv_a_entity = cv(g_entity_a, p_entity - q_entity)
            cv_m_entity = cv(g_entity_m, p_entity - q_entity)
            #cv_a_docspar = cv(g_docspar_a, p_docspar - q_docspar)
            #cv_m_docspar = cv(g_docspar_m, p_docspar - q_docspar)
            cv_eoccur = cv(g_eoccur, p_eoccur - q_eoccur)
            cv_a_events = cv(g_events_a, p_events - q_events)
            cv_m_events = cv(g_events_m, p_events - q_events)

            print "RMSprop"
            # RMSprop: keep running average of gradient magnitudes
            # (the gradient will be divided by sqrt of this later)
            if MS_a_entity.all() == 0:
                MS_a_entity = (g_entity_a**2).sum(0)
                MS_m_entity = (g_entity_m**2).sum(0)
                #MS_a_docspar = (g_docspar_a**2).sum(0)
                #MS_m_docspar = (g_docspar_m**2).sum(0)
                MS_eoccur = (g_eoccur**2).sum(0)
                MS_a_events = (g_events_a**2).sum(0)
                MS_m_events = (g_events_m**2).sum(0)
            else:
                MS_a_entity = 0.9 * MS_a_entity + 0.1 * (g_entity_a**2).sum(0)
                MS_m_entity = 0.9 * MS_m_entity + 0.1 * (g_entity_m**2).sum(0)
                #MS_a_docspar = 0.9 * MS_a_docspar + 0.1 * (g_docspar_a**2).sum(0)
                #MS_m_docspar = 0.9 * MS_m_docspar + 0.1 * (g_docspar_m**2).sum(0)
                MS_eoccur = 0.9 * MS_eoccur + 0.1 * (g_eoccur**2).sum(0)
                MS_a_events = 0.9 * MS_a_events + 0.1 * (g_events_a**2).sum(0)
                MS_m_events = 0.9 * MS_m_events + 0.1 * (g_events_m**2).sum(0)
            #MS_a_entity = 1.0
            #MS_m_entity = 1.0
            #MS_eoccur = 1.0
            #MS_a_events = 1.0
            #MS_m_events = 1.0

            # only set this once (not in below two)
            rho = (iteration + self.params.tau) ** (-1.0 * self.params.kappa)

            # update each variational parameter with average over samples
            self.a_entity += rho * (1. / self.params.num_samples) * \
                (g_entity_a / np.sqrt(MS_a_entity) * \
                (p_entity - q_entity - cv_a_entity)).sum(0)
            self.m_entity += rho * (1. / self.params.num_samples) * \
                (g_entity_m / np.sqrt(MS_m_entity) * \
                (p_entity - q_entity - cv_m_entity)).sum(0)
            #self.a_docspar += rho * (1. / self.params.num_samples) * \
            #    (g_docspar_a / np.sqrt(MS_a_docspar) * \
            #    (p_docspar - q_docspar - cv_a_docspar)).sum(0)
            #self.m_docspar += rho * (1. / self.params.num_samples) * \
            #    (g_docspar_m / np.sqrt(MS_m_docspar) * \
            #    (p_docspar - q_docspar - cv_m_docspar)).sum(0)
            if iteration >= 200:
                #if iteration >= 500:
                self.l_eoccur += rho * (1. / self.params.num_samples) * \
                    (g_eoccur / np.sqrt(MS_eoccur) * \
                    (p_eoccur - q_eoccur - cv_eoccur)).sum(0)
                self.a_events += rho * (1. / self.params.num_samples) * \
                    (g_events_a / np.sqrt(MS_a_events) * \
                    (p_events - q_events - cv_a_events)).sum(0)
                self.m_events += rho * (1. / self.params.num_samples) * \
                    (g_events_m / np.sqrt(MS_m_events) * \
                    (p_events - q_events - cv_m_events)).sum(0)

            # truncate variational parameters
            self.a_entity[self.a_entity < iSP(0.005)] = iSP(0.005)
            self.a_entity[self.a_entity > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
            self.m_entity[self.m_entity < iSP(1e-5)] = iSP(1e-5)
            self.m_entity[self.m_entity > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
            #self.a_docspar[self.a_docspar < iSP(0.005)] = iSP(0.005)
            #self.a_docspar[self.a_docspar > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
            #self.m_docspar[self.m_docspar < iSP(1e-5)] = iSP(1e-5)
            #self.m_docspar[self.m_docspar > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
            self.l_eoccur[self.l_eoccur > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
            self.a_events[self.a_events < iSP(0.005)] = iSP(0.005)
            self.a_events[self.a_events > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))
            self.m_events[self.m_events < iSP(1e-5)] = iSP(1e-5)
            self.m_events[self.m_events > iSP(np.log(sys.float_info.max))] = iSP(np.log(sys.float_info.max))

            # set params with expectation
            self.entity = SP(self.m_entity)
            #self.docspar = SP(self.m_docspar)

            if self.params.event_dist == "Poisson":
                self.eoccur = SP(self.l_eoccur)
            else:
                self.eoccur = S(self.l_eoccur)

            self.events = SP(self.m_events)


            print "entity", self.entity
            #print "sparsity", self.docspar
            print "events", self.eoccur.T
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
    parser.add_argument('meta_filename', type=str, \
        help='a path to document metadata; one line per document with integer values (sender, receiver, time)')
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
        default=1e-5, help = 'likelihood threshold for convergence, default 1e-5')
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
        default=0.1, help = 'sparsity prior on entities; default 0.1')
    parser.add_argument('--m_entities', dest='m_entities', type=float, \
        default=1.0, help = 'mean prior on entities; default 1')
    parser.add_argument('--a_events', dest='a_events', type=float, \
        default=0.1, help = 'sparsity prior on events; default 0.1')
    parser.add_argument('--m_events', dest='m_events', type=float, \
        default=1.0, help = 'mean prior on events; default 1')
    parser.add_argument('--a_docspar', dest='a_docspar', type=float, \
        default=0.1, help = 'sparisty prior on document sparsity; default 0.1')
    parser.add_argument('--m_docspar', dest='m_docspar', type=float, \
        default=1.0, help = 'mean prior on document sparsity; default 1.0')
    parser.add_argument('--event_occur', dest='event_occurance', type=float, \
        default=0.5, help = 'prior to how often events should occur; range [0,1] and default 0.5')

    parser.add_argument('--event_dur', dest='event_duration', type=int, \
        default=7, help = 'the length of time an event can be relevant; default 7')
    parser.add_argument('--event_dist', dest='event_dist', type=str, \
        default="Bernoulli", help = 'what distribution used to model event occurance: \"Poisson\" or \"Bernoulli\" (default)')

    # start a profile of program
    pr = cProfile.Profile()
    pr.enable()

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
        args.a_entities, args.m_entities, args.a_events, args.m_events, args.a_docspar, args.m_docspar, args.event_occurance, \
        args.event_duration, args.event_dist, \
        args.content_filename, args.meta_filename)
    params.save(args.seed, args.message)

    # read in data
    data = Corpus(args.content_filename, args.meta_filename, params.f)

    ## Fit model
    model = Model(data, params)
    model.fit()

    #print out profile data
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
