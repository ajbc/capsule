import numpy as np
import shutil, os, sys
from datetime import datetime as dt
from scipy.special import gammaln, digamma

import warnings #TODO rm
warnings.filterwarnings('error')
from collections import defaultdict #TODO rm

## helper functions

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
        a_ent, b_ent, a_evn, b_evn, b_doc, event_duration):
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

        self.d = event_duration

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
        self.t_entity = 0
        self.t_events = np.zeros(self.data.day_count())

        self.a_entity = np.ones((1,self.data.dimension)) #* self.params.a_entity
        self.b_entity = np.ones((1,self.data.dimension)) #* self.params.b_entity
        self.a_events = np.ones((self.data.day_count(), self.data.dimension)) #* self.params.a_events
        #self.a_events = np.random.gamma(self.params.a_events, 1.0/self.params.b_events, \
        #    (self.data.day_count(), self.data.dimension))
        self.b_events = np.ones((self.data.day_count(), self.data.dimension)) #* self.params.b_events

        self.entity = self.a_entity / self.b_entity
        print "starting entity"
        print self.entity
        self.events = self.a_events / self.b_events

        self.likelihood_decreasing_count = 0

    def compute_likelihood(self):
        log_likelihood = 0
        f_array = np.zeros((self.data.day_count(),1))
        for doc in self.data.validation:
            for day in range(self.data.day_count()):
                f_array[day] = self.params.f(self.data.days[day], doc.day)
            doc_params = self.entity + sum(f_array*self.events)
            p_doc_a = self.params.b_docs * doc_params
            p_doc_a[p_doc_a < np.sqrt(sys.float_info.max)] = - np.sqrt(sys.float_info.max)
            p_doc_a[p_doc_a > np.sqrt(sys.float_info.max)] = np.sqrt(sys.float_info.max)
            #print "log like components"
            #print "p doc a", p_doc_a
            #print "log b  ", np.log(self.params.b_docs)
            #print "neg gma", (-1* lngamma(p_doc_a))
            #print "log doc", np.log(doc.rep)
            #print "b doc  ", self.params.b_docs
            #print "doc rep", doc.rep
            log_likelihood += np.sum(p_doc_a * np.log(self.params.b_docs) - \
                lngamma(p_doc_a) + (p_doc_a - 1) * np.log(doc.rep) - \
                self.params.b_docs * doc.rep)
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
            if self.likelihood_decreasing_count == 3:
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
        fout.write(('\t'.join(["%f"]*len(self.entity[0]))+'\n') % tuple(self.entity[0]))
        fout.close()

        fout = open(os.path.join(self.params.outdir, "events_%s.tsv" % tag), 'w+')
        for i in range(len(self.events)):
            fout.write(('\t'.join(["%f"]*len(self.events[i])) +'\n') % tuple(self.events[i]))
        fout.close()

    def fit(self):
        self.init()

        iteration = 0
        self.save('%04d' % iteration) #TODO: rm; this is just for visualization

        print "starting..."
        while not self.converged(iteration):
            iteration += 1
            dcount = 0 #TODO rm
            print "entity", self.entity

            lambda_a_events = np.zeros((self.data.day_count(), self.data.dimension))
            lambda_b_events = np.zeros((self.data.day_count(), self.data.dimension))
            lambda_a_entity = np.zeros((1, self.data.dimension))
            lambda_b_entity = np.zeros((1, self.data.dimension))

            # this was not in figure i sent t dave
            for s in range(self.params.num_samples):
                entity = np.random.gamma(self.a_entity, 1.0/self.b_entity, \
                    (1, self.data.dimension))
                entity[entity == 0] = sys.float_info.min
                events = np.random.gamma(self.a_events, 1.0/self.b_events, \
                    (self.data.day_count(), self.data.dimension))
                events[events == 0] = sys.float_info.min
                if s < 10:
                    print "event 3 sample", events[3]
                #if s < 3:
                #    print ("i%d\ts%d\t"%(iteration,s))
                #    print "entity", entity
                #    print "events", events

                # entity contributions to updates
                p_entity = self.params.a_entity * np.log(self.params.b_entity) - \
                    lngamma(self.params.a_entity) + \
                    (self.params.a_entity - 1) * np.log(entity) - \
                    self.params.b_entity * entity
                q_entity = self.a_entity * np.log(self.b_entity) - \
                    lngamma(self.a_entity) + \
                    (self.a_entity - 1) * np.log(entity) - \
                    self.b_entity * entity
                #if iteration == 2:
                #    print "3 that go into g_entity_a"
                #    print "log of b", np.log(self.b_entity)
                #    print "neg digamma of a", (-1*digamma(self.a_entity))
                #    print "log of entity", np.log(entity)
                g_entity_a = np.log(self.b_entity) - \
                    digamma(self.a_entity) + \
                    np.log(entity)
                g_entity_b = self.a_entity / self.b_entity - entity

                p_events = self.params.a_events * np.log(self.params.b_events) - \
                    lngamma(self.params.a_events) + \
                    (self.params.a_events - 1) * np.log(events) - \
                    self.params.b_events * events
                q_events = self.a_events * np.log(self.b_events) - \
                    lngamma(self.a_events) + \
                    (self.a_events - 1) * np.log(events) - \
                    self.b_events * events
                g_events_a = np.log(self.b_events) - \
                    digamma(self.a_events) + \
                    np.log(events)
                g_events_b = self.a_events / self.b_events - events

                #for doc in self.data.docs:
                for d in range(self.params.batch_size):
                    doc = self.data.random_doc()
                    f_array = np.zeros((self.data.day_count(),1))
                    for day in range(self.data.day_count()):
                        f_array[day] = self.params.f(self.data.days[day], doc.day)

                    # related to control variates TODO: rethink with reorg; it shouldnt work as is
                    h_a_events = np.zeros((self.params.num_samples, self.data.day_count(), self.data.dimension))
                    h_b_events = np.zeros((self.params.num_samples, self.data.day_count(), self.data.dimension))
                    f_a_events = np.zeros((self.params.num_samples, self.data.day_count(), self.data.dimension))
                    f_b_events = np.zeros((self.params.num_samples, self.data.day_count(), self.data.dimension))

                    doc_params = entity + sum(f_array*events)

                    p_doc_a = self.params.b_docs * doc_params
                    p_doc_a[p_doc_a < -sys.float_info.max**(1./4)] = - sys.float_info.max**(1./4)
                    p_doc_a[p_doc_a > sys.float_info.max**(1./4)] = sys.float_info.max**(1./4) #TODO: make all thresholdin glike this more formal (single function)
                    p_doc = p_doc_a * np.log(self.params.b_docs) - \
                        lngamma(p_doc_a) + \
                        (p_doc_a - 1) * np.log(doc.rep) - \
                        self.params.b_docs * doc.rep
                    p_doc[np.isinf(p_doc)] = - np.sqrt(sys.float_info.max)
                    p_doc[p_doc < -sys.float_info.max**(1./4)] = - sys.float_info.max**(1./4)

                    #if iteration == 2:
                    #    print "lambda components [5]"
                    #    print "g", g_entity_a
                    #    print "p ent", p_entity
                    #    print "# doc", self.data.num_docs()
                    #    print "p doc", p_doc
                    #    print "q ent", q_entity
                    lambda_a_entity += g_entity_a * (p_entity + self.data.num_docs() * p_doc - q_entity)
                    lambda_b_entity += g_entity_b * (p_entity + self.data.num_docs() * p_doc - q_entity)

                    h_a_events[s] = g_events_a * (f_array != 0)
                    h_b_events[s] = g_events_b * (f_array != 0)
                    f_a_events[s] = (f_array != 0) * g_events_a * \
                        (p_events + self.data.num_docs() * p_doc - q_events)
                    f_b_events[s] += (f_array != 0) * g_events_b * \
                        (p_events + self.data.num_docs() * p_doc - q_events)
                    lambda_a_events += f_a_events[s]
                    lambda_b_events += f_b_events[s]
                    '''if f_array[22] != 0 and s<10 and dcount < 10:
                        dcount += 1
                        print "sample", s, "document", d, ("(at day %d)\tevent 22 stuff" % doc.day)
                        print "event", events[22]
                        print "sample", s, "document", d, ("(at day %d)\tevent 23 stuff" % doc.day)
                        print "event", events[23]
                        #print "p    ", p_events[22]
                        #print "\tterm 1", (self.params.a_events * np.log(self.params.b_events))
                        #print "\tterm 2", (-1*lngamma(self.params.a_events))
                        #print "\tterm 3", ((self.params.a_events - 1) * np.log(events))[22]
                        #print "\tterm 4", (self.params.b_events * events * -1)[22]
                        #print "q    ", q_events[22]
                        #print "g (a)", g_events_a[22]
                        #print "g (b)", g_events_b[22]
                        #print "f (a)", f_a_events[s][22]
                        #print "f (b)", f_b_events[s][22]'''

                # this doesn't, but it's correct (or is it?)
                #lambda_a_events -= sum(h_a_events) * cov(f_a_events, h_a_events) / var(h_a_events)
                #lambda_b_events -= sum(h_b_events) * cov(f_b_events, h_b_events) / var(h_b_events)

            lambda_a_entity /= self.params.batch_size * self.params.num_samples
            lambda_b_entity /= self.params.batch_size * self.params.num_samples
            lambda_a_events /= self.params.batch_size * self.params.num_samples
            lambda_b_events /= self.params.batch_size * self.params.num_samples
            #lambda_a_entity /= self.data.num_docs() * self.params.num_samples
            #lambda_b_entity /= self.data.num_docs() * self.params.num_samples
            #lambda_a_events /= self.data.num_docs() * self.params.num_samples
            #lambda_b_events /= self.data.num_docs() * self.params.num_samples

            rho = (iteration + self.params.tau) ** (-1.0 * self.params.kappa)
            self.a_entity += rho * lambda_a_entity
            self.b_entity += rho * lambda_b_entity
            self.a_events += rho * lambda_a_events
            self.b_events += rho * lambda_b_events
            print "end of iteration"
            print "event 5 var params"
            print 'a events', self.a_events[5]
            print 'b events', self.b_events[5]
            print "*************************************"

            min_thresh = 1e-10#sys.float_info.min**(1./4)
            max_thresh = 100
            self.a_entity[self.a_entity <= min_thresh] = min_thresh#sys.float_info.min
            self.b_entity[self.b_entity <= min_thresh] = min_thresh#sys.float_info.min
            self.a_events[self.a_events <= min_thresh] = min_thresh#sys.float_info.min
            self.b_events[self.b_events <= 1e-5] = 1e-5#sys.float_info.min
            #self.a_entity[self.a_entity >= max_thresh] = max_thresh
            self.a_events[self.a_events >= max_thresh] = max_thresh
            #print "a for events  ------- "
            #for e in range(self.a_events.shape[0]):
            #    for k in range(self.a_events.shape[1]):
            #        if self.a_events[e,k] > 1e2:
            #            print "event", e, "component", k, "is large!", self.a_events[e,k]
            #print "b for events  ------- "
            #for e in range(self.b_events.shape[0]):
            #    for k in range(self.b_events.shape[1]):
            #        if self.b_events[e,k] > 1e2:
            #            print "event", e, "component", k, "is large!", self.b_events[e,k]


            self.entity = self.a_entity / self.b_entity
            self.events = self.a_events / self.b_events

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
        default=0.3, help = 'shape prior on entities; default 0.3')
    parser.add_argument('--b_entities', dest='b_entities', type=float, \
        default=0.3, help = 'rate prior on entities; default 0.3')
    parser.add_argument('--a_events', dest='a_events', type=float, \
        default=0.3, help = 'shape prior on events; default 0.3')
    parser.add_argument('--b_events', dest='b_events', type=float, \
        default=0.3, help = 'rate prior on events; default 0.3')
    parser.add_argument('--b_docs', dest='b_docs', type=float, \
        default=0.3, help = 'rate prior (and partial shape prior) on documents; default 0.3')

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
    params = Parameters(args.outdir, args.B, args.S, args.save_freq, \
        args.convergence_thresh, args.max_iter, args.tau, args.kappa, \
        args.a_entities, args.b_entities, args.a_events, args.b_events, args.b_docs, \
        args.event_duration)
    params.save(args.seed)


    ## Fit model
    model = Model(data, params)
    model.fit()
