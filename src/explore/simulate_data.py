import numpy as np
import sys
np.random.seed(int(sys.argv[1]))

T = 100     # time steps
K = 10      # latent topics
N = 100     # entities
D = 100000  # documents
V = 1000    # vocabulary

# generate metadata associated with each document
authors = np.random.choice(N, D, replace=True)
doctime = np.random.choice(T, D, replace=True)

# generate interval parameters
pi = np.random.dirichlet([1./V]*V, T)                   # interval description
psi = np.random.gamma(0.3, scale=1./0.3, size=T)        # interval strength

# generate entity parameters
eta = np.random.dirichlet([1./V]*V, N)                  # entity description
xi = np.random.gamma(0.3, scale=1./0.3, size=N)         # entity strength

# generate topic parameters
beta = np.random.dirichlet([1./V]*V, K)                 # topics
phi = np.random.gamma(0.3, scale=1./0.3, size=(N,K))    # entity general concerns

# local parameters
theta = np.zeros((D,K))     # topics
epsilon = np.zeros((D,T))   # entity relevancy
zeta = np.zeros(D)          # event relevancy

word_counts = np.zeros((D,V))
for d in range(D):
    theta[d] = np.random.gamma(0.3, phi[authors[d]])
    epsilon[d] = np.random.gamma(0.3, psi)
    zeta[d] = np.random.gamma(0.3, xi[authors[d]])

    f = np.zeros(T)
    for t in range(0, doctime[d]+1):
        f[t] = np.exp(-(doctime[d]-t) / 3.0)

    topics = (theta[d] * beta.T).sum(1)
    entity = zeta[d] * eta[authors[d]]
    events = (f * epsilon[d] * pi.T).sum(1)

    word_counts[d] = np.random.poisson(topics + entity + events)


nz = np.count_nonzero(word_counts.sum(0))
print nz, '/', V, "=", (nz / (V+0.0) * 100), "%"

nz = np.count_nonzero(word_counts.sum(1))
print nz, '/', D, "=", (nz / (D+0.0) * 100), "%"

##
eta = eta[:,word_counts.sum(0) > 5]
pi = pi[:,word_counts.sum(0) > 5]
beta = beta[:,word_counts.sum(0) > 5]
wc = word_counts[:,word_counts.sum(0) > 5]
nz = np.count_nonzero(wc.sum(0))
V = wc.shape[1]
print nz, '/', V, "=", (nz / (V+0.0) * 100), "%"

theta = theta[wc.sum(1) > 2, :]
epsilon = epsilon[wc.sum(1) > 2, :]
zeta = zeta[wc.sum(1) > 2]
authors = authors[wc.sum(1) > 2]
doctime = doctime[wc.sum(1) > 2]
wc = wc[wc.sum(1) > 2,:]
nz = np.count_nonzero(wc.sum(1))
D = wc.shape[0]
print nz, '/', D, "=", (nz / (D+0.0) * 100), "%"

nz = np.count_nonzero(wc.sum(0))
print nz, '/', V, "=", (nz / (V+0.0) * 100), "%"

# save 'em
np.savetxt('pi.txt', pi)
np.savetxt('psi.txt', psi)
np.savetxt('eta.txt', eta)
np.savetxt('xi.txt', xi)
np.savetxt('beta.txt', beta)
np.savetxt('phi.txt', phi)
np.savetxt('theta.txt', theta)
np.savetxt('epsilon.txt', epsilon)
np.savetxt('zeta.txt', zeta)

fout_train = open("train.tsv", 'w+')
fout_test = open("test.tsv", 'w+')
fout_valid = open("validation.tsv", 'w+')
fout_meta = open("meta.tsv", "w+")
for d in range(D):
    fout_meta.write("%d\t%d\t%d\n" % (d, authors[d], doctime[d]))

    for v in range(V):
        if wc[d,v] == 0:
            continue
        rv = np.random.random()
        if rv < 0.05:
            fout_test.write("%d\t%d\t%d\n" % (d, v, wc[d,v]))
        elif rv < 0.1:
            fout_valid.write("%d\t%d\t%d\n" % (d, v, wc[d,v]))
        else:
            fout_train.write("%d\t%d\t%d\n" % (d, v, wc[d,v]))

