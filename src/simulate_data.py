import numpy as np
import sys
np.random.seed(238956)#13)
#np.random.seed(int(sys.argv[1]))

D = 100#100 # number of days
d = int(sys.argv[1]) # duration of event
N = 10 # number of entities
alpha = float(100)#sys.argv[1])#10 # controls proportion; bigger means more even, smaller means less even
n_freq = np.random.dirichlet(alpha=np.ones(N)*alpha) # proportion of messages

cable_rate = 100
K = 10
V = 1000

eventDescr = []
for d in range(D):
    eventDescr.append(np.random.dirichlet(alpha=np.ones(V)*0.01))
eventStren = np.random.gamma(0.5, 5.0, D)

topics = []
for k in range(K):
    topics.append(np.random.dirichlet(alpha=np.ones(V)*0.01))

concerns = []
for n in range(N):
    concerns.append(np.random.gamma(0.5, 5.0, K))

cables = np.random.poisson(cable_rate, D)
senders = []
for i in range(D):
    senders.append(np.random.choice(range(N), p=n_freq, size=cables[i]))

event_content = {}
cable_content = {}
fout = open("meta.tsv", 'w+')
fout2a = open("train.tsv", 'w+') # doc content
fout2b = open("test.tsv", 'w+')
fout2c = open("validation.tsv", 'w+')
fout3 = open("event_description.tsv", 'w+')
fout4 = open("entity_concern.tsv", 'w+')
fout5 = open("event_strength.tsv", 'w+')
fout6 = open("topics.tsv", 'w+')
fout7 = open("local_concerns.tsv", 'w+')
fout8 = open("local_events.tsv", 'w+')
fout7.write("doc\ttopic\tvalue\n")
fout8.write("doc\tevent\tvalue\n")

fout5.write("day\tstrength\n")
fout3.write("day\tterm\tvalue\n")
for d in range(D):
    fout5.write("%d\t%f\n" % (d, eventStren[d]))
    for v in range(V):
        fout3.write('%d\t%d\t%f\n' % (d, v, eventDescr[d][v]))
fout5.close()
fout3.close()

fout4.write("entity\ttopic\tvalue\n")
for n in range(N):
    for v in range(V):
        fout4.write('%d\t%d\t%f\n' % (n, v, concerns[n][v]))
fout4.close()

fout6.write("topic\tterm\tvalue\n")
for k in range(K):
    for v in range(V):
        fout4.write('%d\t%d\t%f\n' % (k, v, topics[k][v]))
fout6.close()

doc = 0
for i in range(D):
    print "DAY", i, "***********"
    print "# of cables for today:", cables[i]

    for cable in range(cables[i]):
        sender = senders[i][cable]
        fout.write("%d\t%d\t%d\n" % (doc, sender, i))

        # draw local event and entity params
        theta = np.random.gamma(0.1, concerns[sender])
        epsilon = np.random.gamma(0.1, eventStren[i])
        for k in range(K):
            fout7.write('%d\t%d\t%f\n' % (doc, k, theta[k]))

        #TODO: write out

        mean = np.zeros(V)
        for j in range(max(i-d+1,0),min(i+1,D)):
            f = 1 - ((0.0+i-j)/d)
            print '\tadding in %f x event %d' % (f,j)
            mean += eventDescr[j] * f * epsilon[j]
            fout8.write('%d\t%d\t%f\n' % (doc, j, epsilon[j]))
        mean += np.array(np.matrix(theta) * np.matrix(topics))[0]
        content = np.random.poisson(mean)

        c = np.random.rand()
        if c < 0.01:
            for t in range(V):
                if content[t] != 0:
                    fout2c.write("%d\t%d\t%d\n" % (doc, t, content[t]))
        elif c < 0.1:
            for t in range(V):
                if content[t] != 0:
                    fout2b.write("%d\t%d\t%d\n" % (doc, t, content[t]))
        else:
            for t in range(V):
                if content[t] != 0:
                    fout2a.write("%d\t%d\t%d\n" % (doc, t, content[t]))

        doc += 1

fout.close()
fout2a.close()
fout2b.close()
fout2c.close()
fout3.close()
fout7.close()
fout8.close()
#print n_freq
