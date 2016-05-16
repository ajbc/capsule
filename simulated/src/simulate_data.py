import numpy as np
import sys
np.random.seed(238956)#13)
#np.random.seed(int(sys.argv[1]))

D = 100#100 # number of days
dur = int(sys.argv[1]) # duration of event
N = 10 # number of entities
alpha = float(100)#sys.argv[1])#10 # controls proportion; bigger means more even, smaller means less even
n_freq = np.random.dirichlet(alpha=np.ones(N)*alpha) # proportion of messages
shape = sys.argv[2] # option: step, linear, exp
if shape not in ["step", "linear", "exp"]:
    print "bad decay shape"
    sys.exit(-1)

cable_rate = 100
K = 10
V = 1000

eventDescr = []
for d in range(D):
    eventDescr.append(np.random.dirichlet(alpha=np.ones(V)*0.01))
eventStren = np.random.gamma(1.0, 5.0, D)

topics = []
for k in range(K):
    topics.append(np.random.dirichlet(alpha=np.ones(V)*0.01))

concerns = []
for n in range(N):
    concerns.append(np.random.gamma(1.0, 5.0, K))

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
    fout5.write("%d\t%e\n" % (d, eventStren[d]))
    for v in range(V):
        fout3.write('%d\t%d\t%e\n' % (d, v, eventDescr[d][v]))
fout5.close()
fout3.close()

fout4.write("entity\ttopic\tvalue\n")
for n in range(N):
    for k in range(K):
        fout4.write('%d\t%d\t%e\n' % (n, k, concerns[n][k]))
fout4.close()

fout6.write("topic\tterm\tvalue\n")
for k in range(K):
    for v in range(V):
        fout6.write('%d\t%d\t%e\n' % (k, v, topics[k][v]))
fout6.close()

doc = 0
for i in range(D):
    print "DAY", i, "***********"
    print "# of cables for today:", cables[i]

    for cable in range(cables[i]):
        sender = senders[i][cable]
        fout.write("%d\t%d\t%d\n" % (doc, sender, i))

        notdone = True
        while notdone:
            # draw local event and entity params
            theta = np.random.gamma(0.1, 1.0/concerns[sender])
            epsilon = np.random.gamma(0.1, 1.0/eventStren)

            #TODO: write out

            mean = np.zeros(V)
            if shape != "exp":
                for j in range(max(i-dur+1,0),i+1):
                    if shape == "linear":
                        f = 1 - ((0.0+i-j)/dur)
                    else:
                        f = 1.0
                    #print '\tadding in %e x event %d (day %d, duration %d)' % (f, j, i, dur)
                    mean += eventDescr[j] * f * epsilon[j]
            else:
                for j in range(0,i+1):
                    f = np.exp(-(0.0+i-j)/dur)
                    #print '\tadding in %e x event %d (day %d, duration %d)' % (f, j, i, dur)
                    mean += eventDescr[j] * f * epsilon[j]
            mean += np.array(np.matrix(theta) * np.matrix(topics))[0]
            content = np.zeros(V)
            redo_count = 0
            while np.count_nonzero(content) < 2:
                content = np.random.poisson(mean)
                redo_count += 1
                if redo_count == 10:
                    break
                #print "doc", doc, "words", np.count_nonzero(content)
            if redo_count < 10:
                notdone = False

        for k in range(K):
            fout7.write('%d\t%d\t%e\n' % (doc, k, theta[k]))
        if shape == "exp":
            for j in range(0,i+1):
                fout8.write('%d\t%d\t%e\n' % (doc, j, epsilon[j]))
        else:
            for j in range(max(i-dur+1,0),i+1):
                fout8.write('%d\t%d\t%e\n' % (doc, j, epsilon[j]))

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
