import numpy as np
np.random.seed(13)

D = 25
d = 5
entities = 8# rate for entity selection (poisson)
event_rate = 0.3
cable_rate = 100
K = 6

#events = np.random.poisson(event_rate, D)
#events[events > 1] = 1
events = np.random.binomial(1, event_rate, D)
cables = np.random.poisson(cable_rate, D)
entity_set = set()
senders = []
receivers = []
for d in range(D):
    #senders.append(np.random.poisson(entities, cables[d]))
    #receivers.append(np.random.poisson(entities, cables[d]))
    senders.append(np.random.randint(0, entities, cables[d]))
    receivers.append(np.random.randint(0, entities, cables[d]))
    #entity_set = entity_set.union(set(senders[d]))
    #entity_set = entity_set.union(set(recievers[d]))  not needed in practice

#if min(entity_set) != 1:
#    for d in range(D):
#        senders[d] -= min(entity_set)
#        #recievers[d] -= min(entity_set) ??
#base = np.random.gamma(0.1,1/0.1,size=(len(entity_set), K))
base = np.random.gamma(0.1,1/0.1,size=(entities, K))
base[base < 0.001] = 0.001
print base

event_content = {}
cable_content = {}
fout = open("simulated_meta.dat", 'w+')
fout2 = open("simulated_content.tsv", 'w+')
fout3 = open("simulated_events.tsv", 'w+')
fout4 = open("simulated_entities.tsv", 'w+')
fout3.write("time%s\n" % ('\tk.%d'*K % tuple(range(K))))
fout4.write("entity%s\n" % ('\tk.%d'*K % tuple(range(K))))
for d in range(entities):#len(entity_set)):
    fout4.write("%d%s\n" % (d, '\t%f'*K % tuple(base[d])))
fout4.close()
#s = dict()
for i in range(D):
    #print "event:",events[i], "\t# cables:", cables[i]
    print "DAY", i, "**********"
    if events[i] != 0:
        event_content[i] = np.random.gamma(0.02,20,size=K)
        print "\tevent **", event_content[i]
        fout3.write("%d%s\n" % (i,'\t%f'*K % tuple(event_content[i])))

    mean = np.zeros(K)
    print "constructing shape"
    print "\tbase"
    print "\t\t", mean
    for j in range(max(i-d+1,0),min(i+1,D)):
        if j in event_content:
            f = 1 - ((0.0+i-j)/d)
            print '\tadding in %f x event %d' % (f,j)
            print "\t\t+", (f*event_content[j])
            mean += event_content[j] * f
            print "\t\t=", mean
    print "mean", mean
    #print min(entity_set), max(entity_set), len(entity_set)
    print "# of cables for today:", cables[i]
    print "corresponding senders:",senders[i]
    for sender in senders[i]:
        print sender, base[sender]

    for cable in range(cables[i]):
        print "day", i, "cable", cable, "sender", senders[i][cable]
        #if senders[i][cable] not in s:
        #    s[senders[i][cable]] = len(s)
        #sender = s[senders[i][cable]]
        sender = senders[i][cable]
        fout.write("%d\t%d\t%d\n" % (sender, receivers[i][cable], i))
        content = np.random.gamma(0.1,(mean+base[senders[i][cable]])/0.1,size=K)
        #content = np.random.gamma(0.1,(base[sender])/0.1,size=K)
        content[content < 0.01] = 0.01
        fout2.write(('\t%f'*K % tuple(content)).strip() + '\n')
        #print content
fout.close()
fout2.close()
fout3.close()
