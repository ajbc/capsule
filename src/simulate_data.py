import numpy as np

D = 25 # number of days total
d = 5 # number of days an event lasts
event_rate = 0.1 # how frequent should events occur?
cable_rate = 100 # how frequent should cables occur?
K = 6 # how many dimensions should there be to docs, events, and entities

def gamma(sparsity, mean, size):
    return np.random.gamma(sparsity, mean/sparsity, size)

base = gamma(0.1,1,size=K)
print base

events = np.random.binomial(1, event_rate, D)
cables = np.random.poisson(cable_rate, D)
event_content = {}
cable_content = {}
fout = open("simulated_time.dat", 'w+')
fout2 = open("simulated_content.tsv", 'w+')
fout3 = open("simulated_truth.tsv", 'w+')
fout3.write("source\ttime%s\n" % ('\tk.%d'*K % tuple(range(K))))
fout3.write("base\t-1%s\n" % ('\t%f'*K % tuple(base)))
for i in range(D):
    #print "event:",events[i], "\t# cables:", cables[i]
    print "DAY", i, "**********"
    if events[i] != 0:
        event_content[i] = gamma(0.1,1,size=K)
        print "\tevent **", event_content[i]
        fout3.write("event\t%d%s\n" % (i,'\t%f'*K % tuple(event_content[i])))

    shape = np.zeros(K) + base
    print "constructing shape"
    print "\tbase"
    print "\t\t", shape
    for j in range(max(i-d+1,0),min(i+1,D)):
        if j in event_content:
            f = 1 - ((0.0+i-j)/d)
            print '\tadding in %f x event %d' % (f,j)
            print "\t\t+", (f*event_content[j])
            shape += event_content[j] * f
            print "\t\t=", shape
    print "shape", shape
    for cable in range(cables[i]):
        fout.write("%d\n" % i)
        content = np.random.gamma(shape,1,size=K)
        fout2.write(('\t%f'*K % tuple(content)).strip() + '\n')
        #print content
fout.close()
fout2.close()
fout3.close()
