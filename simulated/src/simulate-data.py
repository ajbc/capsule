import numpy as np

D = 25
d = 5
event_rate = 0.3
cable_rate = 10
K = 6
base = np.random.gamma(100,0.001,size=K)
print base

#events = np.random.poisson(event_rate, D)
#events[events > 1] = 1
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
    if True:#events[i] != 0:
        event_content[i] = np.random.gamma(0.02,20,size=K)
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
