CC = g++ -O3 -larmadillo -lgsl -Wall -fopenmp

LSOURCE = main.cpp utils.cpp data.cpp capsule.cpp
CSOURCE = utils.cpp data.cpp


# main model
capsule: $(LSOURCE)
	  $(CC) $(LSOURCE) -o capsule

profile: $(LSOURCE)
	  $(CC) $(LSOURCE) -o capsule -pg

# cleanup
clean:
	-rm -f capsule
