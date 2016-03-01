CC = g++ -O3 -larmadillo -lgsl -Wall

LSOURCE = main.cpp utils.cpp data.cpp capsule.cpp eval.cpp
CSOURCE = utils.cpp data.cpp eval.cpp


# main model
capsule: $(LSOURCE)
	  $(CC) $(LSOURCE) -o capsule

profile: $(LSOURCE)
	  $(CC) $(LSOURCE) -o capsule -pg

# comparison methods
baselines: baselines.cpp $(CSOURCE)
	  $(CC) baselines.cpp $(CSOURCE) -o baselines

# cleanup
clean:
	-rm -f capsule baselines
