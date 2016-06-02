# To Do list:

## Writing

- [X] add discussion of relationship between our model and Poisson/Hawkes processes

- [X] discuss pipeline to recover top cables for an event

- [X] add in exploratory results (figures + discussion)

- [X] take a pass through document to ensure everything is up to date with current model

- [ ] add contributions to introduction

- [ ] take another pass at related work (incl. looking for new work)

- [ ] polish eval section

- [ ] polish discussion

- [ ] take several editing passes through the whole draft

- [ ] clean up appendix (make sure notation consistent)

- [ ] add arxiv to appendix as a negative results (no real events), possible include other simualted results


## Results

- [?] pinpoint a measure of "eventness" and make plots of events over time (we have one that works well, but we still looking for one that is more interpretable)

- [X]  compute held out log likelihood for full model, event only, and entity only.  (on cables data; just waiting for fits to finish)

- [X] simulations: show that our model outperforms baselines on detecting "eventness," recovering relevant documents, or preferably both

- [X] model update: include entity-specific parameter (implement, test, and kick off new fits)

- [ ] general "terminal based browser" text files


## After Submission

- [ ] Investigate daily model of cables

- [ ] figure out issue of held out log likelihood on cables (event only does better than full)

- [ ] improved measure of eventness, possibly using pi

- [ ] find cables that come after the given time interval, but are still related to the event

- [ ] apply the model to another dataset
