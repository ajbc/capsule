library(ggplot2)
library(reshape2)
setwd('~/Dropbox/Projects/Academic/declass/cables/src/simulated')

dat.truth <- read.csv('src/simulated_truth.tsv', sep='\t')
dat.truth <- melt(dat.truth, id.vars=c("source", "time"))

# events and base distribution: ground truth
p <- ggplot(dat.truth, aes(x=time, y=value)) + facet_wrap(~ variable)
p <- p + geom_point() + geom_segment(aes(xend=time, yend=0))
p


# observed cables
dat.obs <- read.csv('src/simulated_content.tsv', sep='\t', header=F)
dat.obs.time <- read.csv('src/simulated_time.dat', header=F)
colnames(dat.obs) <- c('k.0', 'k.1', 'k.2', 'k.3', 'k.4','k.5')
dat.obs$time <- dat.obs.time$V1

dat.truth$type <- "ground truth"
dat.obs$type <- "observed data"
dat.truth$source <- NULL

dat.obs <- melt(dat.obs, id.vars=c("type", "time"))
dat <- rbind(dat.obs, dat.truth)

p <- ggplot(dat, aes(x=time, y=value, color=type)) + facet_wrap(~ variable)
p <- p + geom_point() #+ geom_segment(aes(xend=time, yend=0))
p



#ggsave(file=paste(paste("k_sweep", dataset, sep="_"), "pdf", sep="."),width=5, height=3)


