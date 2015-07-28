library(ggplot2)
library(reshape2)
setwd('~/Dropbox/Projects/Academic/declass/cables/src/simulated')

dat.truth <- read.csv('dat/simk6/simulated_truth.tsv', sep='\t')
dat.truth <- melt(dat.truth, id.vars=c("source", "time"))

# events and base distribution: ground truth
p <- ggplot(dat.truth, aes(x=time, y=value)) + facet_wrap(~ variable)
p <- p + geom_point() + geom_segment(aes(xend=time, yend=0))
p


# observed cables
dat.obs <- read.csv('dat/simk6/simulated_content.tsv', sep='\t', header=F)
dat.obs.time <- read.csv('dat/simk6/simulated_time.dat', header=F)
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



# fitted vs truth
dat.fit0 <- read.csv('fit/simk6_3/entities_0010.tsv', sep='\t', header=F)
dat.fit0$time <- -1
dat.fit <- read.csv('fit/simk6_3/events_0010.tsv', sep='\t', header=F)
dat.fit$time <- seq(0,nrow(dat.fit)-1)
dat.fit <- rbind(dat.fit0, dat.fit)
colnames(dat.fit) <- c('k.0', 'k.1', 'k.2', 'k.3', 'k.4','k.5', 'time')
dat.fit <- melt(dat.fit, id.vars=c("time"))
dat.fit$type <- "learned events"

dat.truth$alpha <- 0.5
dat.obs$alpha <- 0.1
dat.fit$alpha <- 0.5

dat <- rbind(dat.obs, dat.truth, dat.fit)

p <- ggplot(dat, aes(x=time, y=value, fill=type, alpha=alpha)) + facet_wrap(~ variable)
p <- p + geom_bar(stat='identity', position="dodge") #+ geom_segment(aes(xend=time, yend=0))
p


# likelihood
LL <- read.csv('fit/simk6_3/log.dat', sep='\t')
ggplot(LL, aes(x=iteration, y=likelihood)) + geom_line() + geom_segment(aes(x=min(LL$iteration), xend=max(LL$iteration), y=min(LL$likelihood), yend=max(LL$likelihood)), color='green', size=0.1)
LL <- LL[LL$iteration!=1,]
ggplot(LL, aes(x=iteration, y=change)) + geom_point() + geom_line() + geom_smooth()


#ggsave(file=paste(paste("k_sweep", dataset, sep="_"), "pdf", sep="."),width=5, height=3)


