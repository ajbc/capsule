library(ggplot2)
library(reshape2)
setwd('~/Dropbox/Projects/Academic/declass/cables/src/simulated')

args <- commandArgs(trailingOnly = TRUE)
DATA <- args[1]
FIT <- args[2]
ITER <- as.integer(args[3])
#DATA <- "~/Projects/Academic/declass/cables/src/simulated/dat/simk6_v5"
#FIT <- "~/Projects/Academic/declass/cables/src/fit/27"
#ITER <- 6

dat.truth <- read.csv(paste(DATA, 'simulated_truth.tsv', sep='/'), sep='\t')
dat.truth <- melt(dat.truth, id.vars=c("source", "time"))

# observed cables
dat.obs <- read.csv(paste(DATA, 'simulated_content.tsv', sep='/'), sep='\t', header=F)
dat.obs.time <- read.csv(paste(DATA, 'simulated_time.dat', sep='/'), header=F)
colnames(dat.obs) <- c('k.0', 'k.1', 'k.2', 'k.3', 'k.4','k.5')
dat.obs$time <- dat.obs.time$V1

dat.truth$type <- "ground truth"
dat.obs$type <- "observed data"
dat.truth$source <- NULL

dat.obs <- melt(dat.obs, id.vars=c("type", "time"))
dat <- rbind(dat.obs, dat.truth)


# fitted vs truth
for (iter in seq(0,ITER,1)) {
  print(sprintf('creating plot for iteration %d', iter))
  iterstr <- sprintf('%04d', iter)
  dat.fit0 <- read.csv(paste(FIT, '/entities_', iterstr, '.tsv', sep=''), sep='\t', header=F)
  dat.fit0$time <- -1
  dat.fit0$alpha <- 1.0
  dat.fit <- read.csv(paste(FIT, '/events_', iterstr, '.tsv', sep=''), sep='\t', header=F)
  dat.occur <- read.csv(paste(FIT, '/eoccur_', iterstr, '.tsv', sep=''), sep='\t', header=F)
  dat.fit$time <- seq(0,nrow(dat.fit)-1)
  dat.fit$alpha <- dat.occur$V1
  dat.fit <- rbind(dat.fit0, dat.fit)
  colnames(dat.fit) <- c('k.0', 'k.1', 'k.2', 'k.3', 'k.4','k.5', 'time', 'alpha')
  dat.fit <- melt(dat.fit, id.vars=c("time", "alpha"))
  dat.fit$value <- dat.fit$value * dat.fit$alpha
  dat.fit$alpha <- 1
  dat.fit$type <- "learned events"

  dat.fit[dat.fit$value > 10,4] <- 10

  dat.truth$alpha <- 1.0
  dat.obs$alpha <- 0.1

  dat <- rbind(dat.obs, dat.truth, dat.fit)

  p <- ggplot(dat, aes(x=time, y=value, fill=type, alpha=alpha)) + facet_wrap(~ variable)
  p <- p + geom_bar(stat='identity', position="dodge") #+ geom_segment(aes(xend=time, yend=0))
  p <- p + ylim(0,10)
  p

  ggsave(file=paste("events.iter", iterstr, "jpg", sep="."), width=8, height=5)
}

# likelihood
LL <- read.csv(paste(FIT, 'log.dat', sep='/'), sep='\t')

ymin <- min(LL$ELBO)
ymax <- max(LL$ELBO)

LL$ll.change <- NULL
LL$ELBO.change <- NULL
LL$time <- NULL
LL <- melt(LL, id.vars=c("iteration"))


ggplot(LL, aes(x=iteration, y=value, color=variable)) + geom_line() + geom_segment(aes(x=min(LL$iteration), xend=max(LL$iteration), y=ymin, yend=ymax), color='black', size=0.1) +theme_bw()
ggsave(file='log.pdf',width=5, height=3)


# from pre-melt era
# ggplot(LL, aes(x=iteration, y=likelihood)) + geom_line() + geom_segment(aes(x=min(LL$iteration), xend=max(LL$iteration), y=min(LL$likelihood), yend=max(LL$likelihood)), color='green', size=0.1)
# ggsave(file='likelihood.pdf',width=5, height=3)

# LL <- LL[LL$iteration!=1,]
# ggplot(LL, aes(x=iteration, y=change)) + geom_point() + geom_line() + geom_smooth()
# ggsave(file='likelihood_change.pdf',width=5, height=3)
