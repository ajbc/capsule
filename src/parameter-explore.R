library(ggplot2)
library(reshape2)
setwd('~/Dropbox/Projects/Academic/declass/cables/src/simulated')

args <- commandArgs(trailingOnly = TRUE)
DATA <- args[1]
FIT <- args[2]
ITER <- as.integer(args[3])
#DATA <- "~/Projects/Academic/declass/cables/src/simulated/dat/simk6_v3"
#FIT <- "~/Projects/Academic/declass/cables/src/simulated/fit/simk6v3_aug20_full"
#ITER <- 40
# 
# dat.truth <- read.csv(paste(DATA, 'simulated_truth.tsv', sep='/'), sep='\t')
# dat.truth <- melt(dat.truth, id.vars=c("source", "time"))
# dat.truth$type <- "ground truth"
# 
# dat.obs <- melt(dat.obs, id.vars=c("type", "time"))
# dat <- rbind(dat.obs, dat.truth)


# event occur
data <- data.frame()
for (iter in seq(0,ITER)) {
  iterstr <- sprintf('%04d', iter)
  dat.occur <- read.csv(paste(FIT, '/eoccur_', iterstr, '.tsv', sep=''), sep='\t', header=F)
  colnames(dat.occur) <- c("event.occurance")
  dat.occur$iter <- iter
  dat.occur$time <- seq(0,nrow(dat.occur)-1)
  data <- rbind(data, dat.occur)
}


p <- ggplot(data, aes(x=iter, y=event.occurance, group=time, color=factor(time))) + facet_wrap(~ time)
p <- p + geom_line()
p

# event content
data <- data.frame()
for (iter in seq(0,ITER)) {
  iterstr <- sprintf('%04d', iter)
  dat.fit <- read.csv(paste(FIT, '/events_', iterstr, '.tsv', sep=''), sep='\t', header=F)
  colnames(dat.fit) <- c('k.0', 'k.1', 'k.2', 'k.3', 'k.4','k.5')
  dat.fit$iter <- iter
  dat.fit$time <- seq(0,nrow(dat.fit)-1)
  dat.fit <- melt(dat.fit, id.vars=c("time", "iter"))
  data <- rbind(data, dat.fit)
}


p <- ggplot(data, aes(x=iter, y=value, group=variable, color=factor(variable))) + facet_wrap(~ time)
p <- p + geom_line()
p



