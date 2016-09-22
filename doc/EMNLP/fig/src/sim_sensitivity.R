library(ggplot2)
library(reshape2)

setwd("/Users/ajbc/Projects/Academic/dissertation/fig/cables/")

dat <- read.csv("dat/grid.csv", sep=",", header=F)
colnames(dat) <- c("set", "data.type", "fit.type", "dur", "seed", "metric", "value")


ggplot(dat, aes(x=dur, y=value, color=fit.type, linetype=fit.type)) + facet_grid(metric ~data.type, scales="free") +
  geom_jitter(size=1,width=0.25) + geom_smooth(se=F,size=0.5) + theme_bw() + 
  scale_colour_manual(values=c("#56A383", "#D95F92", "#66BADE")) +
  ylab("") + xlab("fit duration tau") +
  theme(legend.title=element_blank(), legend.position="bottom", legend.box = "horizontal")
ggsave("sim_sensitivity.pdf", width=8.5, height=6)