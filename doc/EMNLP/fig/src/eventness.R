library(ggplot2)
library(reshape2)

setwd("/Users/ajbc/Projects/Academic/cables/doc/EMNLP/fig")

dat <- read.csv("dat/eventness2.dat", sep=" ", header=F)
colnames(dat) <- c("week.id", 'eventness', "e2", 'e3')

#dat <- read.csv("dat/eventness.dat", sep=" ", header=F)
#colnames(dat) <- c("week.id", "eventness")

dat <- read.csv("dat/psi-final.dat", header=F)
colnames(dat) <- c("eventness")
dat$week.id <- seq(nrow(dat))-1

cc <- read.csv("dat/cables_counts.tsv", sep="\t", header=F)
colnames(cc) <- c("week.id", "doc.count")
dat$doc.count <- cc$doc.count
dat$week <- seq(as.Date("1973/1/1"), as.Date("1980/12/31"), "weeks")[1:nrow(dat)]

#dat <- dat[1:(nrow(dat)-26),]
#dat$m <- dat$ave.eps*(dat$psi)
ggplot(dat, aes(x=week, y=eventness, yend=min(eventness), xend=week)) +
  geom_hline(aes(yintercept = min(eventness) + 9000/max(doc.count) * (max(eventness) - min(eventness))), linetype="dashed", alpha=0.2, color="#156946") + 
  geom_ribbon(aes(ymin=min(eventness), ymax=min(eventness) + doc.count/max(doc.count) * (max(eventness) - min(eventness))), alpha=0.1) +
  geom_line(color="#AAAAAA", size=0.2) + geom_point(size=0.5) +
  theme_bw() + ylab("\"eventness\"") + xlab("") +
  theme(axis.ticks.y = element_blank(), axis.text.y = element_blank()) +
  scale_x_date(breaks = seq(as.Date("1973/1/1"), as.Date("1979/01/01"), "3 months"),
               labels = c("1973", rep('',3), "1974", rep('',3), "1975", rep('',3), "1976", rep('',3), "1977", rep('',3), "1978", rep('',3), "1989"))
ggsave("cables_events.pdf", width=8.5,height=3.5)





# discovery
ggplot(dat, aes(x=week, y=eventness, yend=min(eventness), xend=week, label=week.id)) +
  geom_hline(aes(yintercept = min(eventness) + 9000/max(doc.count) * (max(eventness) - min(eventness))), linetype="dashed", alpha=0.2, color="#156946") + 
  geom_ribbon(aes(ymin=min(eventness), ymax=min(eventness) + doc.count/max(doc.count) * (max(eventness) - min(eventness))), alpha=0.1) +
  geom_line(color="#AAAAAA", size=0.2) + geom_text() +
  theme_bw() + ylab("\"eventness\"") + xlab("") +
  theme(axis.ticks.y = element_blank(), axis.text.y = element_blank()) +
  scale_x_date(breaks = seq(as.Date("1973/1/1"), as.Date("1979/01/01"), "3 months"),
               labels = c("1973", rep('',3), "1974", rep('',3), "1975", rep('',3), "1976", rep('',3), "1977", rep('',3), "1978", rep('',3), "1989"))


ggsave("cables_events_tmp2.pdf", width=3*8.5,height=3*3.5)