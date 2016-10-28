library(ggplot2)
library(reshape2)

setwd("/Users/ajbc/Projects/Academic/cables/doc/EMNLP/fig/")

dat <- read.csv("dat/auc_sim.csv", sep=",", header=F)
colnames(dat) <- c("set", "method", "metric", "value")


dat <- dat[!(dat$method %in% c("word.outlier.tfidf", "word.outlier", "ave.tfidf.dev", "ave.doc.dev", "total.doc.tfidf.dev", "total.doc.dev", "doc.outlier", "ave.word.dev", "ave.doc.tfidf.dev", "doc.outlier.tfidf")),]


ggplot(dat, aes(x=reorder(method, value, FUN=max), y=value)) + geom_violin(alpha=0.2, color="white", fill="#156946") + geom_point(alpha=0.4) +
  #stat_summary(fun.y="mean", colour="#156946", alpha=0.3, geom="point", size=5) +
  theme_bw() + #theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust=0.5)) +
  coord_flip() +
  scale_x_discrete(labels=c("random", "\"event only\" Capsule (this paper)",  "term-count deviation + tf-idf (eq. 7)", "term-count deviation (eq. 6)","Capsule (this paper)")) +
  ylab("event detection performance") + xlab("")
ggsave("sim_eventdetect.pdf", width=5.5, height=2.5)