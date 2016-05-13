rm(list=ls());
cat('\f');

library(ggplot2)

dat = read.csv('./output/classifier-significance-short.csv')

p = ggplot(dat, aes(y=missclassification, x=model, fill=dataset))
p = p + geom_boxplot()
p = p + ylab("Missclassification rate")
ggsave("../2d_significant.pdf", p, width = 12, height = 10, units = "cm")
