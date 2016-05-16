#setwd("C:/Users/frederik/Documents/dtu/advanced/course-02460/code/output")
rm(list=ls());
cat('\f');

library(ggplot2)
dat = read.csv('./classifer-significance-25-trials.csv')

p = ggplot(dat, aes(y=missclassification, x=model, fill=dataset))
p = p + geom_boxplot()
p = p + ylab("Missclassification rate")
print(p)
ggsave("../2d_alternative_boxplot.pdf", p, width = 30, height = 10, units = "cm")
