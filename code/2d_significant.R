rm(list=ls());
cat('\f');

library(ggplot2)

dat = read.csv('./output/classifer-significance-short.csv')

p = ggplot(dat, aes(y=score, x=model, fill=dataset))
p = p + geom_boxplot()
print(p)
ggsave("../2d_significant.pdf", p, width = 12, height = 10, units = "cm")
