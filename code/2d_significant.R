rm(list=ls());
cat('\f');

library(ggplot2)

dat = read.csv('./output/classifer-significance.csv')

p = ggplot(dat, aes(y=score, x=model, fill=dataset))
p = p + geom_boxplot()
print(p)
