
import os
import numpy as np
import matplotlib.pyplot as plt

# Load values
filename = './output/reg_opt_dieleman_2016-05-11-19-59-46.csv'
#filename = './output/geometry_classifier_offset_2016-05-11-20-38-14.csv'
#filename = './output/geometry_classifier_offset_2016-05-11-21-26-29.csv'
#filename = './output/reg_opt_dieleman_2016-05-12-01-29-06.csv'
#filename = './output/reg_opt_dieleman_2016-05-12-01-46-47.csv'
#filename = './output/reg_opt_dieleman_2016-05-12-02-19-49.csv'
X = np.loadtxt(filename)

reg_values = X[0,:]
missrates =  X[1:,:]

missrate_means = missrates.mean(axis=0)
missrate_sds = missrates.std(axis=0)

plt.figure()
#plt.plot(reg_values, missrates, color='IndianRed')
plt.errorbar(reg_values, missrate_means, yerr=missrate_sds * 2.776 / np.sqrt(5), color="IndianRed")
plt.ylabel('Misclassification rate')
plt.xlabel('Weight decay parameter')
plt.xscale('log')
plt.savefig('./output/reg_opt_dieleman_speaker_elsdsr.eps', bbox_inches='tight')
plt.show()
