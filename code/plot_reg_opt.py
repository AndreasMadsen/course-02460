
import os
import numpy as np
import matplotlib.pyplot as plt

# Load values
#filename = './output/reg_opt_dieleman_2016-05-11-19-59-46.csv'
#filename = './output/reg_opt_dieleman_weight_decay_speaker.csv'
#filename = './output/reg_opt_dieleman_weight_decay_speaker_fine_grid.csv'
filename = './output/reg_opt_dieleman_weight_decay_sex.csv'
X = np.loadtxt(filename)

reg_values = X[0,:]
missrates =  X[1:,:]

nfolds = missrates.shape[0]

missrate_means = missrates.mean(axis=0)
missrate_sds = missrates.std(axis=0)

plt.figure(figsize=(6, 3))
plt.errorbar(reg_values, missrate_means, yerr=missrate_sds * 2.776 / np.sqrt(nfolds), color="IndianRed")
plt.ylabel('Misclassification rate')
plt.xlabel('Weight decay parameter')
plt.xscale('log')
#plt.savefig('../reg_opt_dieleman_speaker_elsdsr.pdf', bbox_inches='tight')
plt.savefig('../reg_opt_dieleman_sex_elsdsr.pdf', bbox_inches='tight')
plt.show()
