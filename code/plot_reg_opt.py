
import os
import numpy as np
import matplotlib.pyplot as plt

# Load values
filename = './output/reg_opt_dieleman_2016-05-11-15-10-09.csv'
X = np.loadtxt(filename)

reg_values = X[0,:]
missrates =  X[1:,:]

missrate_means = missrates.mean(axis=0)
missrate_sds = missrates.std(axis=0)

plt.figure()
#plt.plot(reg_values, missrates, color='IndianRed')
plt.errorbar(reg_values, missrate_means, yerr=missrate_sds, color="IndianRed")
plt.ylabel('Missrate')
plt.xlabel('Weight decay parameter')
plt.xscale('log')
plt.show()
