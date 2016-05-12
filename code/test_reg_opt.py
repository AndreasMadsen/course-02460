
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats

# Load values
filename = './output/reg_opt_dieleman_2016-05-11-19-59-46.csv'
X = np.loadtxt(filename)

reg_values = X[0,:]
missrates =  X[1:,:]

missrate_means = missrates.mean(axis=0)
missrate_sds = missrates.std(axis=0)
#missrate_vars = missrates.var(axis=0)

idx_1 = np.where(reg_values == 0.0)
#idx_1 = np.where(reg_values == 1e-10)
idx_2 = np.where(reg_values == 5e-3)

mu_1 = missrate_means[idx_1]
mu_2 = missrate_means[idx_2]
n = missrates.shape[0]
n_1 = n
n_2 = n
s_1 = missrate_sds[idx_1]
s_2 = missrate_sds[idx_2]

# Normal students t-test
# Null-hypothesis: mu_1 = mu_2
print('Student\'s t-test')
df = 4

# t distribution value for two-sided test with df=4
# https://www.easycalculation.com/statistics/t-distribution-critical-value-table.php
t_alpha = 2.7764

s_12 = np.sqrt(s_1 ** 2 + s_2 ** 2)
t = (mu_1 - mu_2) / (s_12 * np.sqrt(1/n))

p = scipy.stats.t.sf(np.abs(t), df) * 2 # Two-sided p-value
print('t =\t\t %.4f' % (t))
print('t_alpha =\t %.4f' % (t_alpha))
print('p-value: %.4f' % (p))


import sys
sys.exit()


# Using a Welch's t-test
print('Welch\'s t-test')

s_12 = np.sqrt((s_1 ** 2) / n_1 + (s_2 ** 2) / n_2)
t = (mu_1 + mu_2)/s_12
print(t)

df = (((s_1 ** 2) / n_1 + (s_2 ** 2) / n_2) ** 2) / \
     ((((s_1 ** 2) / n_1) ** 2) / (n_1 - 1) + (((s_2 ** 2) / n_2) ** 2) / (n_2 - 1))

print(df)

# t distribution value for two-sided test with df=7
t_alpha = 2.3646
