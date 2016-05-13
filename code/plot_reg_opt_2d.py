
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import collections
import json

# Load dataset- and classifier names
clf_file = __import__('2d_classifier_reg_opt')

opt_vals = collections.defaultdict(dict)

def mean_confidence(data):
    mu = np.mean(data)
    sem = scipy.stats.sem(data)

    lw, hi = scipy.stats.t.interval(0.95, len(data) - 1, loc=mu, scale=sem)
    return (mu, hi - mu)

figure = plt.figure(figsize=(12.5, 10))
subplot_index = 0
for ds_idx, ds_name in enumerate(clf_file.dataset_name):
    for clf_idx, clf_name in enumerate(clf_file.classifier_name):
        subplot_index += 1

        # Load file
        filename = 'cv-scores-ds-%d-clf-%d' % (ds_idx, clf_idx)
        X = np.loadtxt('./output/cv_results/%s.csv' % (filename))

        # Extract scores
        reg_values = X[0, :]
        missrate = (1 - X[1:, :])

        missrate_means = np.zeros(len(reg_values))
        missrate_cis = np.zeros(len(reg_values))
        for reg_i, reg_missrate in enumerate(missrate.T):
            (missrate_means[reg_i], missrate_cis[reg_i]) = mean_confidence(reg_missrate)

        opt_vals[ds_name][clf_name] = reg_values[np.argmin(missrate_means)]

        plt.subplot(len(clf_file.dataset_name), len(clf_file.classifier_name), subplot_index)
        plt.errorbar(reg_values, missrate_means, yerr=missrate_cis, color="SteelBlue")
        plt.grid(True)
        plt.xscale('log')
        plt.ylim([0, 1])

        if clf_idx != 0:
            plt.gca().yaxis.set_ticklabels([])
            plt.gca().yaxis.set_ticks_position('none')
        if ds_idx != len(clf_file.dataset_name) - 1:
            plt.gca().xaxis.set_ticklabels([])
            plt.gca().xaxis.set_ticks_position('none')

        if clf_idx == 0:
            plt.ylabel(ds_name, fontsize=16)
        if ds_idx == 0:
            plt.title(clf_name, fontsize=16)

with open('./output/classifier_2d_opt_param_values.json', 'w') as f:
    json.dump(opt_vals, f, indent=2)

plt.tight_layout()
plt.savefig('../syntetic_reg_opt.pdf', bbox_inches='tight')
