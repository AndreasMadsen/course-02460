
import os
import numpy as np
import matplotlib.pyplot as plt
import collections
import json

# Load dataset- and classifier names
clf_file = __import__('2d_classifier_reg_opt')

opt_vals = collections.defaultdict(dict)

for ds_idx, ds_name in enumerate(clf_file.dataset_name):
    for clf_idx, clf_name in enumerate(clf_file.classifier_name):

        # Load file
        filename = 'cv-scores-ds-%d-clf-%d' % (ds_idx, clf_idx)
        X = np.loadtxt('./output/cv_results/%s.csv' % (filename))

        # Extract scores
        reg_values = X[0,:]
        scores =  X[1:,:]

        nfolds = scores.shape[0]

        score_means = scores.mean(axis=0)
        score_sds = scores.std(axis=0)

        min_idx = np.argmax(score_means)
        opt_val = reg_values[min_idx]

        opt_vals[ds_name][clf_name] = opt_val

        plt.figure()
        plt.errorbar(reg_values, score_means, yerr=score_sds * 2.776 / np.sqrt(nfolds),
                     color="IndianRed", label="%s & %s" % (ds_name, clf_name))
        plt.ylabel('Score')
        plt.xlabel('Weight decay parameter')
        plt.title(ds_name)
        plt.xscale('log')
        plt.ylim([0, 1])
        plt.title("Optimal: %.0e" % (opt_val))
        plt.legend(loc=3)
        plt.savefig('./output/cv_results_plots/%s.pdf' % (filename), bbox_inches='tight')
        #plt.show()

with open('./output/classifier_2d_opt_param_values.json', 'w') as f:
    json.dump(opt_vals, f, indent=2)
