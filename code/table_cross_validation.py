
import numpy as np
import collections
import json
import textwrap
import scipy.stats

def mean_interval(data):
    lw, hi = scipy.stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=scipy.stats.sem(data))
    return hi - np.mean(data)

def print_table(target):
    with open('./output/cross_validation_%s.json' % target) as fd:
        scores = json.load(fd, object_pairs_hook=collections.OrderedDict)

        output = textwrap.dedent("""\
        \\begin{table}[H]
        \\centering
        \\begin{tabular}{r|c|c}
        model & TIMIT & ELSDSR \\\\ \\hline
        """)

        for model, datasets in scores.items():
            timit = datasets['timit']
            elsdsr = datasets['elsdsr']

            if isinstance(timit, float) and isinstance(elsdsr, float):
                output += '%28s & $%.3f$ & $%.3f$ \\\\\n' % (
                    model, timit, elsdsr
                )
            else:
                output += '%28s & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ \\\\\n' % (
                    model,
                    np.mean(timit), mean_interval(timit),
                    np.mean(elsdsr), mean_interval(elsdsr)
                )

        output += textwrap.dedent("""\
        \\end{tabular}
        \\caption{misclassification rate for %s classification with $95\\%%$ confidence internal}
        \\end{table}
        """ % target)

        return output

print(print_table('sex'))
print(print_table('speaker'))
