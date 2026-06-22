from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def statistical_test(data, alg_names, pv_friedman=0.05, pv_wilcoxon=0.05):
    data = np.asarray(data)
    
    if data.shape[0] != len(alg_names):
        raise Exception('data and alg_names rows must have the same length')

    # 1. FRIEDMAN TEST
    if pv_friedman is not None:
        _, pv_f = friedmanchisquare(*[data[i, :] for i in range(data.shape[0])])
        
        if pv_f > pv_friedman:
            # Não há diferença estatística global, retorna apenas o p-value do Friedman
            return np.round(pv_f, 6)

    # 2. WILCOXON TEST WITH HOLM-BONFERRONI
    if pv_wilcoxon is not None:
        combs = list(combinations(range(data.shape[0]), 2))
        
        # Calcula os p-values brutos
        raw_pvals = [wilcoxon(data[c[0], :], data[c[1], :])[1] for c in combs]
            
        # Aplica a correção de Holm-Bonferroni
        _, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=pv_wilcoxon, method='holm')
        
        # Mapeia os p-values corrigidos para seus respectivos pares e retorna
        results = {
            f"{alg_names[c[0]]}_vs_{alg_names[c[1]]}": np.round(pvals_corrected[i], 6)
            for i, c in enumerate(combs)
        }
        
        return results


class AVGMetrics(object):
    """
    This is a simple class to control the average for a given value. It's useful to control loss and accuracy for a
    mini-batch during the training phase. Essentially, it keeps track of a given value and compute the average when the
    __call__ method is called.
    """

    def __init__(self):
        """
        It starts the method's attributes
        """
        self.sum_value = 0
        self.avg = 0
        self.count = 0
        self.values = list()

    def __call__(self):
        """
        It returns the average when the method is called
        """
        return self.avg

    def std(self):
        """
            It returns the std
        """
        if len(self.values) > 0:
            return np.std(self.values)
        else:
            return 0

    def update(self, val):
        """
        Updates the attributes according to the given value
        """
        self.sum_value += val
        self.count += 1
        self.avg = self.sum_value / float(self.count)
        self.values.append(val)

    def print(self, title='Summary'):
        """
        It prints the class' attributes
        """
        print("-" * 50)
        print(f"- {title}")
        print("-" * 50)
        print('- MEDIAN: {:.3f}'.format(np.median(self.values)))
        print('- AVG: {:.3f}'.format(self.avg))
        print('- STD: {:.3f}'.format(np.std(self.values)))
        print('- MAX: {:.3f}'.format(max(self.values)))
        print('- MIN: {:.3f}'.format(min(self.values)))