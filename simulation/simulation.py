from operator import mul
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("dark_grid")

def p_kth_neg_given_all_neg(x: np.float64, k: int, p: int) -> np.float64:
    xp = x * p # number of true cases
    return reduce(mul, ((p - xp - i - 1)/(p - i - 1) for i in range(k)), 1)

def prob_zero_positive_results(x: np.ndarray, y: int, p: int = 100) -> np.ndarray:
    return 

fig, ax = plt.subplot()
X = np.linspace(0, 1)     # prevalence
Ys =  np.array([1, 2, 5, 10, 25, 50])/100 # testing rate
