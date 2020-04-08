import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path 

nyt = pd.read_csv(nyt_path)

nyt = nyt.set_index('date').sort_values('date')

# GAWC data:
# https://www.lboro.ac.uk/gawc/world2018link.html

# sigmoid curves
# https://en.wikipedia.org/wiki/Gompertz_function
# https://gist.github.com/andrewgiessel/5684769