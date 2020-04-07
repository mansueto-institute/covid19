import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path 

nyt = pd.read_csv(nyt_path)

nyt = nyt.set_index('date').sort_values('date')