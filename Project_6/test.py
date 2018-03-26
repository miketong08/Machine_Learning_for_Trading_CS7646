
from BestPossibleStrategy import bps
from marketsim import compute_portvals

import pandas as pd
import os
os.chdir('../')
from util import get_data, plot_data
os.chdir('./Project_6')

s = bps()
print(s)
#input()
p = compute_portvals(s)
print(p)
