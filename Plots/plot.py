import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



df1 = pd.read_csv('plot_dim.csv')


axi = df1.plot(x = 'dimension', legend = True)

plt.show()