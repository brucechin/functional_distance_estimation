from cProfile import label
from re import S
import numpy as np 
import time
from scipy.stats import unitary_group
import scipy.linalg as la
from scipy.stats import unitary_group
import math
import matplotlib.pyplot as plt

x = np.linspace(-4,4,100)
y_exp = np.exp(x)
y_exp_minus = np.exp(-x)
y_cosh = np.cosh(x)
y_sinh = np.sinh(x)

ticks_size = 25
plt.plot(x, y_exp, label='exp(x)')
plt.plot(x, y_exp_minus, label='exp(-x)')
plt.plot(x, y_cosh, label='cosh(x)')
plt.plot(x, y_sinh, label='sinh(x)')
plt.xlabel("x", fontsize= ticks_size)
plt.ylabel("y", fontsize= ticks_size)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.ylim([-40,60])
plt.legend(loc='best')
plt.legend(loc="best", fontsize=20)
plt.savefig("functions_figure.pdf", dpi=400, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
plt.show()
