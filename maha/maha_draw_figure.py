from cProfile import label
from re import S
import numpy as np 
import time
from scipy.stats import unitary_group
import scipy.linalg as la
from scipy.stats import unitary_group
import math
import matplotlib.pyplot as plt
import random

init_time_exp=[0.7528359889984131, 0.8881633281707764, 1.3749914169311523, 3.5237818559010825, 10.303679466247559, 25.987924973169964, 53.619096755981445, 108.48542165756226, 214.68600312868753]
query_all_time_exp=[0.49235694408416747, 0.5984905004501343, 0.6428071737289429, 0.6225868225097656, 0.6517486333847046, 0.6709957838058471, 0.6736599683761597, 0.7794945240020752, 0.8209025859832764]
query_one_time_exp=[0.12860965728759766, 0.14516282081604004, 0.1766960620880127, 0.41922569274902344, 1.0834941864013672, 2.689610242843628, 5.132524251937866, 10.899214744567871, 21.473173141479492]
memory_consumption_exp=[830.67392, 838.86592, 855.24992, 888.01792, 953.55392, 1084.62592, 1346.76992, 1871.05792, 2919.63392]
accuracy_diff_sketch_size_exp=[0.8052706530542195, 0.8455730276625495, 0.8750144181664794, 0.8838606766557691, 0.9013041027801764, 0.8958703234813509, 0.9009702347958827, 0.9011129946376499, 0.8992997583866558]
accuracy_diff_sketch_size_exp_std_err=[0.004308159801575043, 0.010690348677291902, 0.010936278023653325, 0.009174215583213135, 0.00461793037671468, 0.006024801415385588, 0.005129308832336377, 0.0034170895234244287, 0.002319093284757273]
query_pair_time_exp_std_err=[7.551345356340995e-05, 5.7247721614733275e-05, 0.00010270766888593803, 0.00012220760537288804, 0.0005059629635806133, 6.981472476232248e-05, 0.002946232303720777, 0.001243201460993713, 0.0003340731475665959]
query_all_time_exp_std_err=[0.02573063455140597, 0.007218726618552555, 0.023504200126485367, 0.003465999302752975, 0.0036011289569682567, 0.0203828099661177, 0.0072895285255078525, 0.03044724869934359, 0.004535243263683647]
init_time_exp_std_err=[0.08548904737164241, 0.008742710222395448, 0.135414021488782, 0.10309578117464055, 0.04721583161657926, 0.17382332358061142, 0.3280165463846562, 1.910034054708835, 2.7252720978551275]
query_one_time_exp_std_err=[0.0012133978578043152, 0.00024300409026633006, 0.014771775410837287, 0.009304971657515653, 0.005482210478232035, 0.00689352277795223, 0.01493124301644988, 0.04955810210911713, 0.03508016393914807]
ticks_size = 25

init_time_exp=[0.6725095907847086, 0.977027972539266, 1.2478716373443604, 3.7324185371398926, 11.615556478500366, 27.463221232096355, 57.925594329833984, 113.19341166814168, 222.25753132502237]
query_all_time_exp=[1.051043701171875, 1.5231117725372314, 2.018546152114868, 4.6723803043365475, 12.633266639709472, 27.27038025856018, 57.312576627731325, 111.07617621421814, 214.42619795799254]
memory_consumption_exp=[830.67392, 838.86592, 855.24992, 888.01792, 953.55392, 1084.62592, 1346.76992, 1871.05792, 2919.63392]

#f=exp(x)
x = np.linspace(0, len(init_time_exp), len(init_time_exp))
plt.figure()
plt.errorbar(x, init_time_exp, yerr=init_time_exp_std_err, label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,640,1280,2560], fontsize= ticks_size, rotation=45)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.errorbar(x, query_all_time_exp, yerr=query_all_time_exp_std_err, label='QueryAll time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,640,1280,2560], fontsize= ticks_size, rotation=45)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()
plt.errorbar(x, query_one_time_exp, yerr=query_one_time_exp_std_err, label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,640,1280,2560], fontsize= ticks_size, rotation=45)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.errorbar(x, query_pair_time_exp, yerr=query_pair_time_exp_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,640,1280,2560], fontsize= ticks_size, rotation=45)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()


plt.figure()
plt.errorbar(x, memory_consumption_exp, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,640,1280,2560], fontsize= ticks_size, rotation=45)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()
plt.errorbar(x, accuracy_diff_sketch_size_exp, yerr=accuracy_diff_sketch_size_exp_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,640,1280,2560], fontsize= ticks_size, rotation=45)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()