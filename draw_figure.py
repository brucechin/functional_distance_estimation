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



ticks_size = 20

init_time_exp=[1.5579235553741455, 1.8555331230163574, 2.554344415664673, 3.569812059402466, 5.658602237701416]
query_all_time_exp=[0.4088250398635864, 0.41566147804260256, 0.4614201307296753, 0.465824031829834, 0.49845614433288576]
query_one_time_exp=[0.00010336613655090332, 0.00010600519180297851, 0.00017500638961791992, 0.0002300267219543457, 0.00032434535026550293]
query_pair_time_exp=[4.635047912597656e-05, 4.677700996398926e-05, 7.906031608581543e-05, 7.952761650085449e-05, 8.57393741607666e-05]
memory_consumption_exp=[336.128, 348.928, 374.528, 425.728, 528.128]
accuracy_diff_sketch_size_exp=[0.688970772176638, 0.761944308384785, 0.8466606908290908, 0.8945640846501287, 0.9243982894761898]
accuracy_diff_D_exp=[0.9512294245007139, 0.9887908957257496, 0.9999997497860527, 0.9999999975020486, 0.9999999999792085, 0.9999999999999998]



init_time_cosh=[1.5756340026855469, 1.8174355030059814, 2.3492443561553955, 3.6615216732025146, 5.724668264389038]
query_all_time_cosh=[0.41011040210723876, 0.4128910303115845, 0.45349645614624023, 0.46285266876220704, 0.4910826444625854]
query_one_time_cosh=[0.00010557913780212403, 0.0001080167293548584, 0.00017009210586547852, 0.0002380661964416504, 0.0003697128295898438]
query_pair_time_cosh=[4.622817039489746e-05, 4.6043872833251956e-05, 7.880616188049316e-05, 7.907795906066895e-05, 8.386111259460449e-05]
memory_consumption_cosh=[336.128, 348.928, 374.528, 425.728, 528.128]
accuracy_diff_sketch_size_cosh=[0.6545013927331245, 0.5889090592339886, 0.8153602515739659, 0.8917038877100483, 0.919278838955871]
accuracy_diff_D_cosh=[0.9684208863712137, 0.9899742013220819, 0.9999999743816406, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998]

init_time_sinh=[1.5765914916992188, 1.6245591640472412, 2.6000609397888184, 3.9666836261749268, 5.67213249206543]
query_all_time_sinh=[0.42246718406677247, 0.4254459857940674, 0.44707422256469725, 0.47554898262023926, 0.5000954866409302]
query_one_time_sinh=[0.00010068011283874512, 0.00010612654685974121, 0.00012352967262268066, 0.00021938920021057128, 0.0003417031764984131]
query_pair_time_sinh=[4.7362089157104495e-05, 4.749178886413574e-05, 4.848480224609375e-05, 7.967758178710938e-05, 8.525657653808593e-05]
memory_consumption_sinh=[336.128, 348.928, 374.528, 425.728, 528.128]
accuracy_diff_sketch_size_sinh=[0.6814094643327572, 0.7607099929773038, 0.8417571912079732, 0.881569747487559, 0.9178450484236921]
accuracy_diff_D_sinh=[0.9758511602425217, 0.9897942654136433, 0.9999999957386334, 0.9999999999999999, 0.9999999999999999, 0.9999999999999999]

#f=exp(x)
x = np.linspace(0, len(init_time_exp), len(init_time_exp))
plt.figure()
plt.plot(x, init_time_exp, label='Init time')
plt.plot(x, query_all_time_exp, label='QueryAll time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_queryall_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.plot(x, query_one_time_exp, label='QueryOne time')
plt.plot(x, query_pair_time_exp, label='QueryPair time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.plot(x, memory_consumption_exp)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Memory Consumption(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()
plt.plot(x, accuracy_diff_sketch_size_exp)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()



x = np.linspace(0, len(accuracy_diff_D_exp), len(accuracy_diff_D_exp))
plt.figure()
plt.plot(x, accuracy_diff_D_exp)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_truncation_degree_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()







#f=cosh(x)
plt.figure()

x = np.linspace(0, len(init_time_cosh), len(init_time_cosh))
plt.plot(x, init_time_cosh, label='Init time')
plt.plot(x, query_all_time_cosh, label='QueryAll time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_queryall_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, query_one_time_cosh, label='QueryOne time')
plt.plot(x, query_pair_time_cosh, label='QueryPair time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, memory_consumption_cosh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Memory Consumption(KB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, accuracy_diff_sketch_size_cosh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()



x = np.linspace(0, len(accuracy_diff_D_cosh), len(accuracy_diff_D_cosh))
plt.figure()

plt.plot(x, accuracy_diff_D_cosh)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_truncation_degree_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()




#f=sinh(x)
plt.figure()

x = np.linspace(0, len(init_time_sinh), len(init_time_sinh))
plt.plot(x, init_time_sinh, label='Init time')
plt.plot(x, query_all_time_sinh, label='QueryAll time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_queryall_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, query_one_time_sinh, label='QueryOne time')
plt.plot(x, query_pair_time_sinh, label='QueryPair time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, memory_consumption_sinh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Memory Consumption(KB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, accuracy_diff_sketch_size_sinh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()



x = np.linspace(0, len(accuracy_diff_D_sinh), len(accuracy_diff_D_sinh))
plt.figure()

plt.plot(x, accuracy_diff_D_sinh)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_truncation_degree_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()