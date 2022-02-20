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

init_time_exp=[1.509676218032837, 1.7528493404388428, 3.2664709091186523, 7.131795883178711, 13.135652303695679]
query_all_time_exp=[0.4250762414932251, 0.4457832908630371, 0.47269792556762696, 0.5480687856674195, 0.5435007333755493]
query_one_time_exp=[0.10153484344482422, 0.0969245433807373, 0.1246764659881592, 0.21471452713012695, 0.2976698875427246]
query_pair_time_exp=[0.045194149017333984, 0.04538607597351074, 0.04587721824645996, 0.07840847969055176, 0.08332967758178711]
memory_consumption_exp=[336.128, 348.928, 374.528, 425.728, 528.128]
accuracy_diff_sketch_size_exp=[0.6879636223396864, 0.7340928877440591, 0.8462597073660129, 0.8715688548026141, 0.9248172349567643]
accuracy_diff_D_exp=[0.4001010963348568, 0.7611534117398406, 0.9291826788666716, 0.9994538516524958, 0.9999999859844313, 0.9999999999999999]


init_time_cosh=[2.981541395187378, 2.670764207839966, 4.253162145614624, 7.076452732086182, 14.644951343536377]
query_all_time_cosh=[0.4098921298980713, 0.4182533502578735, 0.5327364206314087, 0.5500081062316895, 0.5610482931137085]
query_one_time_cosh=[0.10489892959594727, 0.10122990608215332, 0.17502856254577637, 0.2172069549560547, 0.3249983787536621]
query_pair_time_cosh=[0.046816349029541016, 0.04538249969482422, 0.07944536209106445, 0.0797569751739502, 0.08404159545898438]
memory_consumption_cosh=[336.128, 348.928, 374.528, 425.728, 528.128]
accuracy_diff_sketch_size_cosh=[0.6428286154063269, 0.7822092584742552, 0.838677271793338, 0.8911460362044472, 0.9252578676558838]
accuracy_diff_D_cosh=[0.6876664451775936, 0.9763297165084222, 0.9991734662964744, 0.9999999980353798, 0.9999999999999999, 0.9999999999999999]


init_time_sinh=[1.48240065574646, 1.5092816352844238, 2.502527952194214, 3.1796116828918457, 5.157666206359863]
query_all_time_sinh=[0.418946361541748, 0.4176386833190918, 0.46564426422119143, 0.46858160495758056, 0.49361453056335447]
query_one_time_sinh=[0.10225629806518555, 0.09775710105895996, 0.17929744720458984, 0.21870970726013184, 0.34084248542785645]
query_pair_time_sinh=[0.048087120056152344, 0.046689510345458984, 0.08057165145874023, 0.08080673217773438, 0.08580350875854492]
memory_consumption_sinh=[336.128, 348.928, 374.528, 425.728, 528.128]
accuracy_diff_sketch_size_sinh=[0.6706986676950437, 0.7519716629535719, 0.8474794882710364, 0.8929117929036826, 0.9217607719023378]
accuracy_diff_D_sinh=[0.8624003618103883, 0.9931348752909338, 0.9998273914485457, 0.9999999997601088, 0.9999999999999998, 0.9999999999999998]


tilde_fA_f_norm_error=[2.3201169227365472, 1.5831009295895941, 0.6918606356680781, 0.22351568624471355, 0.0572053027585779, 0.012106850803676862, 0.0021823169015375015, 0.00034236480979433657, 4.7525107777637e-05, 5.914009594433011e-06, 6.667220481402045e-07]
tilde_fA_spetral_norm_error=[2.3201169227365472, 1.120116922736547, 0.4001169227365473, 0.1121169227365475, 0.025716922736547687, 0.0049809227365478215, 0.000833722736548026, 0.00012277416511929928, 1.613187940519012e-05, 1.912907976464595e-06, 2.0663140487542364e-07]

#show the F norm error between tilde{f}(A) and f(A) matrix
x = np.linspace(0, len(tilde_fA_f_norm_error), len(tilde_fA_f_norm_error))
plt.figure()
plt.plot(x, tilde_fA_f_norm_error, label = "Frobenius Norm")
plt.plot(x, tilde_fA_spetral_norm_error, label = "Spectral Norm")
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Approximation Error", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,4,5,6,7,8,9,10], fontsize= ticks_size)
plt.ylim([-0.2 , 3])
plt.legend(loc='upper right', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("tilde_fA_approximation_error.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')



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
plt.ylabel("Time (millisecond)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.plot(x, memory_consumption_exp)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
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
plt.ylabel("Time (millisecond)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, memory_consumption_cosh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(KB)", fontsize= ticks_size)
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
plt.ylabel("Time (millisecond)", fontsize= ticks_size)
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
plt.ylabel("Mem Usage(KB)", fontsize= ticks_size)
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