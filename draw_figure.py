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


init_time_exp=[1.410215139389038, 1.546872615814209, 2.4196555614471436, 3.328195571899414, 5.03579306602478, 9.929456233978271, 30.837093830108643]
query_all_time_exp=[0.4123673677444458, 0.4163767576217651, 0.46226890087127687, 0.46499674320220946, 0.492268443107605, 0.5196513652801513, 0.634795355796814]
query_one_time_exp=[0.09827971458435059, 0.09829163551330566, 0.16924762725830078, 0.20717930793762207, 0.31299614906311035, 0.47512078285217285, 1.5035526752471924]
query_pair_time_exp=[0.0454249382019043, 0.04500889778137207, 0.07750511169433594, 0.07821774482727051, 0.08260989189147949, 0.0876321792602539, 0.10349678993225098]
memory_consumption_exp=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_exp=[0.6797326679180073, 0.7489687942635819, 0.8445414633135297, 0.8787472792702673, 0.9234339659765861, 0.9194825461512539, 0.9693236318110707]

accuracy_diff_D_exp=[0.4001010963348568, 0.7611534117398406, 0.9291826788666716, 0.9994538516524958, 0.9999999859844313, 0.9999999999999999]


init_time_cosh=[1.482191801071167, 1.595693826675415, 2.4987170696258545, 3.253356456756592, 4.976968050003052, 10.36504578590393, 31.448013067245483]
query_all_time_cosh=[0.4094757318496704, 0.4168199300765991, 0.4626606941223145, 0.46924190521240233, 0.496009373664856, 0.5204360961914063, 0.6517076015472412]
query_one_time_cosh=[0.10057187080383301, 0.09716176986694336, 0.17304754257202148, 0.20639538764953613, 0.30405712127685547, 0.47381138801574707, 1.5033464431762695]
query_pair_time_cosh=[0.04620361328125, 0.04593253135681152, 0.07864141464233398, 0.07856082916259766, 0.083465576171875, 0.08737492561340332, 0.10329318046569824]
memory_consumption_cosh=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_cosh=[0.6646812462625218, 0.7694247272113595, 0.8298702930167182, 0.8615900575743942, 0.9208782385335889, 0.9243355559846357, 0.9669113141414258]



accuracy_diff_D_cosh=[0.6876664451775936, 0.9763297165084222, 0.9991734662964744, 0.9999999980353798, 0.9999999999999999, 0.9999999999999999]



init_time_sinh=[1.4792556762695312, 1.602454662322998, 2.473588228225708, 3.322308301925659, 4.993152856826782, 10.527872800827026, 31.433018922805786]
query_all_time_sinh=[0.4248421430587769, 0.42919578552246096, 0.474113917350769, 0.47925567626953125, 0.5056984186172485, 0.534268593788147, 0.6697338342666626]
query_one_time_sinh=[0.10335755348205566, 0.10453939437866211, 0.173170804977417, 0.20807290077209473, 0.3087117671966553, 0.4868438243865967, 1.5221765041351318]
query_pair_time_sinh=[0.0464630126953125, 0.046538591384887695, 0.07992696762084961, 0.07932114601135254, 0.08460831642150879, 0.08888554573059082, 0.10391449928283691]
memory_consumption_sinh=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_sinh=[0.6941333078301195, 0.7739229212570067, 0.8460486148356292, 0.8516090702721072, 0.9219618013961846, 0.9379476633968368, 0.9670745077853529]


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
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
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
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.plot(x, memory_consumption_exp)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()
plt.plot(x, accuracy_diff_sketch_size_exp)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
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
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
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
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, memory_consumption_cosh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, accuracy_diff_sketch_size_cosh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
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
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
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
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, memory_consumption_sinh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, accuracy_diff_sketch_size_sinh)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
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





init_time_power1=[1.4191477298736572, 1.4459872245788574, 2.130824089050293, 3.308781147003174, 5.055497169494629, 9.97172999382019, 31.094226598739624]
query_all_time_power1=[0.5778135776519775, 0.5813371181488037, 0.5840460538864136, 0.6305680751800538, 0.6691162586212158, 0.6927803039550782, 0.8412427425384521]
query_one_time_power1=[0.12002253532409668, 0.11449408531188965, 0.13846874237060547, 0.24163150787353516, 0.3421027660369873, 0.5000905990600586, 1.5464198589324951]
query_pair_time_power1=[0.06289792060852051, 0.06149005889892579, 0.06233096122741699, 0.10250592231750488, 0.10549306869506836, 0.10869193077087402, 0.1239495277404785]
memory_consumption_power1=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_power1=[0.5791973051464681, 0.6406941895307331, 0.6528224082377293, 0.730016794252805, 0.6842835208810507, 0.7088161040385181, 0.6832213617950174]
accuracy_diff_sketch_size_power2=[0.6957533507913103, 0.783551014934073, 0.8496277875721896, 0.8840501900826527, 0.9215616335965984, 0.9401591159279665, 0.9449981497540033]
accuracy_diff_sketch_size_power3=[0.11127939012665888, 0.16024739897541607, 0.18609067785421285, 0.1508460069838118, 0.15904259478416827, 0.17367304808624773, 0.16885548152914998]
# accuracy_diff_sketch_size_power2=[0.7024335562578536, 0.7752377277632126, 0.8442467955357094, 0.8833619144254536, 0.9241383190240913]
# accuracy_diff_sketch_size_power3=[0.14295580242126582, 0.13493456103646317, 0.15830072647364146, 0.17276530498560572, 0.15982977034749435]

accuracy_diff_D_power1=[0.08305029257459395, 0.24930104860194158, 0.46675519693735523, 0.9234066217725369, 0.9997878787762959, 0.9999999999994392]
accuracy_diff_D_power2=[0.8143604942800233, 0.976555374002664, 0.9978145469489041, 0.9999995485754958, 0.9999999999999836, 0.9999999999999998]
accuracy_diff_D_power3=[0.004392816270307964, 0.023908441370678157, 0.073317530873659, 0.46343044608603834, 0.9681307278575743, 0.9999999310855812]



#f=power1(x)
plt.figure()

x = np.linspace(0, len(init_time_power1), len(init_time_power1))
plt.plot(x, init_time_power1, label='Init time')
plt.plot(x, query_all_time_power1, label='QueryAll time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (millisecond)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_queryall_time_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, query_one_time_power1, label='QueryOne time')
plt.plot(x, query_pair_time_power1, label='QueryPair time')
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryone_pair_time_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, memory_consumption_power1)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.plot(x, accuracy_diff_sketch_size_power3, label="$\Lambda \in (0,0.2)$")
plt.plot(x, accuracy_diff_sketch_size_power1, label="$\Lambda \in (0,0.1)$")
plt.plot(x, accuracy_diff_sketch_size_power2, label="$\Lambda \in (0,0.01)$")

plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(loc='best', fontsize=12)
plt.savefig("accuracy_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()



x = np.linspace(0, len(accuracy_diff_D_power1), len(accuracy_diff_D_power1))
plt.figure()
plt.plot(x, accuracy_diff_D_power3, label="$\Lambda \in (0,0.2)$")
plt.plot(x, accuracy_diff_D_power1, label="$\Lambda \in (0,0.1)$")
plt.plot(x, accuracy_diff_D_power2, label="$\Lambda \in (0,0.01)$")
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(loc='best', fontsize=12)
plt.savefig("accuracy_truncation_degree_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()