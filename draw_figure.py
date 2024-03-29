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



ticks_size = 25


init_time_exp=[1.410215139389038, 1.546872615814209, 2.4196555614471436, 3.328195571899414, 5.03579306602478, 9.929456233978271, 30.837093830108643]
query_all_time_exp=[0.4123673677444458, 0.4163767576217651, 0.46226890087127687, 0.46499674320220946, 0.492268443107605, 0.5196513652801513, 0.634795355796814]
query_one_time_exp=[0.09827971458435059, 0.09829163551330566, 0.16924762725830078, 0.20717930793762207, 0.31299614906311035, 0.47512078285217285, 1.5035526752471924]
query_pair_time_exp=[0.0454249382019043, 0.04500889778137207, 0.07750511169433594, 0.07821774482727051, 0.08260989189147949, 0.0876321792602539, 0.10349678993225098]
memory_consumption_exp=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_exp=[0.6611766767618003, 0.7586692279512598, 0.8334962705580237, 0.8656859580909687, 0.9078250643206239, 0.8932873401611096, 0.9680207153858289]

accuracy_diff_sketch_size_exp_std_err=[0.00771840711434933, 0.004559498521384767, 0.00799416296015209, 0.0063006158088508455, 0.0038798155295495136, 0.0033231126496598897, 0.004063488811372532]
query_pair_time_exp_std_err=[0.0021376682634929446, 0.004571937688889734, 0.0034560170893773567, 0.0008746755298272982, 0.001002712014198947, 0.0012907610649404817, 0.0007347387732332107]
query_all_time_exp_std_err=[0.010232700311783692, 0.007383137852368174, 0.003339988192638718, 0.007252977986170304, 0.016029601197249586, 0.007438071632251022, 0.0057168583479240784]
init_time_exp_std_err=[0.013243012342047002, 0.01962872812032523, 0.009816846868369823, 0.03197181553565051, 0.1074035986009036, 0.16255507320261478, 0.7294346650932081]

init_time_exp_diff_D=[1.25457763671875, 2.5642600059509277, 3.7348780632019043, 4.947084426879883, 9.246024131774902, 19.898767709732056, 40.157081604003906]
query_all_time_exp_diff_D=[0.18727462291717528, 0.2954710006713867, 0.40419440269470214, 0.5089008331298828, 0.7173526048660278, 1.2451520442962647, 2.2919521570205688]
query_one_time_exp_diff_D=[0.06184768676757813, 0.1516885757446289, 0.233931303024292, 0.30304884910583496, 0.5235013961791992, 1.0624721050262451, 2.262873649597168]
query_pair_time_exp_diff_D=[0.028455018997192383, 0.04754638671875, 0.06704330444335938, 0.0850071907043457, 0.1111001968383789, 0.1674213409423828, 0.27608537673950195]
memory_consumption_exp_diff_D=[374.432, 425.664, 476.896, 528.128, 630.592, 886.752, 1399.072]
accuracy_diff_D_exp=[0.27457081095614655, 0.6041601605944237, 0.9126694363901919, 0.9200680917771333, 0.9201172246838839, 0.9243788068860953, 0.9240002034141377]


init_time_exp_diff_D_std_err=[0.018934024513683544, 0.052094936327971436, 0.09699651747789202, 0.1791825216012653, 0.2047147279971211, 0.34226387145877285, 0.3887541891057944]
query_all_time_exp_diff_D_std_err=[0.011333689424210217, 0.009266476799047055, 0.014989747716320069, 0.014994668105782039, 0.0254173316469101, 0.028348059437561968, 0.046862485279624416]
query_pair_time_exp_diff_D_std_err=[6.395815220986705e-05, 0.003694276275718136, 0.0024909145086247676, 0.00017613451458486112, 0.00045174343901486667, 0.0008749003433207955, 0.0007541359080092654]
accuracy_diff_D_exp_std_err=[0.003169704928755899, 0.007258903037738743, 0.007909529878143631, 0.004189327671666965, 0.0011543286506706892, 0.0032901418892565316, 0.001010459612858862]
##############################################################################################################################################################################
init_time_cosh=[1.482191801071167, 1.595693826675415, 2.4987170696258545, 3.253356456756592, 4.976968050003052, 10.36504578590393, 31.448013067245483]
query_all_time_cosh=[0.4094757318496704, 0.4168199300765991, 0.4626606941223145, 0.46924190521240233, 0.496009373664856, 0.5204360961914063, 0.6517076015472412]
query_one_time_cosh=[0.10057187080383301, 0.09716176986694336, 0.17304754257202148, 0.20639538764953613, 0.30405712127685547, 0.47381138801574707, 1.5033464431762695]
query_pair_time_cosh=[0.04620361328125, 0.04593253135681152, 0.07864141464233398, 0.07856082916259766, 0.083465576171875, 0.08737492561340332, 0.10329318046569824]
memory_consumption_cosh=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_cosh=[0.6646812462625218, 0.7694247272113595, 0.8298702930167182, 0.8615900575743942, 0.9208782385335889, 0.9243355559846357, 0.9669113141414258]

accuracy_diff_sketch_size_cosh_std_err=[0.00435265051666124, 0.0008340809413878571, 0.0028464211069144157, 0.0016145065247099588, 0.002222985088205128, 0.0006834285791605079, 0.0007974012294081113]
query_pair_time_cosh_std_err=[0.00652151687574898, 0.003248905709964683, 0.001605613071338512, 0.00044278724885736726, 0.004566122095241302, 0.0010817850344863613, 0.00118119004004322]
query_all_time_cosh_std_err=[0.016089782245772156, 0.007870858572711626, 0.003721021808641634, 0.00794042475818161, 0.017487797620515163, 0.009791008311505941, 0.006599556593972505]
init_time_cosh_std_err=[0.014508881455558353, 0.012894932110387522, 0.053097715959851, 0.051418331166380754, 0.1587455787712245, 0.163671695057087, 1.3406382861981792]

init_time_cosh_diff_D=[1.6174461841583252, 3.241597890853882, 4.728921413421631, 6.259262561798096, 8.889387130737305, 19.441669702529907, 38.89839005470276]
query_all_time_cosh_diff_D=[0.18285346031188965, 0.2885807752609253, 0.39251294136047366, 0.4942338943481445, 0.68833327293396, 1.2160800218582153, 2.3524981260299684]
query_one_time_cosh_diff_D=[0.06141376495361329, 0.1798534393310547, 0.2854635715484619, 0.37432098388671875, 0.5589690208435059, 0.9855766296386719, 2.198000431060791]
query_pair_time_cosh_diff_D=[0.028178930282592773, 0.04703068733215332, 0.066314697265625, 0.08400440216064453, 0.10943865776062012, 0.16537690162658691, 0.28284549713134766]
memory_consumption_cosh_diff_D=[374.432, 425.664, 476.896, 528.128, 630.592, 886.752, 1399.072]
accuracy_diff_D_cosh=[0.6583200427122602, 0.9220585423933717, 0.9209420990061956, 0.9031585430019385, 0.913083124954452, 0.8819865864793275, 0.9064279488360844]

init_time_cosh_diff_D_std_err=[0.029002317493278734, 0.010108891199519883, 0.03878368861320395, 0.050454510201030436, 0.09467839540663567, 0.22817371849957993, 0.9330596583229571]
query_all_time_cosh_diff_D_std_err=[0.009856634995060867, 0.011875053094895296, 0.009471170548543482, 0.013488661242593555, 0.02115979734685697, 0.02952281121686907, 0.04680017157180805]
query_pair_time_cosh_diff_D_std_err=[0.0006929572092972658, 0.007084864060843182, 0.003334770515240801, 0.0018688263983544552, 0.0013743697307438798, 0.0021751723158360933, 0.0018630910251553265]
accuracy_diff_D_cosh_std_err=[0.006697944555151401, 0.0018935158742517813, 0.0023465847841538548, 0.0006090674682631228, 0.0017194840298897225, 0.001093314374601104, 0.004217997208136523]

##############################################################################################################################################################################


init_time_sinh=[1.4792556762695312, 1.602454662322998, 2.473588228225708, 3.322308301925659, 4.993152856826782, 10.527872800827026, 31.433018922805786]
query_all_time_sinh=[0.4248421430587769, 0.42919578552246096, 0.474113917350769, 0.47925567626953125, 0.5056984186172485, 0.534268593788147, 0.6697338342666626]
query_one_time_sinh=[0.10335755348205566, 0.10453939437866211, 0.173170804977417, 0.20807290077209473, 0.3087117671966553, 0.4868438243865967, 1.5221765041351318]
query_pair_time_sinh=[0.0464630126953125, 0.046538591384887695, 0.07992696762084961, 0.07932114601135254, 0.08460831642150879, 0.08888554573059082, 0.10391449928283691]
memory_consumption_sinh=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_sinh=[0.6778750189997567, 0.7741223988773899, 0.8432876765724711, 0.8593286775141528, 0.9185344349691575, 0.9321856702627598, 0.965606929100662]

accuracy_diff_sketch_size_sinh_std_err=[0.00902289925736922, 0.0041999778903294045, 0.006152036799160729, 0.0011466530886642208, 0.002897924097037175, 0.0009725063233218982, 0.0004390041874005013]
query_pair_time_sinh_std_err=[0.0023594949520971075, 0.0035365145867940454, 0.004678679721152799, 0.0004170928666885277, 0.0013509408345835142, 0.0006538534816622761, 0.0011316359756247249]
query_all_time_sinh_std_err=[0.013760505723061315, 0.007337323220509087, 0.0062139758811042955, 0.008039976405645397, 0.010121399339708754, 0.006720755034109107, 0.009188328643407271]
init_time_sinh_std_err=[0.03167464419588432, 0.004268005373146701, 0.01624833451373289, 0.034902870627066075, 0.11221047685464099, 0.10650340134147607, 0.7909938459125406]

init_time_sinh_diff_D=[1.5107817649841309, 2.484938621520996, 3.6695003509521484, 4.851715087890625, 9.044080257415771, 17.637880563735962, 37.066893100738525]
query_all_time_sinh_diff_D=[0.18690090179443358, 0.29481399059295654, 0.400160026550293, 0.5028498888015747, 0.7169574975967408, 1.252603244781494, 2.415850067138672]
query_one_time_sinh_diff_D=[0.08091974258422852, 0.14791226387023926, 0.23009419441223145, 0.30007338523864746, 0.4454679489135742, 0.9483966827392578, 2.0454349517822266]
query_pair_time_sinh_diff_D=[0.028562068939208984, 0.04782390594482422, 0.06629490852355957, 0.0840451717376709, 0.11121153831481934, 0.17101168632507324, 0.28910303115844727]
memory_consumption_sinh_diff_D=[374.432, 425.664, 476.896, 528.128, 630.592, 886.752, 1399.072]
accuracy_diff_D_sinh=[0.7919341511980902, 0.8817835376986091, 0.9217835376986091, 0.9098102654325744, 0.921508276339948, 0.9187847189813966, 0.9189638972911025]

init_time_sinh_diff_D_std_err=[0.02038689121302051, 0.023135602555443287, 0.0704047158563694, 0.11751758849201982, 0.144495245184104, 0.13072337745966225, 0.5861305156024875]
query_all_time_sinh_diff_D_std_err=[0.005778207815518487, 0.015083976102150346, 0.008597917725911501, 0.023148369476305553, 0.028208741718599512, 0.02650856184257023, 0.06261886837316266]
query_pair_time_sinh_diff_D_std_err=[0.0133803605184138, 0.01277239435773798, 0.013539889007546256, 0.012552638970152295, 0.01091408805325089, 0.013711396519767787, 0.013246173724190732]
accuracy_diff_D_sinh_std_err=[0.011384123009783646, 0.0020796317568473415, 0.0019172813690843788, 0.0018989513648527898, 0.0038482730848103204, 0.0008755772943197091, 0.0001873687860992343]
##############################################################################################################################################################################


init_time_power1=[1.4191477298736572, 1.4459872245788574, 2.130824089050293, 3.308781147003174, 5.055497169494629, 9.97172999382019, 31.094226598739624]
query_all_time_power1=[0.5778135776519775, 0.5813371181488037, 0.5840460538864136, 0.6305680751800538, 0.6691162586212158, 0.6927803039550782, 0.8412427425384521]
query_one_time_power1=[0.12002253532409668, 0.11449408531188965, 0.13846874237060547, 0.24163150787353516, 0.3421027660369873, 0.5000905990600586, 1.5464198589324951]
query_pair_time_power1=[0.06289792060852051, 0.06149005889892579, 0.06233096122741699, 0.10250592231750488, 0.10549306869506836, 0.10869193077087402, 0.1239495277404785]
memory_consumption_power1=[336.128, 348.928, 374.528, 425.728, 528.128, 732.928, 1603.328]
accuracy_diff_sketch_size_power=[0.6073180635775157, 0.7955724058357101, 0.8485179591706138, 0.8929934545329127, 0.9119832909372166, 0.9143180185848976, 0.9077249777399619]


accuracy_diff_sketch_size_power_std_err=[0.004602805621630646, 0.006184159211236476, 0.006916862382879543, 0.005310588693862831, 0.003070945079309177, 0.003387401096177966, 0.004044688565126235]
query_all_time_power_std_err=[0.013292388822297715, 0.00744557811107607, 0.013034674981050643, 0.002454595069863565, 0.02197490315029705, 0.004788067028025869, 0.012232963169571572]
query_pair_time_power_std_err=[0.0020346123722532845, 0.0035007038244387464, 0.000884899923377017, 0.00045347981591848817, 0.0005866160141344987, 0.0007021182080251811, 0.0009832898601842537]
init_time_power_std_err=[0.009880869957493287, 0.009042387645986544, 0.015128263694194106, 0.03497455221700734, 0.054715749427158754, 0.06567251954001606, 0.6010367697240828]

init_time_power_diff_D=[1.4754118919372559, 2.4969794750213623, 3.6329517364501953, 4.928919553756714, 8.92063593864441, 18.45004653930664, 37.48466205596924]
query_all_time_power_diff_D=[0.22913029193878173, 0.37753987312316895, 0.5276566743850708, 0.6671140670776368, 0.9474202156066894, 1.6683682680130005, 3.0510674476623536]
query_one_time_power_diff_D=[0.07175445556640625, 0.1659250259399414, 0.2546510696411133, 0.3359041213989258, 0.5078766345977783, 1.1040480136871338, 2.1599583625793457]
query_pair_time_power_diff_D=[0.036034345626831055, 0.0627145767211914, 0.08830714225769043, 0.10663580894470215, 0.13558745384216309, 0.21086978912353516, 0.35059118270874023]
memory_consumption_power_diff_D=[374.432, 425.664, 476.896, 528.128, 630.592, 886.752, 1399.072]
accuracy_diff_D_power=[0.30235569709255194, 0.6182019360497616, 0.879603526680565, 0.9208773370762641, 0.8731339071385821, 0.9238244983898742, 0.927550641745119]


init_time_power_diff_D_std_err=[0.03822961627973265, 0.030188477426691267, 0.02148374941287568, 0.02865473335707612, 0.12787202195072714, 0.27018548946653725, 0.40706529706524086]
query_all_time_power_diff_D_std_err=[0.017626472058671044, 0.007054078767711098, 0.019064728993474302, 0.030014242676031935, 0.027068434036172786, 0.008602334445301053, 0.04147167192100692]
query_pair_time_power_diff_D_std_err=[0.0003349056090786746, 0.00765270585434599, 0.0031930862006514585, 0.0020292981679191327, 0.0022789175772343533, 0.002689531984915913, 0.009268222242178988]
accuracy_diff_D_power_std_err=[0.004848736685088483, 0.003272195855755828, 0.011637162370150689, 0.005844178766359727, 0.0014570775959282085, 0.0005518359473580411, 0.0002442215277160357]



tilde_fA_f_norm_error=[2.3201169227365472, 1.5831009295895941, 0.6918606356680781, 0.22351568624471355, 0.0572053027585779, 0.012106850803676862, 0.0021823169015375015, 0.00034236480979433657, 4.7525107777637e-05, 5.914009594433011e-06, 6.667220481402045e-07]
tilde_fA_spetral_norm_error=[2.3201169227365472, 1.120116922736547, 0.4001169227365473, 0.1121169227365475, 0.025716922736547687, 0.0049809227365478215, 0.000833722736548026, 0.00012277416511929928, 1.613187940519012e-05, 1.912907976464595e-06, 2.0663140487542364e-07]





#show the F norm error between tilde{f}(A) and f(A) matrix
x = np.linspace(0, len(tilde_fA_f_norm_error), len(tilde_fA_f_norm_error))
plt.figure()
plt.errorbar(x, tilde_fA_f_norm_error, label = "Frobenius Norm", marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.errorbar(x, tilde_fA_spetral_norm_error, label = "Spectral Norm", marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Approximation Error", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,4,5,6,7,8,9,10], fontsize= ticks_size)
plt.ylim([-0.2 , 3])
plt.legend(loc='upper right', fontsize=20)
plt.yticks(fontsize=ticks_size)
plt.savefig("tilde_fA_approximation_error.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')



#f=exp(x)
x = np.linspace(0, len(init_time_exp), len(init_time_exp))
plt.figure()
plt.errorbar(x, init_time_exp, yerr=init_time_exp_std_err, label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.errorbar(x, query_all_time_exp, yerr=query_all_time_exp_std_err, label='QueryAll time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

# plt.figure()
# plt.errorbar(x, query_one_time_exp, label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# plt.xlabel("Sketch Size", fontsize= ticks_size)
# plt.ylabel("Time (ms)", fontsize= ticks_size)
# plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.errorbar(x, query_pair_time_exp, yerr=query_pair_time_exp_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()


plt.figure()
plt.errorbar(x, memory_consumption_exp, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()
plt.errorbar(x, accuracy_diff_sketch_size_exp, yerr=accuracy_diff_sketch_size_exp_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()




x = np.linspace(0, len(accuracy_diff_D_exp), len(accuracy_diff_D_exp))
plt.figure()
plt.errorbar(x, init_time_exp_diff_D, yerr=init_time_exp_diff_D_std_err, label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_truncation_degree_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.errorbar(x, query_all_time_exp_diff_D, yerr=query_all_time_exp_diff_D_std_err, label='QueryAll time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_truncation_degree_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

# plt.figure()
# plt.errorbar(x, query_one_time_exp_diff_D, label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# plt.xlabel("Truncation Degree", fontsize= ticks_size)
# plt.ylabel("Time (ms)", fontsize= ticks_size)
# plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_truncation_degree_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()
plt.errorbar(x, query_pair_time_exp_diff_D, yerr=query_pair_time_exp_diff_D_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_truncation_degree_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()


plt.figure()
plt.errorbar(x, memory_consumption_exp_diff_D, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_truncation_degree_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

x = np.linspace(0, len(accuracy_diff_D_exp), len(accuracy_diff_D_exp))
plt.figure()
plt.errorbar(x, accuracy_diff_D_exp, yerr=accuracy_diff_D_exp_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_truncation_degree_exp.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()












#f=cosh(x)
plt.figure()

x = np.linspace(0, len(init_time_cosh), len(init_time_cosh))
plt.errorbar(x, init_time_cosh, yerr=init_time_cosh_std_err, label='Initialize time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()


plt.figure()
x = np.linspace(0, len(init_time_cosh), len(init_time_cosh))
plt.errorbar(x, query_all_time_cosh, yerr=query_all_time_cosh_std_err, label='QueryAll time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()


# plt.figure()

# plt.errorbar(x, query_one_time_cosh, label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# plt.xlabel("Sketch Size", fontsize= ticks_size)
# plt.ylabel("Time (ms)", fontsize= ticks_size)
# plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# plt.show()


plt.figure()

plt.errorbar(x, query_pair_time_cosh, yerr=query_pair_time_cosh_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

plt.errorbar(x, memory_consumption_cosh, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.errorbar(x, accuracy_diff_sketch_size_cosh, yerr=accuracy_diff_sketch_size_cosh_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()




x = np.linspace(0, len(init_time_cosh_diff_D), len(init_time_cosh_diff_D))
plt.figure()
plt.errorbar(x, init_time_cosh_diff_D,  yerr=init_time_cosh_diff_D_std_err, label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# plt.errorbar(x, query_all_time_cosh_diff_D, label='QueryAll time')
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
# plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_truncation_degree_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()
# plt.errorbar(x, init_time_cosh_diff_D, label='Init time')
plt.errorbar(x, query_all_time_cosh_diff_D, yerr=query_all_time_cosh_diff_D_std_err, label='QueryAll time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
# plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_truncation_degree_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

# plt.figure()

# plt.errorbar(x, query_one_time_cosh_diff_D,  label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# # plt.errorbar(x, query_pair_time_cosh_diff_D, label='QueryPair time')
# plt.xlabel("Truncation Degree", fontsize= ticks_size)
# plt.ylabel("Time (ms)", fontsize= ticks_size)
# plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
# # plt.legend(loc='upper left', fontsize=15)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_truncation_degree_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# # plt.show()

plt.figure()

# plt.errorbar(x, query_one_time_cosh_diff_D, label='QueryOne time')
plt.errorbar(x, query_pair_time_cosh_diff_D, query_pair_time_cosh_diff_D_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
# plt.legend(loc='upper left', fontsize=15)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_truncation_degree_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()


plt.figure()



plt.errorbar(x, memory_consumption_cosh_diff_D, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_truncation_degree_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()


x = np.linspace(0, len(accuracy_diff_D_cosh), len(accuracy_diff_D_cosh))
plt.figure()

plt.errorbar(x, accuracy_diff_D_cosh, accuracy_diff_D_cosh_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_truncation_degree_cosh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()























#f=sinh(x)
plt.figure()

x = np.linspace(0, len(init_time_sinh), len(init_time_sinh))
plt.errorbar(x, init_time_sinh, yerr=init_time_sinh_std_err, label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

x = np.linspace(0, len(init_time_sinh), len(init_time_sinh))
plt.errorbar(x, query_all_time_sinh, yerr=query_all_time_sinh_std_err, label='QueryAll', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

# plt.figure()

# plt.errorbar(x, query_one_time_sinh,   label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# # plt.errorbar(x, query_pair_time_sinh, label='QueryPair time')
# plt.xlabel("Sketch Size", fontsize= ticks_size)
# plt.ylabel("Time (sec)", fontsize= ticks_size)
# plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# # plt.show()

plt.figure()
plt.errorbar(x, query_pair_time_sinh, yerr=query_pair_time_sinh_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

plt.errorbar(x, memory_consumption_sinh, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

plt.errorbar(x, accuracy_diff_sketch_size_sinh, yerr=accuracy_diff_sketch_size_sinh_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()



x = np.linspace(0, len(accuracy_diff_D_sinh), len(accuracy_diff_D_sinh))

plt.figure()

x = np.linspace(0, len(init_time_sinh_diff_D), len(init_time_sinh_diff_D))
plt.errorbar(x, init_time_sinh_diff_D, yerr=init_time_sinh_diff_D_std_err ,label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_sketch_size_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

plt.errorbar(x, query_all_time_sinh_diff_D, yerr=query_all_time_sinh_diff_D_std_err, label='QueryAll', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_truncation_degree_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

# plt.figure()

# plt.errorbar(x, query_one_time_sinh_diff_D , label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# # plt.errorbar(x, query_pair_time_sinh, label='QueryPair time')
# plt.xlabel("Truncation Degree", fontsize= ticks_size)
# plt.ylabel("Time (sec)", fontsize= ticks_size)
# plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_truncation_degree_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# # plt.show()

plt.figure()
plt.errorbar(x, query_pair_time_sinh_diff_D, yerr=query_pair_time_sinh_diff_D_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x,[0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_truncation_degree_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

plt.errorbar(x, memory_consumption_sinh_diff_D , marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_truncation_degree_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

plt.errorbar(x, accuracy_diff_D_sinh, yerr=accuracy_diff_D_sinh_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_truncation_degree_sinh.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()







#f=power1(x)
plt.figure()

x = np.linspace(0, len(init_time_power1), len(init_time_power1))
plt.errorbar(x, init_time_power1, yerr=init_time_power_std_err, label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

x = np.linspace(0, len(init_time_power1), len(init_time_power1))
plt.errorbar(x, query_all_time_power1, yerr=query_all_time_power_std_err, label='QueryAll time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
# plt.figure()

# plt.errorbar(x, query_one_time_power1,label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# plt.xlabel("Sketch Size", fontsize= ticks_size)
# plt.ylabel("Time (sec)", fontsize= ticks_size)
# plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# # plt.show()


plt.figure()

plt.errorbar(x, query_pair_time_power1, yerr=query_pair_time_power_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.errorbar(x, memory_consumption_power1, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()


plt.figure()
x = np.linspace(0, len(accuracy_diff_sketch_size_power), len(accuracy_diff_sketch_size_power))
plt.errorbar(x, accuracy_diff_sketch_size_power, yerr=accuracy_diff_sketch_size_power_std_err, marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Sketch Size", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160,320,1000], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
# plt.legend(loc='best', fontsize=12)
plt.savefig("accuracy_sketch_size_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()








x = np.linspace(0, len(accuracy_diff_D_power), len(accuracy_diff_D_power))

plt.figure()

plt.errorbar(x, init_time_power_diff_D, yerr=init_time_power_diff_D_std_err,  label='Init time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("init_time_truncation_degree_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()

x = np.linspace(0, len(init_time_power1), len(init_time_power1))
plt.errorbar(x, query_all_time_power_diff_D, yerr=query_all_time_power_diff_D_std_err , label='QueryAll time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (ms)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("queryall_time_truncation_degree_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

# plt.figure()

# plt.errorbar(x, query_one_time_power_diff_D,   label='QueryOne time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
# plt.xlabel("Truncation Degree", fontsize= ticks_size)
# plt.ylabel("Time (sec)", fontsize= ticks_size)
# plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
# plt.yticks(fontsize=ticks_size)
# plt.savefig("queryone_time_truncation_degree_power.pdf", dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', bbox_inches='tight')
# # plt.show()


plt.figure()

plt.errorbar(x, query_pair_time_power_diff_D, yerr=query_pair_time_power_diff_D_std_err, label='QueryPair time', marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Time (sec)", fontsize= ticks_size)
plt.xticks(x,[0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("querypair_time_truncation_degree_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()
plt.figure()

plt.errorbar(x, memory_consumption_power_diff_D,    marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Mem Usage(MB)", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("memory_consumption_truncation_degree_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()

plt.figure()
plt.errorbar(x, accuracy_diff_D_power, yerr=accuracy_diff_D_power_std_err,  marker='o', markersize=3,capsize=6, ecolor = 'red', elinewidth = 5)
plt.xlabel("Truncation Degree", fontsize= ticks_size)
plt.ylabel("Accuracy", fontsize= ticks_size)
plt.xticks(x, [0,1,2,3,5,10,20], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
# plt.legend(loc='best', fontsize=12)
plt.savefig("accuracy_truncation_degree_power.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
# plt.show()