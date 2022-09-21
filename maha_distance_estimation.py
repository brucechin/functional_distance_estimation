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
class MetricMaintenance:
    def __init__(self, n, d, D,m, enableSketch):
        self.enableSketch = enableSketch
        self.L = 10 
        self.m = m
        self.d = d
        self.n = n
        self.D = D #degree of truncation
        self.R = 5 #number of sampled sketches.
        tmp_sigma = [1.5 - i * 0.6/self.d for i in range(self.d)]
        self.Sigma = np.zeros((self.d, self.d))
        self.Q = np.zeros((self.d, self.d))
        self.Sigma_sqrt  = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.Sigma[i][i] = tmp_sigma[i] #Sigma is a diagonal matrix in SVD
            self.Q[i][i] = 1.0 
            self.Sigma_sqrt[i][i] = math.sqrt(tmp_sigma[i])
        

        self.U = self.Sigma_sqrt
        self.A = self.Sigma
        self.Pi = []
        for i in range(self.L):
            # tmp = np.zeros((self.m, self.d))
            # for i in range(self.d):
            #     tmp[i][i] = np.random.normal(loc=0, scale=math.sqrt(0.1))
            # self.Pi.append(tmp)
            self.Pi.append(np.random.normal(loc=0, scale=math.sqrt(1.0/self.m), size=(self.m, self.d)))

        self.x = []
        for i in range(self.n):
            self.x.append(np.random.rand(self.d))
        
        self.Pi_U = []
        for j in range(self.L):
            if(self.enableSketch):
                self.Pi_U.append(np.dot(self.Pi[j], self.U))
            else:
                self.Pi_U.append(self.U)


        self.tilde_x = []
        for i in range(self.n):
            tmp_1 = []
            for j in range(self.L):
                tmp = np.dot(self.Pi_U[j], self.x[i])
                tmp_1.append(tmp)
            self.tilde_x.append(tmp_1)

    def memory_complexity(self):
        return 32.0 * (len(self.Pi_U)*len(self.Pi_U[0])*len(self.Pi_U[0][0]) + len(self.tilde_x)*len(self.tilde_x[0]) + len(self.x)*len(self.x[0]) + len(self.U) )/1000000 #MB


    def query_all_accurate(self, q):
        result = []
        for i in range(self.n):
            # print( self.U.shape, np.dot(q-self.x[i], self.U).shape,(q-self.x[i]).T.shape,  np.dot(np.dot(q-self.x[i], self.U), (q-self.x[i]).T).shape)
            result.append(np.dot(np.dot(q-self.x[i], self.U), (q-self.x[i]).T))
        return np.array(result)

    def query_one(self, q, i):
        Pi_U_q = []
        sampled_indexes = random.sample(range(self.L), self.R)
        for r in range(self.L):
            Pi_U_q.append(np.dot(self.Pi_U[r], q))
        d_i = []
        res = 0
        for r in sampled_indexes:
            tmp = Pi_U_q[r] - self.tilde_x[i][r]
            tmp_norm = np.dot(tmp, tmp.T)
            d_i.append(tmp_norm)
        res = np.median(d_i) #d_i_tau[int(self.R/2)] #np.median(d_i_tau)
        return res
    
    def query_pair(self, i, j):
        sampled_indexes = random.sample(range(self.L), self.R)
        d_i_tau = []
        for r in sampled_indexes:
            tmp =  self.tilde_x[i][r] - self.tilde_x[j][r]
            tmp_norm =  np.dot(tmp, tmp.T)
            d_i_tau.append(tmp_norm)
        tilde_d_i_tau = np.median(d_i_tau) # d_i_tau[int(self.R/2)] #np.median(d_i_tau)
        return tilde_d_i_tau

    def query_all(self, q):
        Pi_U_q = []
        # start = time.time()
        for r in range(self.L):
            Pi_U_q.append(np.dot(self.Pi_U[r], q))
        # end = time.time()
        # print("first part time {}".format(end-start))
        d = []
        sampled_indexes = random.sample(range(self.L), self.R)
        # start = time.time()
        for i in range(self.n):
            d_i = []
            for r in sampled_indexes:
                tmp = Pi_U_q[r] - self.tilde_x[i][r]
                tmp_norm =  np.dot(tmp, tmp.T)
                d_i.append(tmp_norm)
            tilde_d_i = np.median(d_i) # d_i_tau[int(self.R/2)] #np.median(d_i_tau)
            d.append(tilde_d_i)
        # end = time.time()
        # print("second part {}".format(end-start))
        return np.array(d)


def accuracy(true_result, result):
    # print(true_result[0], result[0])
    return 1 - np.mean(abs(result - true_result)/true_result)



def std_error(input):
    std_err = []
    for i in range(len(input)):
        std_err.append(np.std(input[i])/math.sqrt(len(input[i])))
    return std_err

def compute_mean(input):
    mean_out = []
    for i in range(len(input)):
        mean_out.append(np.mean(input[i]))
    return mean_out

n=10000

num_repeat = 10


# tilde_fA_f_norm_error = []
# tilde_fA_spectral_norm_error = []

# for D in [0,1,2,3,4,5,6,7,8,9,10]:
#     instance = MetricMaintenance(100, 1000, D, 10, False)
#     tilde_fA_f_norm_error.append(instance.F_norm_error())
#     tilde_fA_spectral_norm_error.append(instance.spectral_norm_error())

# print("tilde_fA_f_norm_error={}".format(tilde_fA_f_norm_error))
# print("tilde_fA_spetral_norm_error={}".format(tilde_fA_spectral_norm_error))

init_time = []
query_all_time = []
query_one_time = []
query_pair_time = []
memory_consumption_diff_sketch_size = []  #MB
accuracy_diff_sketch_size = []
accuracy_diff_D = []
memory_consumption_diff_D = []  #MB



# fix D and benchmark under different m

for d in [1000]:
    for m in [10, 20, 40, 80, 160, 320, 1000]:
        D = 0
        instance = MetricMaintenance(n, d, D, m, True)
        init_time.append([])
        for i in range(3):
            start = time.time()
            instance = MetricMaintenance(n, d, D, m, True)
            end = time.time()
            init_time[-1].append(end - start)


        memory_consumption_diff_sketch_size.append(instance.memory_complexity()) 
        #print("d={} sketch_size = {} D ={} \ninit time {} seconds".format(d, m, D, end-start))
        
        accuracy_diff_sketch_size.append([])
        query_all_time.append([])
        for i in range(num_repeat):
            start = time.time()
            q = np.random.rand(d)
            ans = instance.query_all(q)
            end = time.time()
            # used for compute error bar
            query_all_time[-1].append((end-start))
            #print("query all time {} seconds".format((end-start)))
            accuracy_diff_sketch_size[-1].append(accuracy(instance.query_all_accurate(q), ans))

        start = time.time()
        q = np.random.rand(d)
        for i in range(1000):
            instance.query_one(q, random.randint(0, n-1))
        end = time.time()
        query_one_time.append((end-start)/1000 * 1000) #millisecond
        #print("query one average time {} seconds".format((end-start)/1000))

        query_pair_time.append([])
        for j in range(num_repeat):
            start = time.time()
            for i in range(100):
                instance.query_pair(random.randint(0, n-1), random.randint(0, n-1))
            end = time.time()
            query_pair_time[-1].append((end-start)/100 * 1000) #millisecond
        #print("query pair average time {} seconds".format((end-start)/1000))


print("init_time_exp={}".format(compute_mean(init_time)))
print("query_all_time_exp={}".format(compute_mean(query_all_time)))
print("query_one_time_exp={}".format(query_one_time))
print("query_pair_time_exp={}".format(compute_mean(query_pair_time)))
print("memory_consumption_exp={}".format(memory_consumption_diff_sketch_size))
print("accuracy_diff_sketch_size_exp={}".format(compute_mean(accuracy_diff_sketch_size)))


print("accuracy_diff_sketch_size_exp_std_err={}".format(std_error(accuracy_diff_sketch_size)))
print("query_pair_time_exp_std_err={}".format(std_error(query_pair_time)))
print("query_all_time_exp_std_err={}".format(std_error(query_all_time)))
print("init_time_exp_std_err={}".format(std_error(init_time)))


# fix m and benchmark under different D

# init_time_exp=[0.3828011353810628, 0.41206757227579754, 0.5396564801534017, 0.7593146959940592, 1.2068564891815186, 1.5448856353759766, 56.919847885767616]
# query_all_time_exp=[0.3856307029724121, 0.3831638813018799, 0.389874267578125, 0.39477949142456054, 0.4064173698425293, 0.4214326858520508, 0.46199398040771483]
# query_one_time_exp=[0.09442734718322754, 0.09916305541992188, 0.11621475219726562, 0.15912413597106934, 0.15342926979064941, 0.22124099731445312, 5.798307418823242]
# query_pair_time_exp=[0.0488739013671875, 0.05272364616394043, 0.054558515548706055, 0.06737327575683594, 0.06820201873779297, 0.0700075626373291, 0.07530617713928223]
# memory_consumption_exp=[326.432, 329.632, 336.032, 348.832, 374.432, 425.632, 643.232]
# accuracy_diff_sketch_size_exp=[0.8062100233538251, 0.8448789366169189, 0.892325454361996, 0.8864268960149462, 0.8984226680513341, 0.9053748584293633, 0.9058769774566487]
# accuracy_diff_sketch_size_exp_std_err=[0.006903797338981703, 0.00986719876469659, 0.006309017079108121, 0.006647829582645628, 0.008225407524252966, 0.0061410293035376795, 0.0017203114638444513]
# query_pair_time_exp_std_err=[8.567677048443036e-05, 0.0016355790634299244, 0.001794087538249495, 0.0002291211724635064, 0.0001082572632214958, 0.00027208794759621147, 0.0007120546433458741]
# query_all_time_exp_std_err=[0.0027818648990049085, 0.0016902781559280326, 0.0013507068197254946, 0.0008862006505835312, 0.0007744545365252945, 0.0017151573548788237, 0.004021102219847897]
# init_time_exp_std_err=[0.02322881765647202, 0.003119035150572934, 0.005099828246717463, 0.019839401892474424, 0.06642548217779554, 0.09681057424228288, 0.7888658441134386]