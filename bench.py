from cProfile import label
from re import S
import numpy as np 
import time
from scipy.stats import unitary_group
import scipy.linalg as la
from scipy.stats import unitary_group
import math
import matplotlib.pyplot as plt

class MetricMaintenance:
    def __init__(self, n, d, D,m, enableSketch):
        self.enableSketch = enableSketch
        self.L = 10 
        self.m = m
        self.d = d
        self.n = n
        self.D = D #degree of truncation
        self.R = 5 #number of sampled sketches.
        tmp_sigma = [0.05 for i in range(self.d)]
        self.Sigma = np.zeros((self.d, self.d))
        self.Q = np.zeros((self.d, self.d))
        self.Sigma_sqrt  = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.Sigma[i][i] = tmp_sigma[i] #Sigma is a diagonal matrix in SVD
            self.Q[i][i] = 1.0 
            self.Sigma_sqrt[i][i] = math.sqrt(tmp_sigma[i])
        
        
        self.Sigma_exp = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.Sigma_exp[i][i] = math.exp(self.Sigma[i][i])
    
        self.fA = np.dot(np.dot(self.Q, self.Sigma_exp), self.Q.T)

        self.U = [np.identity(self.d)] #the first element is identity matrix
        self.A = [np.identity(self.d)] #the first element is identity matrix

        for i in range(self.D):
            self.U.append(np.dot(self.U[-1], self.Sigma_sqrt)) 
            self.A.append(np.dot(self.A[-1], self.Sigma)) 

        for tau in range(self.D+1):
            self.A[tau] = np.dot(self.A[tau], 1.0/math.factorial(tau))

        self.tilde_fA = np.zeros((self.d, self.d))
        for tau in range(self.D+1):
            self.tilde_fA += self.A[tau]


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
            tmp = []
            for tau in range(self.D + 1):
                if(self.enableSketch):
                    tmp.append(np.dot(self.Pi[j], self.U[tau]))
                else:
                    tmp.append(self.U[tau])
            self.Pi_U.append(tmp)

        self.tilde_x = []
        for i in range(self.n):
            tmp_1 = []
            for j in range(self.L):
                tmp_2 = []
                for tau in range(self.D + 1):
                    tmp = np.dot(self.Pi_U[j][tau], self.x[i])
                    tmp_2.append(tmp)
                tmp_1.append(tmp_2)
            self.tilde_x.append(tmp_1)

    def diagonal_accuracy(self):
        tmp = 0
        for i in range(self.d):
            diag = 0
            for j in range(self.D + 1):
                diag += self.A[j][i][i]
            tmp += diag/ self.fA[i][i]
        print(tmp/self.d)

    def test_tilde_A_accuracy(self,q, truncation_degree):
        acc_result = []
        tilde_result = []
        hat_result = []
        summation_tilde_result = []

        tilde_fA = np.zeros((self.d, self.d))
        hat_tilde_fA = np.zeros((self.d, self.d))

        for tau in range(truncation_degree+1):
            tilde_fA += self.A[tau]
            hat_tilde_fA += np.dot(math.factorial(tau), np.dot(self.U[tau], self.U[tau].T))
        for i in range(self.n):
            acc_result.append(np.dot(np.dot(q-self.x[i], self.fA),(q-self.x[i]).T))
            tilde_result.append(np.dot(np.dot(q-self.x[i], tilde_fA),(q-self.x[i]).T))
            hat_result.append(np.dot(np.dot(q-self.x[i], hat_tilde_fA),(q-self.x[i]).T))
            sum_tmp = 0
            for tau in range(truncation_degree+1):
                sum_tmp += 1.0/math.factorial(tau) * np.dot(np.dot( self.U[tau], q-self.x[i]), np.dot( self.U[tau], q-self.x[i]).T)
            summation_tilde_result.append(sum_tmp)
        return np.array(acc_result), np.array(tilde_result), np.array(summation_tilde_result)


    def query_all_accurate(self, q):
        result = []
        for i in range(self.n):
            result.append(np.dot(np.dot(q-self.x[i], self.fA),(q-self.x[i]).T))
        return np.array(result)

    def query_one(self, q, i):
        Pi_U_q = []
        for r in range(self.R):
            tmp = []
            for tau in range(self.D + 1):
                tmp.append(np.dot(self.Pi_U[r][tau], q))
            Pi_U_q.append(tmp)
        d_i = []
        for tau in range(self.D + 1):
            d_i_tau = []
            for r in range(self.R):
                tmp = Pi_U_q[r][tau] - self.tilde_x[i][r][tau]
                tmp_norm = 1.0/math.factorial(tau) * np.dot(tmp, tmp.T)
                d_i_tau.append(tmp_norm)
            tilde_d_i_tau = d_i_tau[int(self.R/2)] #np.median(d_i_tau)
            d_i.append(tilde_d_i_tau)
        d_i_sum = np.sum(d_i)
        return d_i_sum
    
    def query_pair(self, i, j):
        p = []
        for tau in range(self.D + 1):
            d_i_tau = []
            for r in range(self.R):
                tmp =  self.tilde_x[i][r][tau] - self.tilde_x[j][r][tau]
                tmp_norm = 1.0/math.factorial(tau) * np.dot(tmp, tmp.T)
                d_i_tau.append(tmp_norm)
            tilde_d_i_tau = d_i_tau[int(self.R/2)] #np.median(d_i_tau)
            p.append(tilde_d_i_tau)
        p_sum = np.sum(p)
        return p_sum
        return p

    def query_all(self, q):
        Pi_U_q = []
        # start = time.time()
        for r in range(self.R):#TODO lianke, r should be randomly choosen from [L] for R times.
            tmp = []
            for tau in range(self.D + 1):
                tmp.append(np.dot(self.Pi_U[r][tau], q))
            Pi_U_q.append(tmp)
        # end = time.time()
        # print("first part time {}".format(end-start))
        d = []

        # start = time.time()
        for i in range(self.n):
            d_i = []
            for tau in range(self.D + 1):
                d_i_tau = []
                for r in range(self.R):
                    tmp = Pi_U_q[r][tau] - self.tilde_x[i][r][tau]
                    tmp_norm = 1.0/math.factorial(tau) * np.dot(tmp, tmp.T)
                    d_i_tau.append(tmp_norm)
                tilde_d_i_tau = d_i_tau[int(self.R/2)] #np.median(d_i_tau)
                d_i.append(tilde_d_i_tau)
            d_i_sum = np.sum(d_i)
            d.append(d_i_sum)
        # end = time.time()
        # print("second part {}".format(end-start))
        return np.array(d)


def accuracy(true_result, result):

    return 1 - np.mean(abs(result - true_result)/true_result)





n=10000

init_time = [1.1681427955627441, 1.5417754650115967, 5.062142848968506, 16.5169997215271, 24.563014268875122]
query_time = [0.4315241813659668, 0.44232289791107177, 0.46778163909912107, 0.4692366123199463, 0.5201902389526367]
accuracy_list = [0.6919568568103092, 0.7497045783016956, 0.8461419165704719, 0.8918277703205306, 0.9221402268148761]

# for d in [1000]:
#     for m in [10, 20, 40, 80, 160]:
#         for D in [3]:
#             start = time.time()
#             instance = MetricMaintenance(n, d, D, m, True)
#             end = time.time()
#             init_time.append(end - start)
#             print("d={} sketch_size = {} D ={} \ninit time {} seconds".format(d, m, D, end-start))
#             start = time.time()
#             for i in range(10):
#                 q = np.random.rand(d)
#                 instance.query_all(q)
#             end = time.time()
#             query_time.append((end-start)/10)
#             print("query all time {} seconds".format((end-start)/10))
#             accuracy_list.append(accuracy(instance.query_all_accurate(q), instance.query_all(q)))
#             # print("d={} D ={}query avg time {} seconds".format(d, D,end-start))

print(init_time)
print(query_time)
print(accuracy_list)

ticks_size = 20

x = np.linspace(0, len(init_time), len(init_time))
plt.plot(x, init_time, label='init time')
plt.plot(x, query_time, label='query time')
plt.xlabel("sketch size", fontsize= ticks_size)
plt.ylabel("time (sec)", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.legend(loc='upper left')
plt.yticks(fontsize=ticks_size)
plt.legend(loc="best", fontsize=20)
plt.savefig("time_sketch_size.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
plt.show()


plt.plot(x, accuracy_list)
plt.xlabel("sketch size", fontsize= ticks_size)
plt.ylabel("accuracy", fontsize= ticks_size)
plt.xticks(x, [10, 20, 40, 80, 160], fontsize= ticks_size)
plt.yticks(fontsize=ticks_size)
plt.savefig("accuracy_sketch_size.pdf", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', bbox_inches='tight')
plt.show()


