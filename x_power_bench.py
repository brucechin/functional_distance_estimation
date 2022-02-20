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

def combination_numbers(q, n):
    return math.factorial(q)/(math.factorial(q-n) * math.factorial(n))


# scalar approximation 
# def approximate(x):
#     D = 20
#     res = (x+1)**20
#     tilde = 0
#     for i in range(D+1):
#         tilde += x**i * combination_numbers(D, i)
#         print(tilde, res)


# approximate(0.1)
# approximate(0.2)
# approximate(0.3)
# approximate(0.4)

class MetricMaintenance:
    def __init__(self, n, d, D,m, enableSketch):
        self.enableSketch = enableSketch
        self.q = 40 #(x+1)^40
        self.L = 10 
        self.m = m
        self.d = d
        self.n = n
        self.D = D #degree of truncation
        self.R = 5 #number of sampled sketches.
        tmp_sigma = [0.1 - i * 0.1/self.d for i in range(self.d)]
        self.Sigma = np.zeros((self.d, self.d))
        self.Q = np.zeros((self.d, self.d))
        self.Sigma_sqrt  = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.Sigma[i][i] = tmp_sigma[i] #Sigma is a diagonal matrix in SVD
            self.Q[i][i] = 1.0 
            self.Sigma_sqrt[i][i] = math.sqrt(tmp_sigma[i])
        
        
        self.Sigma_exp = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.Sigma_exp[i][i] = (self.Sigma[i][i] + 1)** self.q
    
        self.fA = np.dot(np.dot(self.Q, self.Sigma_exp), self.Q.T)

        self.U = [np.identity(self.d)] #the first element is identity matrix
        self.A = [np.identity(self.d)] #the first element is identity matrix

        for i in range(self.D):
            self.U.append(np.dot(self.U[-1], self.Sigma_sqrt)) 
            self.A.append(np.dot(self.A[-1], self.Sigma)) 

        for tau in range(self.D+1):
            self.A[tau] = np.dot(self.A[tau], combination_numbers(self.q, tau))

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

    def memory_complexity(self):
        return 32.0 * (len(self.Pi_U)*len(self.Pi_U[0])*len(self.Pi_U[0][0])*len(self.Pi_U[0][0][0]) + len(self.tilde_x)*len(self.tilde_x[0]) + len(self.x)*len(self.x[0]) + len(self.U) * len(self.U[0]))/1000000 #MB

    def F_norm_error(self):
        error_matrix = self.fA - self.tilde_fA
        sum_tmp = 0
        for i in range(len(self.A)):
            for j in range(len(self.A[0])):
                sum_tmp += error_matrix[i][j] * error_matrix[i][j] 
        return math.sqrt(sum_tmp)

    def spectral_norm_error(self):
        error_matrix = self.fA - self.tilde_fA
        max_tmp = 0
        for i in range(len(self.A)):
            for j in range(len(self.A[0])):
                max_tmp = max(max_tmp, error_matrix[i][j])
        return max_tmp

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
            hat_tilde_fA += np.dot(combination_numbers(self.q, tau), np.dot(self.U[tau], self.U[tau].T))
        for i in range(self.n):
            acc_result.append(np.dot(np.dot(q-self.x[i], self.fA),(q-self.x[i]).T))
            tilde_result.append(np.dot(np.dot(q-self.x[i], tilde_fA),(q-self.x[i]).T))
            hat_result.append(np.dot(np.dot(q-self.x[i], hat_tilde_fA),(q-self.x[i]).T))
            sum_tmp = 0
            for tau in range(truncation_degree+1):
                sum_tmp += combination_numbers(self.q, tau) * np.dot(np.dot( self.U[tau], q-self.x[i]), np.dot( self.U[tau], q-self.x[i]).T)
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
                tmp_norm = combination_numbers(self.q, tau) * np.dot(tmp, tmp.T)
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
                tmp_norm = combination_numbers(self.q, tau) * np.dot(tmp, tmp.T)
                d_i_tau.append(tmp_norm)
            tilde_d_i_tau = d_i_tau[int(self.R/2)] #np.median(d_i_tau)
            p.append(tilde_d_i_tau)
        p_sum = np.sum(p)
        return p_sum

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
                    tmp_norm = combination_numbers(self.q, tau) * np.dot(tmp, tmp.T)
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

init_time = []
query_all_time = []
query_one_time = []
query_pair_time = []
memory_consumption_diff_sketch_size = []  #MB
accuracy_diff_sketch_size = []
accuracy_diff_D = []
memory_consumption_diff_D = []  #MB

tilde_fA_f_norm_error = []
tilde_fA_spectral_norm_error = []

for D in [0,1,2,3,4,5,6,7,8,9,10]:
    instance = MetricMaintenance(100, 1000, D, 10, False)
    tilde_fA_f_norm_error.append(instance.F_norm_error())
    tilde_fA_spectral_norm_error.append(instance.spectral_norm_error())

print("tilde_fA_f_norm_error={}".format(tilde_fA_f_norm_error))
print("tilde_fA_spetral_norm_error={}".format(tilde_fA_spectral_norm_error))



for d in [1000]:
    for m in [10, 20, 40, 80, 160]:
        for D in [3]:
            start = time.time()
            instance = MetricMaintenance(n, d, D, m, True)
            end = time.time()
            init_time.append(end - start)
            memory_consumption_diff_sketch_size.append(instance.memory_complexity()) 
            #print("d={} sketch_size = {} D ={} \ninit time {} seconds".format(d, m, D, end-start))
            start = time.time()
            for i in range(10):
                q = np.random.rand(d)
                instance.query_all(q)
            end = time.time()
            query_all_time.append((end-start)/10)
            #print("query all time {} seconds".format((end-start)/10))
            accuracy_diff_sketch_size.append(accuracy(instance.query_all_accurate(q), instance.query_all(q)))

            start = time.time()
            q = np.random.rand(d)
            for i in range(1000):
                instance.query_one(q, random.randint(0, n-1))
            end = time.time()
            query_one_time.append((end-start)/1000 * 1000) #millisecond
            #print("query one average time {} seconds".format((end-start)/1000))


            start = time.time()
            for i in range(1000):
                instance.query_pair(random.randint(0, n-1), random.randint(0, n-1))
            end = time.time()
            query_pair_time.append((end-start)/1000 * 1000) #millisecond
            #print("query pair average time {} seconds".format((end-start)/1000))

print("init_time_exp={}".format(init_time))
print("query_all_time_exp={}".format(query_all_time))
print("query_one_time_exp={}".format(query_one_time))
print("query_pair_time_exp={}".format(query_pair_time))
print("memory_consumption_exp={}".format(memory_consumption_diff_sketch_size))
print("accuracy_diff_sketch_size_exp={}".format(accuracy_diff_sketch_size))

for d in [1000]:
    for m in [10]:
        for D in [0,1,2,5,10,20]:
            start = time.time()
            instance = MetricMaintenance(n, d, D, m, False)
            end = time.time()
            # init_time.append(end - start)
            #print("d={} sketch_size = {} D ={} \ninit time {} seconds".format(d, m, D, end-start))
            start = time.time()
            for i in range(10):
                q = np.random.rand(d)
                instance.query_all(q)
            end = time.time()
            # query_all_time.append((end-start)/10)
            #print("query all time {} seconds".format((end-start)/10))
            accuracy_diff_D.append(accuracy(instance.query_all_accurate(q), instance.query_all(q)))




print("accuracy_diff_D_exp={}".format(accuracy_diff_D))