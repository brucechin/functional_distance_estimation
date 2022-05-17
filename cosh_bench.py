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
        tmp_sigma = [1.2 - i * 0.6/self.d for i in range(self.d)]
        self.Sigma = np.zeros((self.d, self.d))
        self.Q = np.zeros((self.d, self.d))
        self.Sigma_sqrt  = np.zeros((self.d, self.d))
        self.Sigma_square = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.Sigma[i][i] = tmp_sigma[i] #Sigma is a diagonal matrix in SVD
            self.Q[i][i] = 1.0 
            self.Sigma_sqrt[i][i] = math.sqrt(tmp_sigma[i])
            self.Sigma_square[i][i] = tmp_sigma[i] * tmp_sigma[i]
        
        
        self.Sigma_cosh = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.Sigma_cosh[i][i] = math.cosh(self.Sigma[i][i])
    
        self.fA = np.dot(np.dot(self.Q, self.Sigma_cosh), self.Q.T)

        self.U = [np.identity(self.d)] #the first element is identity matrix
        self.A = [np.identity(self.d)] #the first element is identity matrix

        for i in range(self.D):
            self.U.append(np.dot(self.U[-1], self.Sigma)) 
            self.A.append(np.dot(self.A[-1], self.Sigma_square)) 

        for tau in range(self.D+1):
            self.A[tau] = np.dot(self.A[tau], 1.0/math.factorial(2*tau))

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
        return 32.0 * (len(self.Pi_U)*len(self.Pi_U[0])*len(self.Pi_U[0][0])*len(self.Pi_U[0][0][0]) + len(self.tilde_x)*len(self.tilde_x[0]) + len(self.x)*len(self.x[0]) + len(self.U) * len(self.U[0]))/1000000  #MB


    def query_all_accurate(self, q):
        result = []
        for i in range(self.n):
            result.append(np.dot(np.dot(q-self.x[i], self.fA),(q-self.x[i]).T))
        return np.array(result)

    def query_one(self, q, i):
        Pi_U_q = []
        for r in range(self.L):
            tmp = []
            for tau in range(self.D + 1):
                tmp.append(np.dot(self.Pi_U[r][tau], q))
            Pi_U_q.append(tmp)
        d_i = []
        sampled_indexes = random.sample(range(self.L), self.R)
        for tau in range(self.D + 1):
            d_i_tau = []
            for r in sampled_indexes:
                tmp = Pi_U_q[r][tau] - self.tilde_x[i][r][tau]
                tmp_norm = 1.0/math.factorial(2*tau) * np.dot(tmp, tmp.T)
                d_i_tau.append(tmp_norm)
            tilde_d_i_tau = np.median(d_i_tau)#d_i_tau[int(self.R/2)] #np.median(d_i_tau)
            d_i.append(tilde_d_i_tau)
        d_i_sum = np.sum(d_i)
        return d_i_sum
    
    def query_pair(self, i, j):
        p = []
        sampled_indexes = random.sample(range(self.L), self.R)
        for tau in range(self.D + 1):
            d_i_tau = []
            for r in sampled_indexes:
                tmp =  self.tilde_x[i][r][tau] - self.tilde_x[j][r][tau]
                tmp_norm = 1.0/math.factorial(2*tau) * np.dot(tmp, tmp.T)
                d_i_tau.append(tmp_norm)
            tilde_d_i_tau = np.median(d_i_tau)#d_i_tau[int(self.R/2)] #np.median(d_i_tau)
            p.append(tilde_d_i_tau)
        p_sum = np.sum(p)
        return p_sum

    def query_all(self, q):
        Pi_U_q = []
        # start = time.time()
        for r in range(self.L):
            tmp = []
            for tau in range(self.D + 1):
                tmp.append(np.dot(self.Pi_U[r][tau], q))
            Pi_U_q.append(tmp)
        d = []
        sampled_indexes = random.sample(range(self.L), self.R)
        for i in range(self.n):
            d_i = []
            for tau in range(self.D + 1):
                d_i_tau = []
                for r in sampled_indexes:
                    tmp = Pi_U_q[r][tau] - self.tilde_x[i][r][tau]
                    tmp_norm = 1.0/math.factorial(2*tau) * np.dot(tmp, tmp.T)
                    d_i_tau.append(tmp_norm)
                tilde_d_i_tau = np.median(d_i_tau)#d_i_tau[int(self.R/2)] #np.median(d_i_tau)
                d_i.append(tilde_d_i_tau)
            d_i_sum = np.sum(d_i)
            d.append(d_i_sum)

        return np.array(d)


def accuracy(true_result, result):
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

num_repeat = 5

# init_time = []
# query_all_time = []
# query_one_time = []
# query_pair_time = []
# memory_consumption_diff_sketch_size = []  #MB
# accuracy_diff_sketch_size = []
# accuracy_diff_D = []
# memory_consumption_diff_D = []  #MB

# for d in [1000]:
#     for m in [10, 20, 40, 80, 160, 320, 1000]:
#         for D in [3]:
#             instance = MetricMaintenance(n, d, D, m, True)

#             init_time.append([])
#             for i in range(3):
#                 start = time.time()
#                 instance = MetricMaintenance(n, d, D, m, True)
#                 end = time.time()
#                 init_time[-1].append(end - start)
            
#             memory_consumption_diff_sketch_size.append(instance.memory_complexity()) 
#             #print("d={} sketch_size = {} D ={} \ninit time {} seconds".format(d, m, D, end-start))
#             accuracy_diff_sketch_size.append([])
#             query_all_time.append([])
#             for i in range(num_repeat):
#                 start = time.time()
#                 q = np.random.rand(d)
#                 ans = instance.query_all(q)
#                 end = time.time()
#                 # used for compute error bar
#                 query_all_time[-1].append((end-start))
#                 #print("query all time {} seconds".format((end-start)))
#                 accuracy_diff_sketch_size[-1].append(accuracy(instance.query_all_accurate(q), ans))

#             # start = time.time()
#             # q = np.random.rand(d)
#             # for i in range(1000):
#             #     instance.query_one(q, random.randint(0, n-1))
#             # end = time.time()
#             # query_one_time.append((end-start)/1000 * 1000) #millisecond
#             # #print("query one average time {} seconds".format((end-start)/1000))

#             query_pair_time.append([])
#             for j in range(num_repeat):
#                 start = time.time()
#                 for i in range(100):
#                     instance.query_pair(random.randint(0, n-1), random.randint(0, n-1))
#                 end = time.time()
#                 query_pair_time[-1].append((end-start)/100 * 1000) #millisecond
#             #print("query pair average time {} seconds".format((end-start)/1000))

# print(accuracy_diff_sketch_size)

# print("init_time_cosh={}".format(compute_mean(init_time)))
# print("query_all_time_cosh={}".format(compute_mean(query_all_time)))
# # print("query_one_time_cosh={}".format(query_one_time))
# print("query_pair_time_cosh={}".format(compute_mean(query_pair_time)))
# print("memory_consumption_cosh={}".format(memory_consumption_diff_sketch_size))
# print("accuracy_diff_sketch_size_cosh={}".format(compute_mean(accuracy_diff_sketch_size)))


# print("accuracy_diff_sketch_size_cosh_std_err={}".format(std_error(accuracy_diff_sketch_size)))
# print("query_pair_time_cosh_std_err={}".format(std_error(query_pair_time)))
# print("query_all_time_cosh_std_err={}".format(std_error(query_all_time)))
# print("init_time_cosh_std_err={}".format(std_error(init_time)))



init_time = []
query_all_time = []
query_one_time = []
query_pair_time = []
memory_consumption_diff_sketch_size = []  #MB
accuracy_diff_sketch_size = []
accuracy_diff_D = []
memory_consumption_diff_D = []  #MB



for d in [1000]:
    for m in [160]:
        for D in [0,1,2,3,5,10,20]:
            instance = MetricMaintenance(n, d, D, m, True)

            init_time.append([])
            for i in range(3):
                start = time.time()
                instance = MetricMaintenance(n, d, D, m, True)
                end = time.time()
                init_time[-1].append(end - start)
            memory_consumption_diff_sketch_size.append(instance.memory_complexity()) 
            #print("d={} sketch_size = {} D ={} \ninit time {} seconds".format(d, m, D, end-start))

            accuracy_diff_D.append([])
            query_all_time.append([])
            for i in range(num_repeat):
                start = time.time()
                q = np.random.rand(d)
                ans = instance.query_all(q)
                end = time.time()
                query_all_time[-1].append((end-start))
                #print("query all time {} seconds".format((end-start)))
                accuracy_diff_D[-1].append(accuracy(instance.query_all_accurate(q), ans))

            # start = time.time()
            # q = np.random.rand(d)
            # for i in range(1000):
            #     instance.query_one(q, random.randint(0, n-1))
            # end = time.time()
            # query_one_time.append((end-start)/1000 * 1000) #millisecond
            # #print("query one average time {} seconds".format((end-start)/1000))
            query_pair_time.append([])
            for j in range(num_repeat):
                start = time.time()
                for i in range(100):
                    instance.query_pair(random.randint(0, n-1), random.randint(0, n-1))
                end = time.time()
                query_pair_time[-1].append((end-start)/100 * 1000) #millisecond
            #print("query pair average time {} seconds".format((end-start)/1000))

print(accuracy_diff_D)


print("memory_consumption_cosh_diff_D={}".format(memory_consumption_diff_sketch_size))

print("init_time_cosh_diff_D={}".format(compute_mean(init_time)))
print("query_all_time_cosh_diff_D={}".format(compute_mean(query_all_time)))
# print("query_one_time_cosh_diff_D={}".format(query_one_time))
print("query_pair_time_cosh_diff_D={}".format(compute_mean(query_pair_time)))
print("accuracy_diff_D_cosh={}".format(compute_mean(accuracy_diff_D)))

print("init_time_cosh_diff_D_std_err={}".format(std_error(init_time)))
print("query_all_time_cosh_diff_D_std_err={}".format(std_error(query_all_time)))
# print("query_one_time_cosh_diff_D_std_err={}".format(std_error(query_one_time)))
print("query_pair_time_cosh_diff_D_std_err={}".format(std_error(query_pair_time)))
print("accuracy_diff_D_cosh_std_err={}".format(std_error(accuracy_diff_D)))




# for d in [1000]:
#     for m in [10]:
#         for D in [0,1,2,5,10,20]:
#             start = time.time()
#             instance = MetricMaintenance(n, d, D, m, False)
#             end = time.time()
#             # init_time.append(end - start)
#             #print("d={} sketch_size = {} D ={} \ninit time {} seconds".format(d, m, D, end-start))
#             start = time.time()
#             for i in range(10):
#                 q = np.random.rand(d)
#                 instance.query_all(q)
#             end = time.time()
#             # query_all_time.append((end-start)/10)
#             #print("query all time {} seconds".format((end-start)/10))
#             accuracy_diff_D.append(accuracy(instance.query_all_accurate(q), instance.query_all(q)))







