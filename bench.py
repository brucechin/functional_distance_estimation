from re import S
import numpy as np 
import time
from scipy.stats import unitary_group
import scipy.linalg as la
from scipy.stats import unitary_group
import math
import matplotlib.pyplot as plt

class MetricMaintenance:
    def __init__(self, n, d, D):
        self.L = 10 
        self.m = d
        self.d = d
        self.n = n
        self.D = D #degree of truncation
        self.R = 5 #number of sampled sketches.
        tmp_sigma = [0.1 for i in range(self.d)]
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
            self.Pi.append(np.random.normal(loc=0, scale=math.sqrt(1.0/self.m), size=(self.m, self.d))) #TODO Pi is the sketch matrix randomly sampled from normal distribution.

        self.x = []
        for i in range(self.n):
            self.x.append(np.random.rand(self.d))
        
        self.Pi_U = []
        for j in range(self.L):
            tmp = []
            for tau in range(self.D + 1):
                tmp.append(np.dot(self.Pi[j], self.U[tau]))
                #tmp.append(self.U[tau])
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
                sum_tmp += 1.0/math.factorial(tau) * np.linalg.norm(np.dot( self.U[tau], q-self.x[i]), ord=2)
            summation_tilde_result.append(sum_tmp)
        return np.array(acc_result), np.array(tilde_result), np.array(hat_result)

    def query_one(self, q):
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
                tmp_norm = np.linalg.norm(tmp, ord=2)
                d_i_tau.append(tmp_norm)
            tilde_d_i_tau = np.median(d_i_tau)
            d_i.append(tilde_d_i_tau)
        d_i_sum = np.sum(d_i)

        return d_i_sum

    def query_all_accurate(self, q):
        result = []
        for i in range(self.n):
            result.append(np.dot(np.dot(q-self.x[i], self.fA),(q-self.x[i]).T))
        return np.array(result)

    def query_all(self, q):
        Pi_U_q = []
        for r in range(self.R):
            tmp = []
            for tau in range(self.D + 1):
                tmp.append(np.dot(self.Pi_U[r][tau], q))
            Pi_U_q.append(tmp)
        
        d = []
        for i in range(self.n):
            d_i = []
            for tau in range(self.D + 1):
                d_i_tau = []
                for r in range(self.R):
                    tmp = Pi_U_q[r][tau] - self.tilde_x[i][r][tau]
                    tmp_norm = np.linalg.norm(tmp)
                    d_i_tau.append(tmp_norm)
                tilde_d_i_tau = np.median(d_i_tau)
                d_i.append(tilde_d_i_tau)
            d_i_sum = np.sum(d_i)
            d.append(d_i_sum)
        # print(d)
        return np.array(d)


def accuracy(true_result, result):
    # print(true_result)
    # print(result)
    return np.mean(result/true_result)





n = 2
d=100
q = np.random.rand(d)
instance = MetricMaintenance(n, d, 40)

for D in [0, 1,3,5,10,20,40]:
    start = time.time()
    end = time.time()
    # print("init time {} seconds".format(end-start))
    start = time.time()
    # instance.diagonal_accuracy()
    acc_result, tilde_result, summation_tilde_result = instance.test_tilde_A_accuracy(q, D)
    print("D = {}".format(D))
    for i in range(n):
        print("f(A) norm ={}, tilde_f(A) norm={}, summation = {} ".format(acc_result[i], tilde_result[i], summation_tilde_result[i]))
    

    #print("D ={} accuracy : {}".format(D, accuracy(instance.query_all_accurate(q), instance.query_all(q))))
    end = time.time()
    # print("query avg time {} seconds".format(end-start))