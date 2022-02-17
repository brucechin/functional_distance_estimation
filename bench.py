import numpy as np 
import time
from scipy.stats import unitary_group
import scipy.linalg as la
from scipy.stats import unitary_group
import math

class MetricMaintenance:
    def __init__(self, n, d, D):
        self.L = 10 
        self.m = 10
        self.d = d
        self.n = n
        self.D = D #degree of truncation
        self.R = 5 #number of sampled sketches.
        tmp_sigma = [2 - 1.0/n * i for i in range(self.n)]
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
        for i in range(self.D):
            self.U.append(np.dot(self.U[-1], self.Sigma_sqrt))
        
        self.Pi = []
        for i in range(self.L):
            self.Pi.append(np.random.normal(0, 0.3, (self.m, self.d))) #Pi is the sketch matrix randomly sampled from normal distribution.

        self.x = []
        for i in range(self.n):
            self.x.append(np.random.rand(self.d))
        
        self.Pi_U = []
        for j in range(self.L):
            tmp = []
            for tau in range(self.D + 1):
                tmp.append(np.dot(self.Pi[j], self.U[tau]))
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
                tmp_norm = np.linalg.norm(tmp)
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
                d_i.append(tilde_d_i_tau * 1.0/math.factorial(tau))
            d_i_sum = np.sum(d_i)
            d.append(d_i_sum)
        # print(d)
        return np.array(d)


def accuracy(true_result, result):
    return np.sum(np.abs(true_result - result))/np.sum(np.abs(true_result))



for D in [1,2,3,5,8,10,20,40,80]:
    start = time.time()
    instance = MetricMaintenance(10000, 100, D)
    end = time.time()
    # print("init time {} seconds".format(end-start))
    start = time.time()
    q = np.random.rand(instance.d)
    print("D ={} accuracy : {}".format(D, accuracy(instance.query_all_accurate(q), instance.query_all(q))))
    end = time.time()
    # print("query avg time {} seconds".format(end-start))