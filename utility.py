import numpy as np
import scipy
import copy
import scipy.io as scio
from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.stats import poisson, norm, gamma, dirichlet, uniform, beta
import math


def load_data_fan(data, dataNum_1, dataNum_2, Times):
    # Input:
    # data: Original data T*N*N
    # dataNum, Times: N,T

    # Output:
    # relation_matrix: T*N*N, training relation matrix, Rii = 0, test relation = -1
    # test_matrix: T*N*N, test relation matrix, Rii = 0, training relation = -1


    # Initialize the the relation Rij to 0 and 1
    relation_matrix = data.astype(int)
    relation_matrix[relation_matrix>1] = 1
    relation_matrix[relation_matrix<0] = 0

    # Initialize the relation Rii to 0
    for tt in range(Times):
        relation_matrix[tt][(np.arange(dataNum_1), np.arange(dataNum_2))] = 0

    # Initialization size of training = test, with the relation =-1 in each other's matrix
    test_matrix = (np.ones((Times, dataNum_1, dataNum_2))*(-1)).astype(int)
    test_ratio = 0.1
    # test relation chose from the relation from each column with the set test_ratio
    # only choose the N*N' N'*0.1
    for tt in range(Times):
        for ii in range(dataNum_1):
            test_index_i = np.random.choice(dataNum_2, int(dataNum_2*test_ratio), replace=False)
            test_matrix[tt, ii, test_index_i] = copy.copy(relation_matrix[tt, ii, test_index_i])
            relation_matrix[tt, ii, test_index_i] = -1

    return relation_matrix, test_matrix



def initialize_model(whole_dataR, dataNum_1, dataNum_2, Times, KK, LL):
    # Input:
    # dataR_T1/T2: the list locations of Rij=1 for training data: T arrays with L(positive edges) x 2
    # KK: number of communities
    # LL: number of features
    # dataNum_1, dataNum_2, Times: N, N', T

    # Output:
    # M: Poisson distribution parameter in generating X_{ik}
    # X_i: T X N X K, latent counts for node i
    # Z_ik: T X N X K, latent integers summary, calculating as \sum_{j,k_2} Z_{ij,kk_2}/Z_{ji,kk_2}
    # Z_k1k2: K X K latent integers summary, calculating as \sum_{i,j k_1,k_2} Z_{ij,kk_2}
    # pis: LL X T X N X KK: layer-wise mixed-membership distributions
    # betas: LL-1 X T X N' X N: layer-wise information propagation coefficient
    # gammas: LL X T-1 X N' X N: layer-wise information propagation coefficient
    # Lambdas: K*K community compatibility matrix


    #### Pis
    pis = np.zeros((LL, Times, dataNum_1, KK)) ####### LL*T*N*KK
    ### Pis layer 0
    ## Pis layer 0, Times 0
    for ii in range(dataNum_1):
        pis[0, 0, ii] = dirichlet.rvs(0.1 * np.ones(KK))
    ## Pis layer 0, Times 1 to t
    gammas = gamma.rvs(1, 1, size=(LL, Times-1, dataNum_2, dataNum_1))  ####### LL*(T-1)*N'*N
    psi_0 = np.zeros((Times-1, dataNum_1, KK))   ####### (T-1)*N*KK
    for tt in range(1, Times):
        psi_0[tt-1] = np.dot(gammas[0, tt-1].T, pis[0, tt-1]) # (N'*N).T * N'*K
        for ii in range(dataNum_1):
            pis[0, tt, ii] = dirichlet.rvs(0.1 + psi_0[tt-1, ii])

    ### Pis layer 1 to L
    betas = gamma.rvs(1, 1, size=(LL - 1, Times, dataNum_2, dataNum_1))
    for ll in range(1, LL):
        ## Pis layer ll, Times 0
        psi_ll_t0 = np.dot(betas[ll-1, 0].T, pis[ll-1, 0])
        psi_ll_t0 += 1e-16
        for ii in range(dataNum_1):
            pis[ll, 0, ii] = dirichlet.rvs(psi_ll_t0[ii])
        ## Pis layer ll, Times 1 to t
        psi_ll_tt1 = np.zeros((Times, dataNum_1, KK))
        psi_ll_tt2 = np.zeros((Times-1, dataNum_1, KK))
        for tt in range(1, Times):
            psi_ll_tt1[tt] = np.dot(betas[ll-1, tt].T, pis[ll-1, tt])
            psi_ll_tt2[tt-1] = np.dot(gammas[ll, tt-1].T, pis[ll, tt-1])
            psi_ll_tt = psi_ll_tt1[tt] + psi_ll_tt2[tt-1] + 1e-16
            for ii in range(dataNum_1):
                pis[ll, tt, ii] = dirichlet.rvs(psi_ll_tt[ii])


    M = dataNum_1
    X_i = poisson.rvs(M*pis[-1]).astype(int)


    k_Lambda = 1
    theta_Lambda = 1/(M*dataNum_1)
    QQ = theta_Lambda
    Lambdas = gamma.rvs(a = k_Lambda, scale = theta_Lambda, size = (KK, KK))


    Z_ik = np.zeros((Times, dataNum_1, KK), dtype=int)
    Z_k1k2 = np.zeros((KK, KK), dtype=int)
    scale_val = 1
    for tt in range(Times):
        for ii in range(len(whole_dataR[tt])):
           pois_lambda = scale_val*(X_i[tt, whole_dataR[tt][ii][0]][:, np.newaxis] * X_i[tt, whole_dataR[tt][ii][1]][np.newaxis, :]) * Lambdas
           total_val = positive_poisson_sample(np.sum(pois_lambda))

           new_counts = np.random.multinomial(total_val, pois_lambda.reshape((-1)) / np.sum(pois_lambda)).reshape((KK, KK))
           Z_k1k2 += new_counts
           Z_ik[tt, whole_dataR[tt][ii][0]] += np.sum(new_counts, axis=1)
           Z_ik[tt, whole_dataR[tt][ii][1]] += np.sum(new_counts, axis=0)

    return M, X_i, Z_ik, Z_k1k2, pis, betas, Lambdas, gammas, QQ, scale_val

def positive_poisson_sample(z_lambda):
    # return positive truncated poisson random variables Z = 1, 2, 3, 4, ...
    # z_lambda: parameter for Poisson distribution

    sum_1 = np.exp(z_lambda)-1
    candidate = 1000
    can_val = np.arange(1, candidate)
    vals = np.exp(can_val*np.log(z_lambda)-np.cumsum(np.log(can_val)))
    select_val = can_val[np.sum((sum_1*uniform.rvs())>np.cumsum(vals))]

    # candidate = 1000
    # can_val = np.arange(1, candidate)
    # log_vals = can_val * np.log(z_lambda) - np.cumsum(np.log(can_val))
    # vals = np.exp(log_vals - np.max(log_vals))
    # select_val = np.random.choice(can_val, p=(vals / np.sum(vals)))
    return select_val



class sDGRM_class:
    def __init__(self, dataNum_1, dataNum_2, Times, LL, KK, Lambdas, QQ, M, X_i, Z_ik, Z_k1k2, pis, betas, gammas, scale_val):

        self.dataNum_1 = dataNum_1
        self.dataNum_2 = dataNum_2
        self.Times = Times
        self.LL = LL
        self.KK = KK
        self.Lambdas = Lambdas # K * K

        self.QQ = QQ
        self.M = M

        self.X_i = X_i

        self.Z_ik = Z_ik # T * N * K
        self.Z_k1k2 = Z_k1k2 # K * K

        self.pis = pis  # LL * T * N * K
        self.betas = betas # LL-1 * T * N * K
        self.gammas = gammas # LL * T-1 * N * K
        self.alphas = 0.1

        self.scale = scale_val


    def back_propagate_fan(self, dataR_H):
        # Back propagate the latent counts from X_i to the feature layer
        # Input:
        # dataR_H: the non-zeros locations of \beta (the information propagation matrix)

        # Output:
        # X_ik: LL X T X N X KK: layer-wise latent counting statistics matrix
        # y_ik: LL X T X N X KK: auxiliary counts for each pi introduced in back propagation
        # q_il: LL X T X N auxiliary variables used
        # psi_ll_tt: LL X T X N X KK: Prior of Pi's parameter
        # z_llj1_tt_ji_k: y_ik pass to pi of next layer, the layer L is 0
        # A_ll_tj1_ji_k: y_ik pass to pi of next time, the time T is 0
        # Z_ik_sum_k, A_ik_sum_k: LL X T X N X N auxiliary variables used

        ## ll=L,tt=T X(ll*tt)_ik = self.Xi[T]
        ## ll=L,tt=t, X(ll*tt)_ik = self.Xi[t]+


        #### y_ik, LL X T X N X KK
        y_ik = np.zeros((self.LL, self.Times, self.dataNum_1, self.KK)) # LL * T * N * K
        #### Z&A (LL X T X N X KK, LL layer of Z and T time of A should be all zero)
        Z_ik_sum_k = np.zeros((self.LL-1, self.Times, self.dataNum_1, self.dataNum_2)) # LL * T * N' * N
        A_ik_sum_k = np.zeros((self.LL, self.Times-1, self.dataNum_1, self.dataNum_2)) # LL * T * N' * N
        z_llj1_tt_ji_k = np.zeros((self.LL-1, self.Times, self.dataNum_1, self.dataNum_2,self.KK)) # LL * T * N' * N *K
        A_ll_tj1_ji_k = np.zeros((self.LL, self.Times-1, self.dataNum_1, self.dataNum_2,self.KK)) # LL * T * N' * N *K
        ##### Phi
        psi_ll_tt = np.zeros((self.LL, self.Times, self.dataNum_1, self.KK))
        #### q_il, LL X T X N
        q_il = np.zeros((self.LL, self.Times, self.dataNum_1))

        ### Layer L, time T, X_ik[L, T]=self.X_i[T],
        ### Layer L, time t, X_i[L,t] = self.X_ik[L, t] + A((L,t)_i'ik_sum i
        ### Layer L, time 0, X_i[L,0] = self.X_ik[L, 0] + A((L,0)_i'ik_sum i, psi_ll_tt only has beta*pi, y only generate z
        ### Layer ll!=0, time T, X_i[ll,T]= Z(ll,t)_i'ik_sum i
        ### Layer ll!=0, time t, X_i[ll,t]= Z(ll,t)_i'ik_sum i + A(ll,t)_i'ik_sum i,



        #### X_ik, LL X T X N X KK
        X_ik = np.zeros((self.LL, self.Times, self.dataNum_1, self.KK))  # LL * T * N * K
        X_ik[-1] = self.X_i    # T * N * K
        ## Layer L to 1, time T to 1
        for ll in np.arange(self.LL-1, -1, -1):
            for tt in np.arange(self.Times-1, -1, -1):
                if ll > 0:
                    if tt > 0:
                       ## beta(ll-1,tt)_i',i * Pi(ll-1, tt)_i',k  N'*1*K * N'*N*1 = N'*N*K
                       psi_ll_tt_kk_1 = self.pis[ll-1, tt][:, np.newaxis, :] * ((self.betas[ll-1, tt] * (dataR_H[tt]))[:, :, np.newaxis])
                       ## gamma(ll,tt-1)_i',i * Pi(ll, tt-1)_i',k N'*1*K * N'*N*1 = N'*N*K
                       psi_ll_tt_kk_2 = self.pis[ll, tt - 1][:, np.newaxis, :] * ((self.gammas[ll, tt - 1] * (dataR_H[tt - 1]))[:, :, np.newaxis])
                       ## phi N*K
                       psi_ll_tt[ll, tt] = np.sum(psi_ll_tt_kk_1, axis=0) + np.sum(psi_ll_tt_kk_2, axis=0)

                       for nn in range(self.dataNum_1):
                           for kk in range(self.KK):
                               if X_ik[ll, tt, nn, kk] > 0:
                                   if np.sum(psi_ll_tt[ll, tt]) > 0:
                                       y_ik[ll, tt, nn, kk] = np.sum(uniform.rvs(size=int(X_ik[ll, tt, nn, kk])) < (
                                                   psi_ll_tt[ll, tt, nn, kk] / (psi_ll_tt[ll, tt, nn, kk] + np.arange(
                                               int(X_ik[ll, tt, nn, kk])))))
                                       pp = np.zeros(self.dataNum_2 + self.dataNum_2)
                                       pp[:self.dataNum_2] = psi_ll_tt_kk_1[:, nn, kk] / psi_ll_tt[ll, tt, nn, kk]
                                       pp[self.dataNum_2:] = psi_ll_tt_kk_2[:, nn, kk] / psi_ll_tt[ll, tt, nn, kk]
                                       counts = np.random.multinomial(y_ik[ll, tt, nn, kk], pp)
                                       z_llj1_tt_ji_k[ll - 1, tt, :, nn, kk] = counts[:self.dataNum_2]
                                       A_ll_tj1_ji_k[ll, tt - 1, :, nn, kk] = counts[self.dataNum_2:]
                                       Z_ik_sum_k[ll - 1, tt, :, nn] += z_llj1_tt_ji_k[ll - 1, tt, :, nn, kk]  # N' = (L-1*T)*N*:*K sum K
                                       A_ik_sum_k[ll, tt - 1, :, nn] += A_ll_tj1_ji_k[ll, tt - 1, :, nn, kk]  # N' = ((L*T-1)*N*:*K sum K

                                       # back to ll-1, tt, kk
                                       X_ik[ll - 1, tt, :, kk] += z_llj1_tt_ji_k[ll - 1, tt, :, nn, kk]
                                       # back to ll, tt , kk
                                       X_ik[ll, tt - 1, :, kk] += A_ll_tj1_ji_k[ll, tt - 1, :, nn, kk]
                    else:
                       psi_ll_tt_kk_1 = self.pis[ll-1, tt][:, np.newaxis, :] * ((self.betas[ll-1, tt] * (dataR_H[tt]))[:, :, np.newaxis])
                       psi_ll_tt[ll, tt] = np.sum(psi_ll_tt_kk_1, axis=0)

                       for nn in range(self.dataNum_1):
                           for kk in range(self.KK):
                               if X_ik[ll, tt, nn, kk] > 0:
                                   if np.sum(psi_ll_tt[ll, tt]) > 0:
                                       y_ik[ll, tt, nn, kk] = np.sum(uniform.rvs(size=int(X_ik[ll, tt, nn, kk])) < (
                                                   psi_ll_tt[ll, tt, nn, kk] / (psi_ll_tt[ll, tt, nn, kk] + np.arange(
                                               int(X_ik[ll, tt, nn, kk])))))
                                       pp = psi_ll_tt_kk_1[:, nn, kk] / psi_ll_tt[ll, tt, nn, kk]
                                       counts = np.random.multinomial(y_ik[ll, tt, nn, kk], pp)
                                       z_llj1_tt_ji_k[ll - 1, tt, :, nn, kk] = counts
                                       Z_ik_sum_k[ll - 1, tt, :, nn] += z_llj1_tt_ji_k[ll - 1, tt, :, nn,
                                                                        kk]  # N' = (L-1*T)*N*:*K sum K
                                       # back to ll-1, tt, kk
                                       X_ik[ll - 1, tt, :, kk] += z_llj1_tt_ji_k[ll - 1, tt, :, nn, kk]
                else:
                    if tt>0:
                       psi_ll_tt_kk_2 = self.pis[ll, tt - 1][:, np.newaxis, :] * ((self.gammas[ll, tt - 1] * (dataR_H[tt - 1]))[:, :, np.newaxis])
                       psi_ll_tt[ll, tt] = np.sum(psi_ll_tt_kk_2, axis=0)

                       for nn in range(self.dataNum_1):
                           for kk in range(self.KK):
                               if X_ik[ll, tt, nn, kk] > 0:
                                   if np.sum(psi_ll_tt[ll, tt]) > 0:
                                       y_ik[ll, tt, nn, kk] = np.sum(uniform.rvs(size=int(X_ik[ll, tt, nn, kk])) < (
                                                   psi_ll_tt[ll, tt, nn, kk] / (psi_ll_tt[ll, tt, nn, kk] + np.arange(
                                               int(X_ik[ll, tt, nn, kk])))))
                                       pp = psi_ll_tt_kk_2[:, nn, kk] / psi_ll_tt[ll, tt, nn, kk]
                                       counts = np.random.multinomial(y_ik[ll, tt, nn, kk], pp)
                                       A_ll_tj1_ji_k[ll, tt-1, :, nn, kk] = counts
                                       A_ik_sum_k[ll, tt-1, :, nn] += A_ll_tj1_ji_k[ll, tt-1, :, nn,
                                                                        kk]  # N' = (L-1*T)*N*:*K sum K
                                       # back to ll, tt-1, kk
                                       X_ik[ll, tt-1, :, kk] += z_llj1_tt_ji_k[ll, tt-1, :, nn, kk]


                latent_count_i = np.sum(X_ik[ll, tt], axis=1).astype(float)  # N X(ll,tt)_ik sum K
                beta_para_1 = np.sum(psi_ll_tt[ll, tt], axis=1)  # Phi(L,tt)_ik sum k

                inte1 = gamma.rvs(a=beta_para_1 + 1e-16, scale=1) + 1e-16
                inte2 = gamma.rvs(a=latent_count_i + 1e-16, scale=1) + 1e-16

                qil_val = inte1 / (inte1 + inte2)

                q_il[ll, tt] = qil_val


        return X_ik, y_ik, q_il, z_llj1_tt_ji_k, A_ll_tj1_ji_k, Z_ik_sum_k, A_ik_sum_k, psi_ll_tt


    def sample_pis(self, X_ik, psi_ll_tt):
        # layer-wise sample mixed-membership distribution
        # Input:
        # X_ik: LL X T X N X KK: layer-wise latent counting statistics matrix
        # psi_ll_tt: LL X T X N X KK: Prior of Pi's parameter


        for ll in np.arange(self.LL):
            for tt in np.arange(self.Times):
                psi_ll = psi_ll_tt[ll, tt]
                if ll == 0:
                    psi_ll += self.alphas

                para_nn = psi_ll+X_ik[ll, tt]
                nn_pis = gamma.rvs(a=para_nn, scale=1) + 1e-16
                self.pis[ll, tt] = nn_pis / (np.sum(nn_pis, axis=1)[:, np.newaxis])


    def sample_X_i(self, dataR_matrix):
        # sample the latent counts X_i

        idx = (dataR_matrix != (-1))
        for tt in range(dataR_matrix.shape[0]):
            np.fill_diagonal(idx[tt], False)


        for tt in range(self.Times):
            for nn in range(self.dataNum_1):

                Xik_Lambda = np.sum(np.dot(self.Lambdas, ((idx[tt, nn][:, np.newaxis] * self.X_i[tt]).T)), axis=1) + \
                             np.sum(np.dot(self.Lambdas.T, (idx[tt, :, nn][:, np.newaxis] * self.X_i[tt]).T), axis=1)
                log_alpha_X = np.log(self.M) + np.log(self.pis[-1, tt][nn]) - Xik_Lambda

                for kk in range(self.KK):
                    n_X = self.Z_ik[tt, nn, kk]
                    if n_X == 0:
                        select_val = poisson.rvs(np.exp(log_alpha_X[kk]))
                    else:
                        candidates = np.arange(1, 10*self.M+1)
                        pseudos = candidates*log_alpha_X[kk]+n_X*np.log(candidates)-np.cumsum(np.log(candidates))
                        pseudos_max = np.max(pseudos)
                        proportions = np.exp(pseudos-pseudos_max)
                        select_val = np.random.choice(candidates, p=proportions / np.sum(proportions))
                    self.X_i[tt, nn, kk] = select_val


    def sample_Lambda_k1k2(self, dataR_matrix):
        # sample Lambda according to the gamma distribution

        idx = (dataR_matrix != (-1))
        Phi_KK = np.zeros((self.KK, self.KK))
        for tt in range(dataR_matrix.shape[0]):
            np.fill_diagonal(idx[tt], False)
            Phi_KK += np.dot(np.dot(self.X_i[tt].T, idx[tt]), self.X_i[tt])

        R_KK = np.ones((self.KK, self.KK))/(self.KK**2)
        np.fill_diagonal(R_KK, 1/self.KK)

        self.Lambdas = gamma.rvs(a = self.Z_k1k2 + R_KK, scale = 1)/(1+Phi_KK)


    def sample_Z_ik_k1k2(self, whole_dataR):
        # sampling the latent integers
        # Input:
        # whole_dataR: tt list of array, the L list of Rij=1 for training data

        Z_ik = np.zeros((self.Times, self.dataNum_1, self.KK), dtype=int)
        Z_k1k2 = np.zeros((self.KK, self.KK), dtype=int)
        for tt in range(self.Times):
           for ii in range(len(whole_dataR[tt])):
              pois_lambda = self.scale*(self.X_i[tt, whole_dataR[tt][ii][0]][:, np.newaxis]*self.X_i[tt, whole_dataR[tt][ii][1]][np.newaxis, :])*self.Lambdas
              total_val = positive_poisson_sample(np.sum(pois_lambda))
              new_counts = np.random.multinomial(total_val, pois_lambda.reshape((-1))/np.sum(pois_lambda)).reshape((self.KK, self.KK))
              Z_k1k2 += new_counts
              Z_ik[tt, whole_dataR[tt][ii][0]] += np.sum(new_counts, axis=1)
              Z_ik[tt, whole_dataR[tt][ii][1]] += np.sum(new_counts, axis=0)

        self.Z_k1k2 = Z_k1k2
        self.Z_ik = Z_ik


    def sample_beta(self, Z_ik_sum_k, q_il):
        # Sampling the information propagation coefficients
        # Input:
        # Z_ik_sum_k: LL X T X N X N auxiliary variables used
        # q_il: LL X T X N auxiliary variables used

        hyper_alpha = 1
        hyper_beta = 1
        for ll in range(self.betas.shape[0]):
            for tt in range(self.betas.shape[1]):
                self.betas[ll, tt] = gamma.rvs(a=hyper_alpha+Z_ik_sum_k[ll, tt], scale=1)/(hyper_beta -np.log(q_il[ll, tt][:, np.newaxis]))


    def sample_gammas(self, A_ik_sum_k, q_il):
        # Sampling the information propagation coefficients
        # Input:
        # A_ik_sum_k: LL X T X N X N auxiliary variables used
        # q_il: LL X T X N auxiliary variables used

        hyper_alpha = 1
        hyper_beta = 1
        for ll in range(self.gammas.shape[0]):
            for tt in range(self.gammas.shape[1]):
                self.gammas[ll, tt] = gamma.rvs(a=hyper_alpha+A_ik_sum_k[ll, tt], scale=1)/(hyper_beta -np.log(q_il[ll, tt][np.newaxis, :]))


    def sample_M(self, M_val):
        # updating the hyper-parameter M
        # Input:
        # M_val: Poisson distribution parameter in generating X_{ik}

        k_M = M_val
        theta_M_inverse = 1
        self.M = gamma.rvs(a=k_M+np.sum(self.X_i), scale=1)/(theta_M_inverse+self.Times*self.dataNum_1)
