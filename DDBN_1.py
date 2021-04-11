import numpy as np
import scipy
import copy
import glob
import scipy.io as sio

import time
import csv
import os

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import poisson, norm, gamma, bernoulli

from utility_v1 import *

if __name__ == '__main__':

    pathss = '...'

    filess = glob.glob(pathss + '*.npy')
    for file_i in filess[::-1]:
        ##### read file and re-arrange data to T*N*N'
        data = np.load(file_i)

        if len(data[:, 0, 0]) == len(data[0, :, 0]):
           [dataNum_1, dataNum_2, Times] = data.shape
           data_reshape_m = np.swapaxes(data, 0, 2)
           data_reshape = np.swapaxes(data_reshape_m, 1, 2)
        elif len(data[0, :, 0])== len(data[0, 0, :]):
           [Times, dataNum_1, dataNum_2] = data.shape
           data_reshape = data
        elif len(data[:, 0, 0])== len(data[0, 0, :]):
           [dataNum_1, Times, dataNum_2] = data.shape
           data_reshape = np.swapaxes(data_reshape_m, 0, 1)


        #### data preprocess
        ## training dat , test data T*N*N'
        dataR_matrix, test_relation = load_data_fan(data_reshape, dataNum_1, dataNum_2, Times)

        ## the L list of Rij=1 location for training data: T arrays with L*2
        whole_dataR = []
        for tt in range(Times):
            dataR_tt = np.asarray(np.where(dataR_matrix[tt, :, :] == 1)).T
            whole_dataR.append(dataR_tt)

        ## The list of Rij=test location and without Rii for test data
        whole_dataR_test = []
        whole_dataR_test_val = []
        for tt in range(Times):
            dataR_test_tt = np.asarray(np.where(test_relation[tt, :, :] != -1)).T
            dataR_test_val_tt = test_relation[tt][test_relation[tt, :, :] != -1]
            notdelete_index_tt = (dataR_test_tt[:, 0] != dataR_test_tt[:, 1])
            dataR_test_tt = dataR_test_tt[notdelete_index_tt]
            dataR_test_val_tt = dataR_test_val_tt[notdelete_index_tt]
            whole_dataR_test.append(dataR_test_tt)
            whole_dataR_test_val.append(dataR_test_val_tt)

        whole_dataR_test_val_together = np.asarray([item for sublist in whole_dataR_test_val for item in sublist])

        KK = 10
        LL = 3

        #### model initialization
        M_val, X_i, Z_ik, Z_k1k2, pis, betas, Lambdas, gammas, QQ, scale_val = initialize_model(whole_dataR, dataNum_1, dataNum_2, Times, KK, LL)

        #### Iteration
        IterationTime = 1000
        sDGRM = sDGRM_class(dataNum_1, dataNum_2, Times, LL, KK, Lambdas, QQ, M_val, X_i, Z_ik, Z_k1k2, pis, betas, gammas, scale_val)

        ## The T*N*N' matrix with where (the training data==1 & Rii=1) = 1, others = 0
        dataR_H = np.zeros((Times, dataNum_1, dataNum_2))
        for tt in range(Times):
            dataR_H[tt] = (dataR_matrix[tt] == 1).astype(int)+np.eye(sDGRM.dataNum_1)
        dataR_H[dataR_H > 1] = 1

        auc_seq = []
        time_seq = []
        test_precision_seq = []

        mean_pis = 0
        mean_lambda = 0

        mean_predict_val = 0
        burnInTime = int(IterationTime-100)
        # burnInTime = 0
        for ite in range(IterationTime):
            start_time = time.time()
            X_ik, y_ik, q_il, z_llji_tt_ii_k, A_ll_tj1_ii_k, Z_ik_sum_k, A_ik_sum_k, psi_ll_tt = sDGRM.back_propagate_fan(dataR_H)
            sDGRM.sample_pis(X_ik, psi_ll_tt)
            sDGRM.sample_beta(Z_ik_sum_k, q_il)
            sDGRM.sample_gammas(A_ik_sum_k, q_il)
            sDGRM.sample_M(M_val)
            sDGRM.sample_X_i(dataR_matrix)
            sDGRM.sample_Z_ik_k1k2(whole_dataR)
            sDGRM.sample_Lambda_k1k2(dataR_matrix)

            time_seq.append(time.time()-start_time)

            if ite > burnInTime:
                whole_dataR_predicted_val = []
                for tt in range(Times):
                    predicted_val = np.zeros(len(whole_dataR_test_val[tt]))
                    for ti in range(len(whole_dataR_test_val[tt])):
                        predicted_val[ti] = np.dot(np.dot(sDGRM.X_i[tt][whole_dataR_test[tt][ti, 0]][np.newaxis, :], sDGRM.Lambdas), sDGRM.X_i[tt][whole_dataR_test[tt][ti, 1]][:, np.newaxis])
                    whole_dataR_predicted_val.append(predicted_val)

                whole_dataR_predicted_val_together = np.asarray([item for sublist in whole_dataR_predicted_val for item in sublist])

                mean_predict_val = (mean_predict_val*(ite-burnInTime-1)+whole_dataR_test_val_together)/(ite-burnInTime)

                mean_pis = (mean_pis*(ite-burnInTime-1)+sDGRM.pis)/(ite-burnInTime)
                mean_lambda = (mean_lambda*(ite-burnInTime-1)+sDGRM.Lambdas)/(ite-burnInTime)


                current_AUC = roc_auc_score(whole_dataR_test_val_together, mean_predict_val)
                auc_seq.append(current_AUC)

                current_precision = average_precision_score(whole_dataR_test_val_together, mean_predict_val)
                test_precision_seq.append(current_precision)


