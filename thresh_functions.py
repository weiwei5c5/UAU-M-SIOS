import math
import time
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

from multiprocessing.pool import Pool

from uau_m_sios_backup_functions import get_nei_dic
import uau_m_sios_backup_functions as us
import os


def check_threshold(net_a=None, net_b=None, initial_p=None, lamb=None, beta_u=None, gamma=None, delta=None,
                    alpha=None, sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None, min_err=None):
    # 1.store the probability(pU & pA) and initial states
    print("阈值计算开始！beta_u = ", beta_u, "gamma = ", gamma, "m = ", m, "lamb = ", lamb, "alpha = ", alpha)
    pu_dic = {}
    pa_dic = {}
    theta_i_dic = {}
    node_list = list(net_a.nodes())
    node_num = len(node_list)
    for node in node_list:
        pu_dic[node] = 1
        pa_dic[node] = 0
        theta_i_dic[node] = 1
    neigbor_dic_a = get_nei_dic(net_a)
    neigbor_dic_b = get_nei_dic(net_b)

    # seed initial
    seed_list = []
    seed_num = int(node_num * initial_p)
    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)  # the range is from 0 to node_num-1 included
        seed = node_list[seed_index]  # be careful about the index and the node-label
        if seed not in seed_list:
            seed_list = seed_list + [seed]
            pa_dic[seed] = 1
            pu_dic[seed] = 0

    # 2.iteratation of process
    for t in range(max_step):
        fra_a = 0
        temp_a = 0
        for node in node_list:
            fra_a = fra_a + pa_dic[node]
        fra_a = fra_a / node_num
        # save time
        if (fra_a - temp_a) < min_err and (t > min_step):
            break
        temp_a = fra_a

        # caculate the theta_i
        for node in node_list:
            theta_i = 1
            for j in neigbor_dic_a[node]:
                theta_i = theta_i * (1 - pa_dic[j] * lamb)
            theta_i_dic[node] = theta_i
        # iteration of pu and pa
        for node in node_list:
            pu = pu_dic[node]
            pa = pa_dic[node]
            theta_i = theta_i_dic[node]
            new_pu = pu * theta_i * (1 - m) + pa * delta * (1 - m)
            new_pa = pu * (1 - theta_i * (1 - m)) + pa * (1 - delta * (1 - m))
            pu_dic[node] = new_pu
            pa_dic[node] = new_pa

    # 2.creat the matrix H
    h = np.zeros((node_num, node_num))
    for i in range(node_num):
        node_i = node_list[i]
        pu = pu_dic[node_i]
        pa = pa_dic[node_i]
        for j in neigbor_dic_b[node_i]:
            # h[j, i] = (pu + gamma * pa)
            i = int(i)
            j = int(j)
            h[j, i] = pu * (1 - alpha * sigma) + pa * (1 - alpha) * gamma

    # 3.calculate the largest real eigenvalues
    eigenvalue, featurevector = np.linalg.eig(h)
    eigen_list = np.real(eigenvalue)
    v_max = max(eigen_list)
    thred = mu_1 / v_max

    return thred


def check_threshold_muti_process(pro_num=None, beta_list=None, net_a=None, net_b=None, initial_p=None, lamb=None,
                                 gamma=None, delta=None,
                                 alpha=None, sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                 min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for beta in beta_list:
        result.append(po.apply_async(check_threshold,
                                     args=(net_a, net_b, initial_p, lamb, beta, gamma, delta,
                                           alpha, sigma, m, mu_1, mu_2, max_step, min_step, min_err)))
    po.close()
    po.join()

    thresh_f_list = []

    for res in result:
        print(res.get())
        thresh_f = res.get()
        thresh_f_list.append(thresh_f)
    return thresh_f_list


def check_threshold_muti_process_for_lamb_list(pro_num=None, beta=None, net_a=None, net_b=None, initial_p=None, lamb_list=None,
                                               gamma=None, delta=None,
                                               alpha=None, sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                               min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for lamb in lamb_list:
        result.append(po.apply_async(check_threshold,
                                     args=(net_a, net_b, initial_p, lamb, beta, gamma, delta,
                                           alpha, sigma, m, mu_1, mu_2, max_step, min_step, min_err)))
    po.close()
    po.join()

    thresh_f_list = []

    for res in result:
        print(res.get())
        thresh_f = res.get()
        thresh_f_list.append(thresh_f)
    return thresh_f_list


def check_threshold_muti_process_for_gamma_list(pro_num=None, beta=None, net_a=None, net_b=None, initial_p=None, lamb=None,
                                                gamma_list=None, delta=None,
                                                alpha=None, sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                                min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for gamma in gamma_list:
        result.append(po.apply_async(check_threshold,
                                     args=(net_a, net_b, initial_p, lamb, beta, gamma, delta,
                                           alpha, sigma, m, mu_1, mu_2, max_step, min_step, min_err)))
    po.close()
    po.join()

    thresh_f_list = []

    for res in result:
        print(res.get())
        thresh_f = res.get()
        thresh_f_list.append(thresh_f)
    return thresh_f_list




def check_threshold_muti_process_for_m_list(pro_num=None, beta=None, net_a=None, net_b=None, initial_p=None, lamb=None,
                                            gamma=None, delta=None,
                                            alpha=None, sigma=None, m_list=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                            min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for m in m_list:
        result.append(po.apply_async(check_threshold,
                                     args=(net_a, net_b, initial_p, lamb, beta, gamma, delta,
                                           alpha, sigma, m, mu_1, mu_2, max_step, min_step, min_err)))
    po.close()
    po.join()

    thresh_f_list = []

    for res in result:
        print(res.get())
        thresh_f = res.get()
        thresh_f_list.append(thresh_f)
    return thresh_f_list


def check_threshold_muti_process_for_alpha_list(pro_num=None, beta=None, net_a=None, net_b=None, initial_p=None, lamb=None,
                                 gamma=None, delta=None,
                                 alpha_list=None, sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                 min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for alpha in alpha_list:
        result.append(po.apply_async(check_threshold,
                                     args=(net_a, net_b, initial_p, lamb, beta, gamma, delta,
                                           alpha, sigma, m, mu_1, mu_2, max_step, min_step, min_err)))
    po.close()
    po.join()

    thresh_f_list = []

    for res in result:
        print(res.get())
        thresh_f = res.get()
        thresh_f_list.append(thresh_f)
    return thresh_f_list


def check_threshold_muti_process_for_sigma_list(pro_num=None, beta=None, net_a=None, net_b=None, initial_p=None, lamb=None,
                                 gamma=None, delta=None,
                                 alpha=None, sigma_list=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                 min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for sigma in sigma_list:
        result.append(po.apply_async(check_threshold,
                                     args=(net_a, net_b, initial_p, lamb, beta, gamma, delta,
                                           alpha, sigma, m, mu_1, mu_2, max_step, min_step, min_err)))
    po.close()
    po.join()

    thresh_f_list = []

    for res in result:
        print(res.get())
        thresh_f = res.get()
        thresh_f_list.append(thresh_f)
    return thresh_f_list