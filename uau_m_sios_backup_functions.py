import math
import sys
import time
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from numba import njit
from scipy import optimize
from collections import namedtuple
import operator
import scipy.stats as stats
from multiprocessing.pool import Pool


def read_graph(filename=None, separator=None):
    fname = filename
    graph = nx.read_edgelist(fname, delimiter=separator, nodetype=None)
    graph = graph.to_undirected()
    print('is_connected:', nx.is_connected(graph))
    return graph, fname.split('/')[-1]


def get_nei_dic(net=None):
    node_list = net.nodes()
    adjlist = {}
    for temp in node_list:
        adjlist[temp] = [n for n in net.neighbors(temp)]
    return adjlist


def del_dic_b_from_dic_a(dic_a=None, dic_b=None):
    for key in dic_b:
        if key in dic_a:
            dic_a.pop(key)
    return dic_a


def check_nodes_equality(net_a=None, net_b=None):
    # 检查网络中的节点是否一一对应
    nodes_a = net_a.nodes()
    nodes_b = net_b.nodes()
    print('the two networks have the same node list?', operator.eq(nodes_a, nodes_b))


def check_edges_equality(net_a=None, net_b=None):
    # 检查网络中的节点是否一一对应
    edges_a = net_a.edges()
    edges_b = net_b.edges()
    print('the two networks have the same edge list?', operator.eq(edges_a, edges_b))


def calculate_interlayer_degree_correlation(net_a=None, net_b=None):
    # calculate the correlation,due to the existence of nodes with the same degree,the corr should not be exact 1 or -1
    node_list = net_a.nodes()
    x = []
    y = []
    for i in node_list:
        x = x + [net_a.degree(i)]
        y = y + [net_b.degree(i)]
        # print(i,ER_a.degree(i))
    corr = stats.spearmanr(x, y)
    return corr


def calculate_average_degree(net=None):
    # calculate the average degree
    isolated_node_number = nx.degree_histogram(net)[0]
    node_num = len(net.nodes())
    print('the isolated node number is :', isolated_node_number)
    degree_list = nx.degree_histogram(net)[1:]  # remove the nodes with degree 0
    max_degree = len(degree_list)
    avg_k = 0
    Pk_dic = {}
    for k in range(1, max_degree + 1):
        if degree_list[k - 1] != 0:
            Pk_dic[k] = degree_list[k - 1] / node_num
            avg_k = avg_k + k * Pk_dic[k]
    return avg_k


def check_multinet_validity(net_a=None, net_b=None):
    check_nodes_equality(net_a=net_a, net_b=net_b)
    check_edges_equality(net_a=net_a, net_b=net_b)
    ave_a = calculate_average_degree(net=net_a)
    ave_b = calculate_average_degree(net=net_b)
    correlation = calculate_interlayer_degree_correlation(net_a=net_a, net_b=net_b)
    print('the average degree of net_a is', ave_a)
    print('the average degree of net_b is', ave_b)
    print('the interlayer degree correlation is', correlation)


def mmca_of_uau_m_sios(net_a=None, net_b=None, initial_p=None, lamb=None, beta_u=None, beta_a=None, delta=None,
                       alpha=None, sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                       min_err=None):
    # initial states
    joint_dic = {}  # store the states probability of nodes
    node_list = list(net_a.nodes())
    node_num = len(node_list)
    for node in node_list:
        joint_dic[node] = {'AI': 0, 'UI': 0, 'AS': 0, 'US': 1, 'AO': 0}

    # initial seeds
    seed_list = []
    seed_num = int(node_num * initial_p)
    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)  # the range is from 0 to node_num-1 included
        seed = node_list[seed_index]
        # print('node list', node_list)
        # print('seed index', seed_index)
        # print('seed', seed)
        if seed not in seed_list:
            seed_list = seed_list + [seed]
            joint_dic[seed] = {'AI': 0, 'UI': 1, 'AS': 0, 'US': 0, 'AO': 0}
    # print('seed list:', seed_list)

    # states evolution starts
    neigbor_dic_a = get_nei_dic(net_a)
    neigbor_dic_b = get_nei_dic(net_b)  # for calculating the complex transmission probablity
    temp_i = 0
    temp_a = 0
    fac_list_s, fac_list_i, fac_list_u, fac_list_a, fac_list_o, time_list = [], [], [], [], [], []
    for t in range(max_step):
        # calculate the fraction of nodes in each state
        fac_s, fac_i, fac_u, fac_a, fac_o = 0, 0, 0, 0, 0
        state_dic = {}
        for node in node_list:
            state_s = joint_dic[node]['AS'] + joint_dic[node]['US']
            fac_s = fac_s + state_s
            state_i = joint_dic[node]['AI'] + joint_dic[node]['UI']
            fac_i = fac_i + state_i
            state_u = joint_dic[node]['UI'] + joint_dic[node]['US']
            fac_u = fac_u + state_u
            state_a = joint_dic[node]['AI'] + joint_dic[node]['AO'] + joint_dic[node]['AS']
            fac_a = fac_a + state_a
            state_o = joint_dic[node]['AO']
            fac_o = fac_o + state_o
            state_dic[node] = {'S': state_s, 'I': state_i, 'U': state_u, 'A': state_a, 'O': state_o}
        fac_s = fac_s / node_num
        fac_i = fac_i / node_num
        fac_u = fac_u / node_num
        fac_a = fac_a / node_num
        fac_o = fac_o / node_num
        fac_list_s.append(fac_s)
        fac_list_i.append(fac_i)
        fac_list_u.append(fac_u)
        fac_list_a.append(fac_a)
        fac_list_o.append(fac_o)
        time_list.append(t)
        # check whether the model is nomalized
        if (fac_a + fac_u - 1.0) > min_err or (fac_s + fac_i + fac_o - 1.0) > min_err:
            print('The model is not normalized')
            print('the value of fac_a+fac_u is:', fac_a + fac_u)
            print('the value of fac_s+fac_i+fac_o is:', fac_a + fac_u)
            break
        # save time
        if (fac_i - temp_i) < min_err and (fac_a - temp_a) < min_err and (t > min_step):
            # print('complete ahead of schedule at step:', t)
            # print(fac_a, fac_list_a[-1], fac_i, fac_list_i[-1], fac_o, fac_list_o[-1])
            return time_list, fac_list_a, fac_list_i, fac_list_o, fac_a, fac_i, fac_o
        temp_i = fac_i
        temp_a = fac_a

        # calculate the complex transmission probablity of nodes
        complex_state_dic = {}  # store the complex transmission probablity of nodes
        for node in node_list:
            theta_i, qu_i, qa_i = 1, 1, 1
            for j in neigbor_dic_a[node]:
                theta_i = theta_i * (1 - state_dic[j][
                    'A'] * lamb)  # the node in U state did not receive message from all the neighbors
            for j in neigbor_dic_b[node]:
                qu_i = qu_i * (1 - state_dic[j]['I'] * beta_u)
                qa_i = qa_i * (1 - state_dic[j]['I'] * beta_a)
            complex_state_dic[node] = {'theta': theta_i, 'qa': qa_i, 'qu': qu_i}

        # update the states in t+1 according to the states in t
        for node in node_list:
            p_ai = joint_dic[node]['AI']
            p_ui = joint_dic[node]['UI']
            p_as = joint_dic[node]['AS']
            p_us = joint_dic[node]['US']
            p_ao = joint_dic[node]['AO']
            theta = complex_state_dic[node]['theta']
            qa = complex_state_dic[node]['qa']
            qu = complex_state_dic[node]['qu']

            new_ui = 1.0 * (p_ui * theta * (1 - m) * (1 - mu_1) * (1 - sigma) +
                            p_ai * delta * (1 - m) * (1 - mu_1) * (1 - sigma) +
                            p_us * theta * (1 - m) * (1 - qu) * (1 - sigma) +
                            p_as * delta * (1 - m) * (1 - qu) * (1 - sigma)
                            )

            new_ai = 1.0 * (p_ui * ((1 - theta) * (1 - mu_1) * (1 - alpha) + theta * m * (1 - mu_1) * (1 - alpha) +
                                    theta * (1 - m) * (1 - mu_1) * sigma * (1 - alpha)) +
                            p_ai * (delta * m * (1 - mu_1) * (1 - alpha) + delta * (1 - m) * (1 - mu_1) * sigma * (
                            1 - alpha) +
                                    (1 - delta) * (1 - mu_1) * (1 - alpha)) +
                            p_us * ((1 - theta) * (1 - qa) * (1 - alpha) + theta * m * (1 - qa) * (1 - alpha) +
                                    theta * (1 - m) * (1 - qu) * sigma * (1 - alpha)) +
                            p_as * (delta * m * (1 - qa) * (1 - alpha) + delta * (1 - m) * (1 - qu) * sigma * (
                            1 - alpha) +
                                    (1 - delta) * (1 - qa) * (1 - alpha))
                            )

            new_us = p_ui * theta * (1 - m) * mu_1 + \
                     p_ai * delta * (1 - m) * mu_1 + \
                     p_us * theta * (1 - m) * qu + \
                     p_as * delta * (1 - m) * qu

            new_as = 1.0 * (p_ui * ((1 - theta) * mu_1 + theta * m * mu_1) +
                            p_ai * ((1 - delta) * mu_1 + delta * m * mu_1) +
                            p_us * ((1 - theta) * qa + theta * m * qa) +
                            p_as * ((1 - delta) * qa + delta * m * qa) +
                            p_ao * mu_2
                            )

            new_ao = 1.0 * (p_ui * ((1 - theta) * (1 - mu_1) * alpha + theta * m * (1 - mu_1) * alpha +
                                    theta * (1 - m) * (1 - mu_1) * sigma * alpha) +
                            p_ai * (delta * m * (1 - mu_1) * alpha + delta * (1 - m) * (1 - mu_1) * sigma * alpha +
                                    (1 - delta) * (1 - mu_1) * alpha) +
                            p_us * ((1 - theta) * (1 - qa) * alpha + theta * m * (1 - qa) * alpha +
                                    theta * (1 - m) * (1 - qu) * sigma * alpha) +
                            p_as * (delta * m * (1 - qa) * alpha + delta * (1 - m) * (1 - qu) * sigma * alpha +
                                    (1 - delta) * (1 - qa) * alpha) +
                            p_ao * (1 - mu_2)
                            )
            joint_dic[node]['AI'] = new_ai
            joint_dic[node]['UI'] = new_ui
            joint_dic[node]['AS'] = new_as
            joint_dic[node]['US'] = new_us
            joint_dic[node]['AO'] = new_ao

    # fac_i = fac_list_i[-1]
    # fac_a = fac_list_a[-1]
    # fac_o = fac_list_o[-1]
    # print(fac_a, fac_list_i[-1], fac_i, fac_list_a[-1], fac_o,fac_list_o[-1])
    return time_list, fac_list_a, fac_list_i, fac_list_o, fac_a, fac_i, fac_o


def mmca_of_uau_m_sios_for_multi_process(net_a, net_b, initial_p, lamb, beta_u, beta_a, delta,
                                         alpha, sigma, m, mu_1, mu_2, max_step, min_step,
                                         min_err):
    # initial states
    joint_dic = {}  # store the states probability of nodes
    node_list = list(net_a.nodes())
    node_num = len(node_list)
    for node in node_list:
        joint_dic[node] = {'AI': 0, 'UI': 0, 'AS': 0, 'US': 1, 'AO': 0}

    # initial seeds
    seed_list = []
    seed_num = int(node_num * initial_p)
    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)  # the range is from 0 to node_num-1 included
        seed = node_list[seed_index]
        # print('node list', node_list)
        # print('seed index', seed_index)
        # print('seed', seed)
        if seed not in seed_list:
            seed_list = seed_list + [seed]
            joint_dic[seed] = {'AI': 0, 'UI': 1, 'AS': 0, 'US': 0, 'AO': 0}
    # print('seed list:', seed_list)

    # states evolution starts
    neigbor_dic_a = get_nei_dic(net_a)
    neigbor_dic_b = get_nei_dic(net_b)  # for calculating the complex transmission probablity
    temp_i = 0
    temp_a = 0
    fac_list_s, fac_list_i, fac_list_u, fac_list_a, fac_list_o, time_list = [], [], [], [], [], []
    for t in range(max_step):
        # calculate the fraction of nodes in each state
        fac_s, fac_i, fac_u, fac_a, fac_o = 0, 0, 0, 0, 0
        state_dic = {}
        for node in node_list:
            state_s = joint_dic[node]['AS'] + joint_dic[node]['US']
            fac_s = fac_s + state_s
            state_i = joint_dic[node]['AI'] + joint_dic[node]['UI']
            fac_i = fac_i + state_i
            state_u = joint_dic[node]['UI'] + joint_dic[node]['US']
            fac_u = fac_u + state_u
            state_a = joint_dic[node]['AI'] + joint_dic[node]['AO'] + joint_dic[node]['AS']
            fac_a = fac_a + state_a
            state_o = joint_dic[node]['AO']
            fac_o = fac_o + state_o
            state_dic[node] = {'S': state_s, 'I': state_i, 'U': state_u, 'A': state_a, 'O': state_o}
        fac_s = fac_s / node_num
        fac_i = fac_i / node_num
        fac_u = fac_u / node_num
        fac_a = fac_a / node_num
        fac_o = fac_o / node_num
        fac_list_s.append(fac_s)
        fac_list_i.append(fac_i)
        fac_list_u.append(fac_u)
        fac_list_a.append(fac_a)
        fac_list_o.append(fac_o)
        time_list.append(t)
        # check whether the model is nomalized
        if (fac_a + fac_u - 1.0) > min_err or (fac_s + fac_i + fac_o - 1.0) > min_err:
            print('The model is not normalized')
            print('the value of fac_a+fac_u is:', fac_a + fac_u)
            print('the value of fac_s+fac_i+fac_o is:', fac_a + fac_u)
            break
        # save time
        if (fac_i - temp_i) < min_err and (fac_a - temp_a) < min_err and (t > min_step):
            # print('complete ahead of schedule at step:', t)
            # print(fac_a, fac_list_a[-1], fac_i, fac_list_i[-1], fac_o, fac_list_o[-1])
            return (fac_a, fac_i, fac_o)
        temp_i = fac_i
        temp_a = fac_a

        # calculate the complex transmission probablity of nodes
        complex_state_dic = {}  # store the complex transmission probablity of nodes
        for node in node_list:
            theta_i, qu_i, qa_i = 1, 1, 1
            for j in neigbor_dic_a[node]:
                theta_i = theta_i * (1 - state_dic[j][
                    'A'] * lamb)  # the node in U state did not receive message from all the neighbors
            for j in neigbor_dic_b[node]:
                qu_i = qu_i * (1 - state_dic[j]['I'] * beta_u)
                qa_i = qa_i * (1 - state_dic[j]['I'] * beta_a)
            complex_state_dic[node] = {'theta': theta_i, 'qa': qa_i, 'qu': qu_i}

        # update the states in t+1 according to the states in t
        for node in node_list:
            p_ai = joint_dic[node]['AI']
            p_ui = joint_dic[node]['UI']
            p_as = joint_dic[node]['AS']
            p_us = joint_dic[node]['US']
            p_ao = joint_dic[node]['AO']
            theta = complex_state_dic[node]['theta']
            qa = complex_state_dic[node]['qa']
            qu = complex_state_dic[node]['qu']

            new_ui = 1.0 * (p_ui * theta * (1 - m) * (1 - mu_1) * (1 - sigma) +
                            p_ai * delta * (1 - m) * (1 - mu_1) * (1 - sigma) +
                            p_us * theta * (1 - m) * (1 - qu) * (1 - sigma) +
                            p_as * delta * (1 - m) * (1 - qu) * (1 - sigma)
                            )

            new_ai = 1.0 * (p_ui * ((1 - theta) * (1 - mu_1) * (1 - alpha) + theta * m * (1 - mu_1) * (1 - alpha) +
                                    theta * (1 - m) * (1 - mu_1) * sigma * (1 - alpha)) +
                            p_ai * (delta * m * (1 - mu_1) * (1 - alpha) + delta * (1 - m) * (1 - mu_1) * sigma * (
                            1 - alpha) +
                                    (1 - delta) * (1 - mu_1) * (1 - alpha)) +
                            p_us * ((1 - theta) * (1 - qa) * (1 - alpha) + theta * m * (1 - qa) * (1 - alpha) +
                                    theta * (1 - m) * (1 - qu) * sigma * (1 - alpha)) +
                            p_as * (delta * m * (1 - qa) * (1 - alpha) + delta * (1 - m) * (1 - qu) * sigma * (
                            1 - alpha) +
                                    (1 - delta) * (1 - qa) * (1 - alpha))
                            )

            new_us = p_ui * theta * (1 - m) * mu_1 + p_ai * delta * (1 - m) * mu_1 + p_us * theta * (
                    1 - m) * qu + p_as * delta * (1 - m) * qu

            new_as = 1.0 * (p_ui * ((1 - theta) * mu_1 + theta * m * mu_1) +
                            p_ai * ((1 - delta) * mu_1 + delta * m * mu_1) +
                            p_us * ((1 - theta) * qa + theta * m * qa) +
                            p_as * ((1 - delta) * qa + delta * m * qa) +
                            p_ao * mu_2
                            )

            new_ao = 1.0 * (p_ui * ((1 - theta) * (1 - mu_1) * alpha + theta * m * (1 - mu_1) * alpha +
                                    theta * (1 - m) * (1 - mu_1) * sigma * alpha) +
                            p_ai * (delta * m * (1 - mu_1) * alpha + delta * (1 - m) * (1 - mu_1) * sigma * alpha +
                                    (1 - delta) * (1 - mu_1) * alpha) +
                            p_us * ((1 - theta) * (1 - qa) * alpha + theta * m * (1 - qa) * alpha +
                                    theta * (1 - m) * (1 - qu) * sigma * alpha) +
                            p_as * (delta * m * (1 - qa) * alpha + delta * (1 - m) * (1 - qu) * sigma * alpha +
                                    (1 - delta) * (1 - qa) * alpha) +
                            p_ao * (1 - mu_2)
                            )
            joint_dic[node]['AI'] = new_ai
            joint_dic[node]['UI'] = new_ui
            joint_dic[node]['AS'] = new_as
            joint_dic[node]['US'] = new_us
            joint_dic[node]['AO'] = new_ao
    # fac_i = fac_list_i[-1]
    # fac_a = fac_list_a[-1]
    # fac_o = fac_list_o[-1]
    # print(fac_a, fac_list_i[-1], fac_i, fac_list_a[-1], fac_o,fac_list_o[-1])
    return (fac_a, fac_i, fac_o)


def mmca_of_uau_m_sios_multi_process(pro_num=None, beta_list=None, net_a=None, net_b=None, initial_p=None, lamb=None,
                                     gamma=None, delta=None,
                                     alpha=None, sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                     min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for beta in beta_list:
        # print(beta)
        result.append(po.apply_async(mmca_of_uau_m_sios_for_multi_process,
                                     args=(net_a, net_b, initial_p, lamb, beta, beta * gamma, delta,
                                           alpha, sigma, m, mu_1, mu_2, max_step, min_step,
                                           min_err)))
    po.close()
    po.join()
    fac_list_a = []
    fac_list_i = []
    fac_list_o = []
    for res in result:
        # print(res.get())
        fac_a = res.get()[0]
        fac_i = res.get()[1]
        fac_o = res.get()[2]
        fac_list_a.append(fac_a)
        fac_list_i.append(fac_i)
        fac_list_o.append(fac_o)
    return fac_list_a, fac_list_i, fac_list_o


def mc_of_uau_m_sios(net_a=None, net_b=None, initial_p=None, lamb=None, beta_u=None, beta_a=None, delta=None,
                     alpha=None,
                     sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None, min_err=None):
    node_list = list(net_a.nodes())
    node_num = len(node_list)

    # initial seeds
    seed_list = []
    seed_num = int(node_num * initial_p)
    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)  # the range is from 0 to node_num-1 included
        seed = node_list[seed_index]
        # print('node list', node_list)
        # print('seed index', seed_index)
        # print('seed', seed)
        if seed not in seed_list:
            seed_list = seed_list + [seed]

    neigbor_dic_a = get_nei_dic(net_a)
    neigbor_dic_b = get_nei_dic(net_b)
    # initial the dicts
    u_dic = {}
    a_dic = {}
    i_dic = {}
    s_dic = {}
    o_dic = {}
    for node in node_list:
        u_dic[node] = 0
        if node in seed_list:
            i_dic[node] = 0
        else:
            s_dic[node] = 0

    # normalization check
    # normalization_check_mc_of_uau_sios(node_list=node_list, a_dic=a_dic, u_dic=u_dic, s_dic=s_dic, i_dic=i_dic, o_dic=o_dic)

    fac_list_s, fac_list_i, fac_list_u, fac_list_a, fac_list_o, time_list = [], [], [], [], [], []
    temp_i = 0
    temp_a = 0
    for t in range(max_step):
        # UAU process starts
        # UI/US -->lambda AI/AS. Note UO does not exist
        # temp dicts
        new_u_dic = {}
        new_a_dic = {}
        for i in a_dic:
            neighbor_list = neigbor_dic_a[i]
            for j in neighbor_list:
                if j in u_dic:
                    temp = random.random()
                    if temp < lamb:
                        new_a_dic[j] = t
        # AS/AI --> delta US/UI. Note AO can not change to UO
        for i in a_dic:
            if i not in o_dic:
                temp = random.random()
                if temp < delta:
                    new_u_dic[i] = t

        # update the UAU states of nodes
        a_dic.update(new_a_dic)
        a_dic = del_dic_b_from_dic_a(a_dic, new_u_dic)

        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)
        u_dic.update(new_u_dic)

        # mass-media process start
        for node in u_dic:
            temp = random.random()
            if temp < m:
                new_u_dic[node] = t
        a_dic = del_dic_b_from_dic_a(a_dic, new_u_dic)
        u_dic.update(new_u_dic)

        # SIOS process start
        # US -->betaU UI; AS -->betaA AI
        new_i_dic = {}
        new_s_from_i = {}
        new_s_from_o = {}
        for i in i_dic:
            neighbor_list = neigbor_dic_b[i]
            for j in neighbor_list:
                if j in s_dic:
                    if j in u_dic:
                        temp = random.random()
                        if temp < beta_u:
                            new_i_dic[j] = t
                    elif j in a_dic:
                        temp = random.random()
                        if temp < beta_a:
                            new_i_dic[j] = t
        # AI\UI -->mu_1 AS\US; AO --> mu_2 AS
        for i in i_dic:
            temp = random.random()
            if temp < mu_1:
                new_s_from_i[i] = t
        for i in o_dic:
            if i in a_dic:
                temp = random.random()
                if temp < mu_2:
                    new_s_from_o[i] = t
        # update the SIOS states of nodes
        s_dic.update(new_s_from_o)
        s_dic.update(new_s_from_i)
        s_dic = del_dic_b_from_dic_a(s_dic, new_i_dic)
        i_dic.update(new_i_dic)
        i_dic = del_dic_b_from_dic_a(i_dic, new_s_from_i)
        o_dic = del_dic_b_from_dic_a(o_dic, new_s_from_o)

        # self-awakening process start
        # UI --> sigma AI
        new_a_dic = {}
        for i in u_dic:
            if i in i_dic:
                temp = random.random()
                if temp < sigma:
                    new_a_dic[i] = t
        # update the self-awakening states of nodes
        a_dic.update(new_a_dic)
        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)

        # self-quarantine process start
        # AI --> alpha AO
        new_o_dic = {}
        for i in i_dic:
            if i in a_dic:
                temp = random.random()
                if temp < alpha:
                    new_o_dic[i] = t
        # update the self-quarantine states of nodes
        o_dic.update(new_o_dic)
        i_dic = del_dic_b_from_dic_a(i_dic, new_o_dic)

        # normalization check
        # normalization_check_mc_of_uau_sios(node_list=node_list, a_dic=a_dic, u_dic=u_dic, s_dic=s_dic, i_dic=i_dic, o_dic=o_dic)

        fac_u = len(u_dic) / node_num
        fac_a = len(a_dic) / node_num
        fac_s = len(s_dic) / node_num
        fac_i = len(i_dic) / node_num
        fac_o = len(o_dic) / node_num

        fac_list_s.append(fac_s)
        fac_list_i.append(fac_i)
        fac_list_u.append(fac_u)
        fac_list_a.append(fac_a)
        fac_list_o.append(fac_o)
        time_list.append(t)

        # save time
        if (fac_i - temp_i) < min_err and (fac_a - temp_a) < min_err and (t > min_step):
            # print('complete ahead of schedule at step:', t)
            # print(fac_a, fac_list_a[-1], fac_i, fac_list_i[-1], fac_o, fac_list_o[-1])
            return time_list, fac_list_a, fac_list_i, fac_list_o, fac_a, fac_i, fac_o
        temp_i = fac_i
        temp_a = fac_a
    # print(beta_u, beta_a, fac_a, fac_i, fac_o)
    return time_list, fac_list_a, fac_list_i, fac_list_o, fac_a, fac_i, fac_o


def mc_of_uau_m_sios_for_multi_process(net_a, net_b, initial_p, lamb, beta_u, beta_a, delta, alpha,
                                       sigma, m, mu_1, mu_2, max_step, min_step, min_err):
    node_list = list(net_a.nodes())
    node_num = len(node_list)

    #
    seed_list = []
    seed_num = int(node_num * initial_p)
    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)  # the range is from 0 to node_num-1 included
        seed = node_list[seed_index]
        # print('node list', node_list)
        # print('seed index', seed_index)
        # print('seed', seed)
        if seed not in seed_list:
            seed_list = seed_list + [seed]

    neigbor_dic_a = get_nei_dic(net_a)
    neigbor_dic_b = get_nei_dic(net_b)
    # initial the dicts
    u_dic = {}
    a_dic = {}
    i_dic = {}
    s_dic = {}
    o_dic = {}
    for node in node_list:
        u_dic[node] = 0
        if node in seed_list:
            i_dic[node] = 0
        else:
            s_dic[node] = 0

    # normalization check
    # normalization_check_mc_of_uau_sios(node_list=node_list, a_dic=a_dic, u_dic=u_dic, s_dic=s_dic, i_dic=i_dic, o_dic=o_dic)

    fac_list_s, fac_list_i, fac_list_u, fac_list_a, fac_list_o, time_list = [], [], [], [], [], []
    temp_i = 0
    temp_a = 0
    for t in range(max_step):
        # UAU process starts
        # UI/US -->lambda AI/AS. Note UO does not exist
        # temp dicts
        new_u_dic = {}
        new_a_dic = {}
        for i in a_dic:
            neighbor_list = neigbor_dic_a[i]
            for j in neighbor_list:
                if j in u_dic:
                    temp = random.random()
                    if temp < lamb:
                        new_a_dic[j] = t
        # AS/AI --> delta US/UI. Note AO can not change to UO
        for i in a_dic:
            if i not in o_dic:
                temp = random.random()
                if temp < delta:
                    new_u_dic[i] = t

        # update the UAU states of nodes
        a_dic.update(new_a_dic)
        a_dic = del_dic_b_from_dic_a(a_dic, new_u_dic)

        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)
        u_dic.update(new_u_dic)

        # mass-media process start
        for node in u_dic:
            temp = random.random()
            if temp < m:
                new_a_dic[node] = t
        # update the UAU states of nodes
        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)
        a_dic.update(new_a_dic)

        # SIOS process start
        # US -->betaU UI; AS -->betaA AI
        new_i_dic = {}
        new_s_from_i = {}
        new_s_from_o = {}
        for i in i_dic:
            neighbor_list = neigbor_dic_b[i]
            for j in neighbor_list:
                if j in s_dic:
                    if j in u_dic:
                        temp = random.random()
                        if temp < beta_u:
                            new_i_dic[j] = t
                    elif j in a_dic:
                        temp = random.random()
                        if temp < beta_a:
                            new_i_dic[j] = t
        # AI\UI -->mu_1 AS\US; AO --> mu_2 AS
        for i in i_dic:
            temp = random.random()
            if temp < mu_1:
                new_s_from_i[i] = t
        for i in o_dic:
            if i in a_dic:
                temp = random.random()
                if temp < mu_2:
                    new_s_from_o[i] = t
        # update the SIOS states of nodes
        s_dic.update(new_s_from_o)
        s_dic.update(new_s_from_i)
        s_dic = del_dic_b_from_dic_a(s_dic, new_i_dic)
        i_dic.update(new_i_dic)
        i_dic = del_dic_b_from_dic_a(i_dic, new_s_from_i)
        o_dic = del_dic_b_from_dic_a(o_dic, new_s_from_o)

        # self-awakening process start
        # UI --> sigma AI
        new_a_dic = {}
        for i in u_dic:
            if i in i_dic:
                temp = random.random()
                if temp < sigma:
                    new_a_dic[i] = t
        # update the self-awakening states of nodes
        a_dic.update(new_a_dic)
        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)

        # self-quarantine process start
        # AI --> alpha AO
        new_o_dic = {}
        for i in i_dic:
            if i in a_dic:
                temp = random.random()
                if temp < alpha:
                    new_o_dic[i] = t
        # update the self-quarantine states of nodes
        o_dic.update(new_o_dic)
        i_dic = del_dic_b_from_dic_a(i_dic, new_o_dic)

        # normalization check
        # normalization_check_mc_of_uau_sios(node_list=node_list, a_dic=a_dic, u_dic=u_dic, s_dic=s_dic, i_dic=i_dic, o_dic=o_dic)

        fac_u = len(u_dic) / node_num
        fac_a = len(a_dic) / node_num
        fac_s = len(s_dic) / node_num
        fac_i = len(i_dic) / node_num
        fac_o = len(o_dic) / node_num

        fac_list_s.append(fac_s)
        fac_list_i.append(fac_i)
        fac_list_u.append(fac_u)
        fac_list_a.append(fac_a)
        fac_list_o.append(fac_o)
        time_list.append(t)

        # save time
        if (fac_i - temp_i) < min_err and (fac_a - temp_a) < min_err and (t > min_step):
            # print('complete ahead of schedule at step:', t)
            # print(fac_a, fac_list_a[-1], fac_i, fac_list_i[-1], fac_o, fac_list_o[-1])
            return (fac_a, fac_i, fac_o)
        temp_i = fac_i
        temp_a = fac_a
    # print(beta_u, beta_a, fac_a, fac_i, fac_o)
    return (fac_a, fac_i, fac_o)


def mc_of_uau_m_sios_multi_process(pro_num=None, repeat_time=None, net_a=None, net_b=None, initial_p=None, lamb=None,
                                   beta_u=None, beta_a=None, delta=None, alpha=None,
                                   sigma=None, m=None, mu_1=None, mu_2=None, max_step=None, min_step=None,
                                   min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for i in range(repeat_time):
        result.append(po.apply_async(mc_of_uau_m_sios_for_multi_process,
                                     args=(net_a, net_b, initial_p, lamb, beta_u, beta_a, delta, alpha,
                                           sigma, m, mu_1, mu_2, max_step, min_step, min_err)))
    po.close()
    po.join()
    fac_a = 0
    fac_i = 0
    fac_o = 0
    for res in result:
        fac_a += res.get()[0]
        fac_i += res.get()[1]
        fac_o += res.get()[2]
    fac_a = fac_a / repeat_time
    fac_i = fac_i / repeat_time
    fac_o = fac_o / repeat_time
    return fac_a, fac_i, fac_o


# def uau_sios_update_parameters(parameter_list=None, para=None, sigma_list=None, alpha_list=None, rs_list=None,
#                                gamma_a_list=None, gamma_o_list=None):
#

