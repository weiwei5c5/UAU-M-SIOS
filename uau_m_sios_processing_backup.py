import math
import time
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import ticker

import uau_m_sios_backup_functions as us
from numba import njit
from scipy import optimize
from collections import namedtuple
import os

if __name__ == '__main__':

    start = time.time()

    # UAU paprameters
    lamb_, delta_ = 0.15, 0.6
    # mass-media process
    m_ = 0.5
    # SIOS parameters
    initial_p_, beta_u_, gamma_, mu_1_, mu_2_ = 0.2, 0.3, 0.5, 0.4, 0.8
    # joint parameters
    # sigma_, alpha_ = 1, 0.5
    sigma_, alpha_ = 0.2, 0.5
    # global parameter
    max_step_, min_step_, min_err_ = 1000, 200, 1e-12
    # picture parameter
    point = 20
    repeat_time = 200
    correlation = ['-1', '0', '1']
    # multiproess parameter
    pro_num_ = 5
    # pro_num_ = os.cpu_count() - 3

    lamb_list = []
    for i in range(1, point):
        lamb_list += [delta_ * i / point]

    beta_u_list = []
    for i in range(point + 1):
        beta_u_list += [1 * i / point]
    print(beta_u_list)

    plt.rcParams.update({'font.size': 12})

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    axs[0].set_position([0.13, 0.2, 0.1, 0.8])
    # plt.subplots_adjust(top=0.5)
    idx = 0
    for corr in correlation:
        title = corr + 'graphalpha=2.5ave=10A'
        graph_a, fname_a = us.read_graph(filename=title + '.txt', separator='\t')
        print(fname_a)

        title = corr + 'graphalpha=2.5ave=5B'
        graph_b, fname_b = us.read_graph(filename=title + '.txt', separator='\t')
        print(fname_b)

        # mmac multi-process
        mp_a_list_mmca, mp_i_list_mmca, mp_o_list_mmca = us.mmca_of_uau_m_sios_multi_process(pro_num=pro_num_,
                                                                                             beta_list=beta_u_list,
                                                                                             net_a=graph_a,
                                                                                             net_b=graph_b,
                                                                                             initial_p=initial_p_,
                                                                                             lamb=lamb_,
                                                                                             gamma=gamma_,
                                                                                             delta=delta_, alpha=alpha_,
                                                                                             sigma=sigma_, m=m_,
                                                                                             mu_1=mu_1_,
                                                                                             mu_2=mu_2_,
                                                                                             max_step=max_step_,
                                                                                             min_step=min_step_,
                                                                                             min_err=min_err_)
        # print(mp_a_list_mmca)
        # print(mp_i_list_mmca)
        # print(mp_o_list_mmca)
        print(str(beta_u_) + ':mmca multiprocess finished!')
        i_list_mmca = []
        a_list_mmca = []
        o_list_mmca = []
        i_list_mc = []
        a_list_mc = []
        o_list_mc = []
        i_list_mc_mp = []
        a_list_mc_mp = []
        o_list_mc_mp = []
        # f = open('uau_si3o_mu2' + 'check.txt', 'w')
        # # f = open('uau_sio_mu2' + 'check.txt', 'w')
        # f.write('beta_u' + '\t' + 'beta_a' + '\t' +
        #         'mmca_a' + '\t' + 'mmca_i' + 'mmca_o' + '\t' +
        #         'mc_a' + '\t' + 'mc_i' + '\t' + 'mc_o' + '\n')
        for beta_u_ in beta_u_list:
            # result = us.mmca_of_uau_sios_for_multi_process(graph_a, graph_b, initial_p_, lamb_, beta_u_,
            #                                                gamma_ * beta_u_, delta_,
            #                                                alpha_, sigma_, mu_1_, mu_2_, max_step_, min_step_, min_err_)
            # print(beta_u_, result)
            # time_list, fac_list_a, fac_list_i, fac_list_o, fac_a_mmca, fac_i_mmca, fac_o_mmca = us.mmca_of_uau_sios(
            #     net_a=graph_a, net_b=graph_b, initial_p=initial_p_, lamb=lamb_, beta_u=beta_u_, beta_a=gamma_ * beta_u_,
            #     delta=delta_, alpha=alpha_, sigma=sigma_, mu_1=mu_1_, mu_2=mu_2_, max_step=max_step_,
            #     min_step=min_step_, min_err=min_err_)
            # # print(title + ':mmca non-multiprocess finished!')
            # i_list_mmca += [fac_i_mmca]
            # a_list_mmca += [fac_a_mmca]
            # o_list_mmca += [fac_o_mmca]
            # print(fac_a_mmca, fac_i_mmca, fac_o_mmca)

            ave_a_mp, ave_i_mp, ave_o_mp = us.mc_of_uau_m_sios_multi_process(pro_num=pro_num_, repeat_time=repeat_time,
                                                                             net_a=graph_a, net_b=graph_b,
                                                                             initial_p=initial_p_, lamb=lamb_,
                                                                             beta_u=beta_u_, beta_a=gamma_ * beta_u_,
                                                                             delta=delta_, sigma=sigma_, alpha=alpha_,
                                                                             m=m_,
                                                                             mu_1=mu_1_, mu_2=mu_2_, max_step=max_step_,
                                                                             min_step=min_step_, min_err=min_err_)
            print(str(beta_u_) + ':mc multiprocess finished!')

            i_list_mc_mp += [ave_i_mp]
            a_list_mc_mp += [ave_a_mp]
            o_list_mc_mp += [ave_o_mp]
        #
        #     # ave_a, ave_i, ave_o = us.average_mc_of_uau_sios(repeat_time=repeat_time, net_a=graph_a, net_b=graph_b,
        #     #                                                 initial_p=initial_p_, lamb=lamb_, beta_u=beta_u_,
        #     #                                                 beta_a=gamma_ * beta_u_,
        #     #                                                 delta=delta_, alpha=alpha_, sigma=sigma_, mu_1=mu_1_,
        #     #                                                 mu_2=mu_2_, max_step=max_step_,
        #     #                                                 min_step=min_step_, min_err=min_err_)
        #     # print(title + ':mc non-multiprocess finished!')
        #     #
        #     # i_list_mc += [ave_i]
        #     # a_list_mc += [ave_a]
        #     # o_list_mc += [ave_o]
        #
        #     # print(beta_u_, ave_i, ave_i_mp, ave_a, ave_a_mp, ave_o, ave_o_mp)
        #     f.write(str(beta_u_) + '\t' + str(beta_u_*gamma_) + '\t' +
        #             str(fac_a) + '\t' + str(fac_i) + '\t' + str(fac_o) + '\t' +
        #             str(ave_a) + '\t' + str(ave_i) + '\t' + str(ave_o) + '\n')
        # f.close()
        # plt.plot(time_list, fac_list_i, 'ro')
        # plt.plot(time_list, fac_list_a, 'y*')
        # plt.plot(time_list, fac_list_o, 'g^')
        # plt.plot(beta_u_list, i_list_mmca, 'r-', label='ρI_mmca')
        # plt.plot(beta_u_list, a_list_mmca, 'y-', label='ρA_mmca')
        # plt.plot(beta_u_list, o_list_mmca, 'g-', label='ρO_mmca')

        # axs[idx].tick_params(axis='both', which='both', direction='in', top='on', bottom='on', left='on', right='on')

        ax = axs[idx]
        ax.tick_params(axis='both', which='both', direction='in')
        ax.plot(beta_u_list, mp_i_list_mmca, 'r-', label=r'$\rho^I_{MMCA}$ ')
        ax.plot(beta_u_list, mp_a_list_mmca, 'y-', label=r'$\rho^A_{MMCA}$ ')
        ax.plot(beta_u_list, mp_o_list_mmca, 'g-', label=r'$\rho^O_{MMCA}$ ')
        # plt.plot(beta_u_list, i_list_mc, 'ro', label='ρI_mc')
        # plt.plot(beta_u_list, a_list_mc, 'y*', label='ρA_mc')
        # plt.plot(beta_u_list, o_list_mc, 'g^', label='ρO_mc')
        ax.plot(beta_u_list, i_list_mc_mp, 'ro', label=r'$\rho^I_{MC}$')
        ax.plot(beta_u_list, a_list_mc_mp, 'y*', label=r'$\rho^A_{MC}$')
        ax.plot(beta_u_list, o_list_mc_mp, 'g^', label=r'$\rho^O_{MC}$')
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\rho$')
        if idx == 0:
            ax.set_title(r"(a) $r_s$ = {0}".format(corr))
            # ax.legend(loc='lower left', bbox_to_anchor=(0, 1.2), ncol=6)
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6)

        elif idx == 1:
            ax.set_title(r"(b) $r_s$ = {0}".format(corr))
        else:
            ax.set_title(r"(c) $r_s$ = {0}".format(corr))

        # 移位置 设为原点相交
        # ax.set_xlim([0, 1])
        # ax.spines['bottom'].set_position(('data', 0))
        ax.xaxis.set_major_locator(ticker.FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1]))
        ax.xaxis.set_minor_locator(ticker.FixedLocator([_ / 10 for _ in range(1, 10)]))
        ax.set_xticklabels(['{:g}'.format(x) if int(x) == x else '{:.1f}'.format(x) for x in ax.get_xticks()])

        # 设置纵坐标范围
        # ax.set_ylim([0, 1])
        # ax.spines['left'].set_position(('data', 0))
        ax.yaxis.set_major_locator(ticker.FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1]))
        ax.yaxis.set_minor_locator(ticker.FixedLocator([_ / 10 for _ in range(1, 10)]))
        ax.set_yticklabels(['{:g}'.format(x) if int(x) == x else '{:.1f}'.format(x) for x in ax.get_yticks()])

        idx += 1
        # plt.savefig(corr + 'checko_mu2_mp.png')
    # 调整子图之间的间距
    plt.tight_layout()
    plt.savefig('total_mc_mmca_mp.pdf', bbox_inches='tight', dpi=300)
    plt.close()
        # plt.savefig('checko_mu2.png')





