
import math
import sys

import numpy as np
import copy
from utils.options import args_parser
def ObjectiveFunction(U,C,E, A,iter,alpha):
    args = args_parser()
    v=0
    c=0
    if U!=None:
        for x in U:
            v+=E[x]
            c+= alpha *(iter- A[x])

    return math.log(1+v)+c


def payment(ui,U,C,E,B,V,V_sort, A,iter,alpha):
    k=-1
    U_first=[]
    for i in V_sort:

        U_first.append(i)
        U_first_temp=copy.deepcopy(U_first)
        U_first.remove(i)
        if i==ui: continue
        if (C[i]*E[i]<=B/2.0*(ObjectiveFunction(U_first_temp,C,E, A,iter,alpha)-ObjectiveFunction(U_first,C,E, A,iter,alpha))
                /ObjectiveFunction(U_first_temp,C,E, A,iter,alpha)):
            U_first.append(i)
            break
        k += 1
        U_first.append(i)

    P1=[]
    P2=[]
    P_min=[]
    U_first_P1 = []
    U_first_P2 = []

    for i in U_first:
        U_first_P1.append(ui)
        U_first_P1_ui=copy.deepcopy(U_first_P1)
        U_first_P1.remove(ui)
        U_first_P1.append(i)
        U_first_P1_i=copy.deepcopy(U_first_P1)
        U_first_P1.remove(i)
        pay_temp_1 = C[i]*E[i]*((ObjectiveFunction(U_first_P1_ui,C,E, A,iter,alpha)-ObjectiveFunction(U_first_P1,C,E, A,iter,alpha))
                            /((ObjectiveFunction(U_first_P1_i,C,E, A,iter,alpha)-ObjectiveFunction(U_first_P1,C,E, A,iter,alpha))))
        U_first_P1.append(i)
        P1.append(pay_temp_1)

    for i in U_first:
        U_first_P2.append(ui)
        U_first_P2_ui = copy.deepcopy(U_first_P2)
        U_first_P2.remove(ui)
        pay_temp_2 =B/2.0*((ObjectiveFunction(U_first_P2_ui,C,E, A,iter,alpha)-ObjectiveFunction(U_first_P2,C,E, A,iter,alpha))
                       /ObjectiveFunction(U_first_P2_ui,C,E, A,iter,alpha))
        U_first_P2.append(i)
        P2.append(pay_temp_2)


    for i in range(len(U_first)):
        P_min.append(min(P1[i], P2[i]))

    pay = max(P_min)
    cost=C[ui]*E[ui]

    pay_finally=max(pay,cost)
    return pay_finally


def Client_Selection(B,U_origin,C,E, A,iter,alpha):
    U=copy.deepcopy(U_origin)
# 初始化 St=∅
    S=[]
    P=[0]*len(C)
    socialwel_temp=[]

    while B > 0:
        V = [0] * len(C)
        for i in U:
            S.append(i)
            temp_S=copy.deepcopy(S)
            S.remove(i)

            if C[i] * E[i] <= B / 2.0 * (ObjectiveFunction(temp_S,C,E, A,iter,alpha)-ObjectiveFunction(S,C,E, A,iter,alpha)) / ObjectiveFunction(S, C, E, A, iter, alpha):
                V[i] = (ObjectiveFunction(temp_S, C, E, A, iter, alpha) - ObjectiveFunction(S, C, E, A, iter, alpha)) / (
                            C[i] * E[i])
            else :
                V[i]=-sys.float_info.max

        V_ndarry=np.array(V)
        V_sort_temp=np.argsort(-V_ndarry)
        V_sort=V_sort_temp.tolist()

        ui=V_sort[0]
        if V[i] >0:
            S.append(ui)
        else:
            break
        P[ui] = payment(ui, U, C, E, B, V, V_sort,  A,iter,alpha)
        B -= P[ui]
        U.remove(ui)

    while len(S)>0:
        socialwel_temp.append(ObjectiveFunction(S, C, E, A,iter, alpha))

    profit=[]
    for u in S:
        profit.append(P[u]-C[u]*E[u])
    return S, P, B, E, ObjectiveFunction(S, C, E, A,iter, alpha)


