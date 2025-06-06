
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from opacus.grad_sample import GradSampleModule
from utils.client_selection import Client_Selection


if __name__ == '__main__':

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None


    if args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar100':
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar100', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar100', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)

    img_size = dataset_train[0][0].shape

    net_glob = None

    if args.model == 'cnn' and(args.dataset == 'cifar' or args.dataset == 'cifar100'):
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')


    if args.dp_mechanism != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()


    w_glob = net_glob.state_dict()
    all_clients = list(range(args.real_clients))
    print("all_clients:",all_clients)


    acc_test = []
    loss_test=[]
    time_test=[]
    leakage_all=[]
    leakage_round=[]

    clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]

    B_every = args.B/args.epochs
    print("every epoch B:",B_every)
    B_rest=0
    social_welfare=[]
    U_number=[]
    t_start = time.time()
    profit_all=[]
    cost_all=[]
    cost_user_count=0
    payment_all=[]
    A=args.real_clients*[0]
    flag = 0
    for iter in range(args.epochs):
        print(" global epoch:",iter)
        w_locals, loss_locals, weight_locols = [], [], []

        B = B_every
        print("B:",B)
        U = list(range(args.real_clients))
        C=[random.uniform(args.C-0.001, args.C+0.001) for _ in range(len(all_clients))]
        E = [random.uniform(args.dp_epsilon -0.5, args.dp_epsilon + 0.5) for _ in range(len(all_clients))]

        alpha = args.alpha

        if(args.baseline == 1):
            idxs_users, payment, B_rest, E_rest, social_welfare_epoch = Client_Selection(B, U, C, E, A,iter, alpha)
        # elif(args.baseline == 2):
        #     idxs_users, payment, B_rest, E_rest, social_welfare_epoch =Random_selection(B, U, C, E, A,iter, alpha)
        # elif (args.baseline == 3):
        #     idxs_users, payment, B_rest, E_rest, social_welfare_epoch =Cost_selection(B, U, C, E, A,iter, alpha)
        # elif (args.baseline == 4):
        #     idxs_users, payment, B_rest, E_rest, social_welfare_epoch =Privacy_selection(B, U, C, E, A, iter,alpha)
        print("idxs_users:", idxs_users)
        print("num_choice_users:", len(idxs_users))
        print("payment:", payment)
        for i in idxs_users:
            A[i]=iter

        for i in payment:
            payment_all.append(i)

        print("B_rest:",B_rest)
        print("social_welfare_epoch:",social_welfare_epoch)
        U_number.append(len(idxs_users))
        if(len(social_welfare)!=0):
            temp=social_welfare.pop()
            social_welfare.append(temp)
            social_welfare.append(temp+social_welfare_epoch)
        else:
            social_welfare.append(social_welfare_epoch)


        t_start = time.time()


        for idx in idxs_users:
            local = clients[idx]
            print("idx:", idx)
            w, loss, leakage = local.train(net=copy.deepcopy(net_glob).to(args.device), E=E[idx] * args.dp_frac)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))
            leakage_all.append(leakage)


        w_glob = FedWeightAvg(w_locals, weight_locols)

        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        t_end = time.time()
        print("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)
        time_test.append(t_end - t_start)
        if len(time_test)>0:
            time_test.append(time_test[-1]+(t_end-t_start))
        else:
            time_test.append(t_end-t_start)



# ##############ACC#############################
#     rootpath = './log/acc'
#     if not os.path.exists(rootpath):
#         os.makedirs(rootpath)
#     accfile = open(rootpath + '/accfile_fed_{}_{}_{}_{}_iid{}_dp_{}_B_{}_epsilon_{}_other_{}_numusers{}.dat'.
#                    format(args.baseline, args.dataset, args.model, args.epochs, args.iid,
#                           args.dp_mechanism, args.B, args.dp_epsilon, args.other, args.num_users), "w")
#
#     for ac in acc_test:
#         sac = str(ac)
#         accfile.write(sac)
#         accfile.write('\n')
#     accfile.close()
#
#     # plot acc curve
#     plt.figure()
#     plt.plot(range(len(acc_test)), acc_test)
#     plt.ylabel('test accuracy')
#     plt.show()
#     plt.savefig(rootpath + '/fed_{}_{}_{}_{}_iid{}_dp_{}_B_{}_epsilon_{}_other_{}_acc.png'.format(
#         args.baseline, args.dataset, args.model, args.epochs,  args.iid, args.dp_mechanism, args.B, args.dp_epsilon, args.other))


# ##############OBJECTIVE FUNCTION#############################
#     rootpath = './log/welfare'
#     if not os.path.exists(rootpath):
#         os.makedirs(rootpath)
#     welfarefile = open(rootpath + '/welfare_fed_{}_{}_{}_{}_iid{}_dp_{}_B_{}_epsilon_{}_other_{}_numusers_{}.dat'.
#                    format(args.baseline, args.dataset, args.model, args.epochs, args.iid,
#                           args.dp_mechanism, args.B, args.dp_epsilon, args.other, args.num_users), "w")
#
#     for sw in social_welfare:
#         ssw = str(sw)
#         welfarefile.write(ssw)
#         welfarefile.write('\n')
#     welfarefile.close()
#
#     # plot social_welfare curve
#     plt.figure()
#     plt.plot(range(len(social_welfare)), social_welfare)
#     plt.ylabel('social welfare')
#     plt.show()
#     plt.savefig(rootpath + '/welfare_fed_{}_{}_{}_{}_iid{}_dp_{}_B_{}_epsilon_{}_other_{}_acc.png'.format(
#         args.baseline, args.dataset, args.model, args.epochs, args.iid, args.dp_mechanism, args.B, args.dp_epsilon, args.other))

# ##############LOSS#############################
#     rootpath = './log/loss'
#     if not os.path.exists(rootpath):
#         os.makedirs(rootpath)
#     lossfile = open(rootpath + '/lossfile_fed_{}_{}_{}_{}_iid{}_dp_{}_B_{}_epsilon_{}_other_{}_numusers_{}.dat'.
#                    format(args.baseline, args.dataset, args.model, args.epochs, args.iid,
#                           args.dp_mechanism, args.B, args.dp_epsilon, args.other, args.num_users), "w")
#
#     for ls in loss_test:
#         sls = str(ls)
#         lossfile.write(sls)
#         lossfile.write('\n')
#     lossfile.close()
#
#     # plot acc curve
#     plt.figure()
#     plt.plot(range(len(loss_test)), loss_test)
#     plt.ylabel('test loss')
#     plt.show()
#     plt.savefig(rootpath + '/timeloss_fed_{}_{}_{}_{}_iid{}_dp_{}_B_{}_epsilon_{}_other_{}_acc.png'.format(
#         args.baseline, args.dataset, args.model, args.epochs,  args.iid, args.dp_mechanism, args.B, args.dp_epsilon, args.other))

# ##############PRIVACY#############################
#     rootpath = './log/leakage'
#     if not os.path.exists(rootpath):
#         os.makedirs(rootpath)
#     roundleakagefile = open(rootpath + '/roundleakagefile_fed_{}_{}_{}_{}_iid{}_dp_{}_B_{}_epsilon_{}_other_{}_numusers_{}.dat'.
#                    format(args.baseline, args.dataset, args.model, args.epochs, args.iid,
#                           args.dp_mechanism, args.B, args.dp_epsilon, args.other, args.num_users), "w")
#
#     for lr in leakage_round:
#         slr = str(lr)
#         roundleakagefile.write(slr)
#         roundleakagefile.write('\n')
#     roundleakagefile.close()





