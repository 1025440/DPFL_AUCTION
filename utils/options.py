
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")#客户端的占比
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")#每轮学习率衰减多少
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size")
    parser.add_argument('--real_clients', type=int, default=100, help="the real number of clients")#用在不同数量客户这里，就是实际参与的
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--dp_mechanism', type=str, default='Gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=2,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='differential privacy clip')
    parser.add_argument('--dp_sample', type=float, default=1, help='sample rate for moment account')#矩会计的采样率
    parser.add_argument('--dp_frac', type=float, default=1, help="the fraction of noise")  # 改变噪声

    parser.add_argument('--serial', action='store_true', help='partial serial running to save the gpu memory')#部分串行以节省gpu
    parser.add_argument('--serial_bs', type=int, default=128, help='partial serial running batch size')

    parser.add_argument('--B',type=float,default=100,help="budget")
    parser.add_argument('--alpha', type=float, default=1, help="alpha")
    parser.add_argument('--C', type=float, default=1, help="cost per unit")
    parser.add_argument('--other',type=str,default="none")
    parser.add_argument('--baseline', type=int, default="1", help="1 mine 2 random 3 cost 4 privacy ")

    args = parser.parse_args()
    return args
