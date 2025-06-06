

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity,   Gaussian_Simple
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(self.args.dp_sample * len(idxs)), replace=False)

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.idxs = idxs
        self.times = self.args.epochs * self.args.frac
        self.lr = args.lr
        self.noise_scale=0

    def calculate_noise_scale(self,E):
       if self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = E
            delta_single_query = self.args.dp_delta
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)

    def train(self, net, E):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_client = 0

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()

                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                loss.backward()
                if self.args.dp_mechanism != 'no_dp':
                    self.clip_gradients(net)
                optimizer.step()
                scheduler.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        leakage = 0
        if self.args.dp_mechanism != 'no_dp':
            self.noise_scale = self.calculate_noise_scale(E)
            print('E every epoch: ', E)
            print('Noise scale: ', self.noise_scale)
            leakage = self.add_noise(net)

        self.lr = scheduler.get_last_lr()[0]

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), leakage



    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Gaussian' :
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):

        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs))
        sum_noise = 0
        for param in net.parameters():
            temp = np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                    size=param.grad.size())
            param.grad+=torch.from_numpy(temp).to(self.args.device)
            sum_noise += np.mean(np.absolute(temp))
        return sum_noise / 8





