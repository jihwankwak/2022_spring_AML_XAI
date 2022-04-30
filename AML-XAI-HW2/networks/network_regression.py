import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



class mlp(nn.Module):
    def __init__(self, ninput=1, noutput=1):
        super(mlp, self).__init__()
        self.config = [
            ('linear', [40, ninput]),
            ('relu', [True]),
            ('linear', [40, 40]),
            ('relu', [True]),
            ('linear', [noutput, 40]),
        ]

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):

            if name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError        

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None):
        """
        :param x: [b, 1]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            
            if name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)

        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars    

