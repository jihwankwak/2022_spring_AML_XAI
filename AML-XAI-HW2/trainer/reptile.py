import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    copy import deepcopy
import trainer


class Trainer(trainer.GenericTrainer):
    """
    Meta Learner
    """
    def __init__(self, model, args):
        """

        :param args:
        """
        super(Trainer, self).__init__(model, args)
        self.inner_optim = optim.Adam(self.net.parameters(), lr=self.inner_lr, betas=(0,0.999))        
        
        if self.args.dataset == 'omniglot':
            self.loss = nn.CrossEntropyLoss()
        elif self.args.dataset == 'sine':
            self.loss = nn.MSELoss()
        
    def _train_epoch(self, x_spt, y_spt, x_qry, y_qry):
        """
        You should combine x_spt and y_spt to make data loader that produces mini-batches
        :param x_spt:   [b, setsz, c_, h, w] or [b, setsz, 1]
         - Training input data
        :param y_spt:   [b, setsz] or [b, setsz, 1]
         - Training target data
        :param x_qry:   [b, querysz, c_, h, w] or [b, setsz, 1]
         - Test input data
        :param y_qry:   [b, querysz] or [b, setsz, 1]
         - Test target data
        :return: 'results' (a list)
        """
        
        # results for meta-training
        # Sine wave: MSE loss for all tasks
        # Omniglot: Average accuracy for all tasks
        # In a list 'results', it should contain MSE loss or accuracy computed at each inner loop step.
        # The components in 'results' are as follows:
        # results[0]: results for pre-update model
        # results[1:]: results for the adapted model at each inner loop step
        results = [0 for _ in range(self.inner_step + 1)]
        
        x = torch.cat((x_spt, x_qry),1)
        y = torch.cat((y_spt, y_qry),1)
        
        ##########################################################################################
        
        # Write your code here
        
        ##########################################################################################

        return results


    def _finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w] or [setsz, 1]
         - Training input data
        :param y_spt:   [setsz] or [setsz, 1]
         - Training target data
        :param x_qry:   [querysz, c_, h, w] or [querysz, 1]
         - Test input data
        :param y_qry:   [querysz] or [querysz, 1]
         - Test target data
        :return: 'results' (a list)
        """
        
        # results for meta-test
        # Sine wave: MSE loss for current task
        # Omniglot: Average accuracy for current task
        # In a list 'results', it should contain MSE loss or accuracy computed at each inner loop step.
        # The components in 'results' are as follows:
        # results[0]: results for pre-update model
        # results[1:]: results for the adapted model at each inner loop step
        results = [0 for _ in range(self.inner_step + 1)]
        
        ##########################################################################################
        
        # Write your code here
        
        ##########################################################################################
        
        return results
