from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)
        
        self.lamb=args.lamb
        self.loss = nn.CrossEntropyLoss()       

    def train(self, train_loader, test_loader, t, device = None):
        
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t

        self.t = t

        if t>0: # update fisher before starting training new task
            self.update_frozen_model()
            self.update_fisher()
        
        # Now, you can update self.t
        
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)
        
        
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output,target)

                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        
    def criterion(self,output,targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the regularization-based continual learning
        
        For the hyperparameter on regularization, please use self.lamb
        """
        
        #######################################################################################
        if self.t > 0:
            loss = self.loss(output, targets) + self.lamb/2 * self.fisher_loss()
            # print('loss', self.loss(output, targets), self.fisher_loss())
        else:
            loss = self.loss(output, targets)

        return loss
        
        #######################################################################################
    
    def fisher_loss(self):

        loss = 0

        for (n1, prev_model), (n2, cur_model) in zip(self.model_fixed.named_parameters(), self.model.named_parameters()):
            if cur_model.requires_grad == True:

                weight = (prev_model-cur_model).flatten()

                if 'last' not in n2:
                    # print(self.fisher[n2].shape)
                    # print(weight.shape)
                    # print(ss)
                    loss += torch.sum(weight.pow(2)*self.fisher[n2])

        return loss

    def compute_diag_fisher(self):
        """
        Arguments: None. Just use global variables (self.model, self.criterion, ...)
        Return: Diagonal Fisher matrix. 
        
        This function will be used in the function 'update_fisher'
        """


        fisher = {}
        criterion = nn.CrossEntropyLoss()
        
        # fisher init
        for name,param in self.model_fixed.named_parameters():
            fisher[name]=0*param.flatten()

        self.model.train()
        batch_count = 0

        for samples in tqdm(self.fisher_iterator):
            data, target = samples
            data, target = data.to('cuda'), target.to('cuda')
            batch_size = data.shape[0]

            self.model.zero_grad()

            output = self.model.forward(data)[self.t]
            loss_CE = criterion(output,target)
            loss_CE.backward()
            
            batch_count += batch_size

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += ((param.grad)**2).flatten() * batch_size
                    # print(fisher[name][0])
                    # if name == 'conv1.weight':
                    #     print('fisher', fisher[name])
        # print(batch_count)/
        # print('before', fisher)
        with torch.no_grad():
            for name, value in fisher.items():
                # print(name)
                # print(value.shape)
                fisher[name] = value/batch_count
                # if name == 'conv1.weight':
                #         print('fisher final', fisher[name])

        # print('after', fisher)
        # print(ss)/

        return fisher
        
        #######################################################################################        
    
    def update_fisher(self):
        
        """
        Arguments: None. Just use global variables (self.model, self.fisher, ...)
        Return: None. Just update the global variable self.fisher
        Use 'compute_diag_fisher' to compute the fisher matrix
        """
        
        #######################################################################################        

        # init
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad == True:
        #         self.fisher[name] = torch.zeros(param.flatten().shape) 
        if self.t == 1:
            self.fisher = self.compute_diag_fisher()
        elif self.t > 1:
            fisher_temp = self.fisher
            self.fisher = self.compute_diag_fisher()
            
            for name, param in self.model.named_parameters():
                self.fisher[name]=(self.fisher[name]+fisher_temp[name]*(self.t-1))/(self.t)
        
        #######################################################################################
