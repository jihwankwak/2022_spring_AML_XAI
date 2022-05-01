import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
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

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.inner_optim = optim.SGD(self.net.parameters(), lr=self.inner_lr)

        self.graph = True if self.args.trainer == 'maml' else False
        print('retain_graph init', self.graph)

        if self.args.dataset == 'omniglot':
            self.loss = nn.CrossEntropyLoss()
        elif self.args.dataset == 'sine':
            self.loss = nn.MSELoss()

    def _train_epoch(self, x_spt, y_spt, x_qry, y_qry):
        """
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
        losses = [0 for _ in range(self.inner_step + 1)]
        
        ##########################################################################################

        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
        b, setsz = x_spt.shape[0], x_spt.shape[1]
        querysz = x_qry.shape[1]
        
        for meta_idx in range(b):
            
            # first loop    
            logit = self.net(x_spt[meta_idx], vars=None)
            loss = self.loss(logit, y_spt[meta_idx])

            grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=self.graph, retain_graph=self.graph)
            updated_w = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, self.net.parameters())))
            
            with torch.no_grad():
                logit = self.net(x_qry[meta_idx], vars=None)
                loss = self.loss(logit, y_qry[meta_idx])

                if self.args.dataset == 'sine':
                    results[0] += loss
                elif self.args.dataset == 'omniglot':
                    losses[0] += loss
                    pred = F.softmax(logit, dim=1).argmax(dim=1)
                    correct = torch.eq(pred, y_qry[meta_idx]).sum().item()
                    # print(pred.shape)
                    # print(correct, querysz)
                    # print(ss)
                    results[0] += correct / querysz
            
            with torch.no_grad():
                logit = self.net(x_qry[meta_idx], vars=updated_w)
                loss = self.loss(logit, y_qry[meta_idx])

                if self.args.dataset == 'sine':
                    results[1] += loss
                elif self.args.dataset == 'omniglot':
                    losses[1] += loss
                    pred = F.softmax(logit, dim=1).argmax(dim=1)
                    correct = torch.eq(pred, y_qry[meta_idx]).sum().item()
                    results[1] += correct / querysz        
            
            for k in range(2, self.inner_step+1):
                logit = self.net(x_spt[meta_idx], vars=updated_w)
                
                loss = self.loss(logit, y_spt[meta_idx])

                grad = torch.autograd.grad(loss, updated_w, create_graph=self.graph, retain_graph=self.graph)
                updated_w = list(map(lambda w: w[1] - self.inner_lr * w[0], zip(grad, updated_w)))
                
                logit_qry = self.net(x_qry[meta_idx], updated_w)
                loss_qry = self.loss(logit_qry, y_qry[meta_idx])

                if self.args.dataset == 'sine':
                    results[k] += loss_qry
                elif self.args.dataset == 'omniglot':
                    losses[k] += loss_qry
                    pred = F.softmax(logit_qry, dim=1).argmax(dim=1)
                    correct = torch.eq(pred, y_qry[meta_idx]).sum().item()
                    results[k] += correct / querysz
            
        # if self.args.trainer == 'maml':

        self.meta_optim.zero_grad()
                    
        if self.args.dataset == 'sine':

            loss_final = results[-1] / b
            loss_final.backward()

            results = [ i.cpu().detach().numpy() for i in results]

        elif self.args.dataset == 'omniglot':

            loss_final = losses[-1] / b
            loss_final.backward()

        results = np.array(results) / b

        self.meta_optim.step()
        
        # elif self.args.trainer == 'fomaml':
            
        #     self.meta_optim.zero_grad()
        #     for w, g in zip(self.net.parameters(), grad):
        #         w.grad = g
        #     self.meta_optim.step()

        #     self.net()

        # for i in range(1, self.inner_step+1):
        #     results[i] = self.loss()

        # results = [ i.detach().numpy() for i in results ]
        # results = np.array(results.cpu().detach().numpy())
        
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
        results = [0 for _ in range(self.inner_step_test + 1)]
        
        ##########################################################################################
        
        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
        # print(ss)
        
        net = deepcopy(self.net)

        setsz = x_spt.shape[0]
        querysz = x_qry.shape[0]
            
        # first loop    
        logit = net(x_spt, vars=None)
        loss = self.loss(logit, y_spt)

        grad = torch.autograd.grad(loss, net.parameters())
        updated_w = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, net.parameters())))
            
        with torch.no_grad():
            logit = net(x_qry, vars=None)
            loss = self.loss(logit, y_qry)

            if self.args.dataset == 'sine':
                results[0] += loss
            elif self.args.dataset == 'omniglot':
                pred = F.softmax(logit, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y_qry).sum().item()
                results[0] += correct / querysz
            
        with torch.no_grad():
            logit = net(x_qry, vars=updated_w)
            loss = self.loss(logit, y_qry)

            if self.args.dataset == 'sine':
                results[1] += loss
            elif self.args.dataset == 'omniglot':
                pred = F.softmax(logit, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y_qry).sum().item()
                results[1] += correct / querysz      

        for k in range(2, self.inner_step_test+1):
            logit = net(x_spt, vars=updated_w)
                
            loss = self.loss(logit, y_spt)

            grad = torch.autograd.grad(loss, updated_w)
            updated_w = list(map(lambda w: w[1] - self.inner_lr * w[0], zip(grad, updated_w)))
            
            logit_qry = net(x_qry, updated_w)
            loss_qry = self.loss(logit_qry, y_qry)

            if self.args.dataset == 'sine':
                results[k] += loss_qry
            elif self.args.dataset == 'omniglot':
                pred = F.softmax(logit_qry, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y_qry).sum().item()
                results[k] += correct / querysz
                

        
        if self.args.dataset == 'sine':
            results = [ i.cpu().detach().numpy() for i in results]
        results = np.array(results)
        # results = np.array(results.cpu().detach().numpy()) / querysz
        # results = results / b
        ##########################################################################################
        
        return results


