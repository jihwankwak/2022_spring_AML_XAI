import torch
import numpy as np
from torch.nn import functional as F
from copy import deepcopy
class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(myModel, args):
        
        if args.trainer == 'maml' or args.trainer == 'fomaml':
            import trainer.maml as trainer
        elif args.trainer == 'reptile':
            import trainer.reptile as trainer
        
        return trainer.Trainer(myModel, args)
    
class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, model, args):
        
        self.net = model
        self.args = args
        
        self.device = args.device
        # N way K shots
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        
        self.epoch = args.epoch
        self.inner_lr = args.inner_lr        
        self.meta_lr = args.meta_lr
        self.inner_step = args.inner_step
        self.inner_step_test = args.inner_step_test

        self.result = {}
        self.result['training acc'] = []
        self.result['test acc'] = []
        self.result['ground'] = []
        self.result['prediction'] = []
        self.result['qry'] = []
    
    def train(self, dataloader, epoch):
        eval_flag = True
        for step in range(epoch):

            print(step)
            x_spt, y_spt, x_qry, y_qry = dataloader.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(self.device), torch.from_numpy(y_spt).to(self.device), \
            torch.from_numpy(x_qry).to(self.device), torch.from_numpy(y_qry).to(self.device)
            accs = self._train_epoch(x_spt, y_spt, x_qry, y_qry)
            eval_flag = False

            self.result['training acc'].append(accs)
            
            if step % 50 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 100 == 0:
                accs = []
                for _ in range(1000//self.task_num):
                    # test
                    x_spt, y_spt, x_qry, y_qry = dataloader.next('test')
                    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(self.device), torch.from_numpy(y_spt).to(self.device), \
                    torch.from_numpy(x_qry).to(self.device), torch.from_numpy(y_qry).to(self.device)

                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        test_acc = self._finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        accs.append( test_acc )

                # [b, inner_step+1]
                accs = np.array(accs).mean(axis=0).astype(np.float16)
                self.result['test acc'].append(accs)
                print('Test acc:', accs)
            
    def _train_epoch(self, x_spt, y_spt, x_qry, y_qry):
        pass

    def _finetunning(self, x_spt, y_spt, x_qry, y_qry):
        pass
