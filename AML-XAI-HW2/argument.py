import argparse
import torch

def get_args():
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--trainer', default='maml', type=str,
                    choices=['maml', 'fomaml', 'reptile'], 
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset', default='omniglot', type=str,
                    choices=['omniglot','sine'], 
                    help='(default=%(default)s)')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--seed', type=int, help='seed number', default=0)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--inner_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--inner_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--inner_step_test', type=int, help='update steps for finetunning', default=10)
    
    argparser.add_argument('--batch-size', type=int, help='mini batch size for reptile only', default=5)
    args = argparser.parse_args()
    
    return args

