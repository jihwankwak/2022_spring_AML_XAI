import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset,n_way=5):
        
        if dataset == 'omniglot':
            import networks.network as net
            return net.conv_net(n_way)
        elif dataset == 'sine':
            import networks.network_regression as net
            return net.mlp(1,1)
        
