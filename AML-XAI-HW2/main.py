import  torch, os
import  numpy as np
import data_handler
from argument import get_args
import trainer
import networks
import pickle

def main():
    args = get_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    log_name = '{}_{}_{}_{}_inner_lr_{}_meta_lr_{}_step_{}_{}_k_qry_{}_k_spt_{}_task_num_{}'.format(args.dataset, args.trainer,args.seed, args.epoch,
                                                                           args.inner_lr, args.meta_lr, args.inner_step, args.inner_step_test, args.k_qry, args.k_spt, args.task_num)
    args.output = './result_data/' + log_name

    print(args)

    device = torch.device('cuda')  

    args.device = device
    myModel = networks.ModelFactory.get_model(args.dataset, args.n_way).to(device)

    myTrainer = trainer.TrainerFactory.get_trainer(myModel, args)
    dataloader = data_handler.get_dataset(args)
    myTrainer.train(dataloader, args.epoch)
    # result
    with open(args.output, "wb") as f:
        pickle.dump(myTrainer.result, f)
    # model
    torch.save(myModel.state_dict(), './trained_model/' + log_name)
    print('done!')

if __name__ == '__main__':
    main()
