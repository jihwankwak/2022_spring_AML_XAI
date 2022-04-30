from data_handler.omniglotNShot import OmniglotNShot
from data_handler.SineNShot import SineNShot




def get_dataset(args):
    name = args.dataset
    if name == 'omniglot':
        return OmniglotNShot('omniglot',
                   batchsz=args.task_num,
                   n_way=args.n_way,
                   k_shot=args.k_spt,
                   k_query=args.k_qry,
                   imgsz=args.imgsz,
                   trainer=args.trainer)
    elif name == "sine":
        return SineNShot(batchsz=args.task_num,
                         k_shot=args.k_spt,
                         k_query=args.k_qry)
