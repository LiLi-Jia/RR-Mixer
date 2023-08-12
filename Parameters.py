import argparse
from Utils import str2bools, str2floats

def parse_args():
    parser = argparse.ArgumentParser()

    # Names, paths, logs
    parser.add_argument("--ckpt_path", default="./ckpt")
    parser.add_argument("--log_path", default="./log")
    parser.add_argument("--task_name", default="test")

    # Data parameters
    parser.add_argument("--dataset", default='absa', type=str)
    parser.add_argument("--normalize", default='0-0-0', type=str2bools)
    parser.add_argument("--text", default='robert', type=str)
    parser.add_argument("--audio", default='robert', type=str)
    parser.add_argument("--video", default='ViT', type=str)
    parser.add_argument("--d_t", default=1024, type=int)
    parser.add_argument("--d_a", default=1024, type=int)
    parser.add_argument("--d_v", default=1024, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--persistent_workers", action='store_true')
    parser.add_argument("--pin_memory", action='store_true')
    parser.add_argument("--drop_last", action='store_true')
    parser.add_argument("--task", default='classification', type=str, choices=['classification', 'regression'])
    parser.add_argument("--num_class", default=3, type=int)
    #
    # # Model parameters
    parser.add_argument("--d_common", default=256, type=int)
    parser.add_argument("--encoders", default='lstm', type=str)
    parser.add_argument("--dropout", default='0.1-0.1-0.1-0.1', type=str2floats)

    # Training and optimization
    # parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--loss", default='CEL', choices=['CEL','Focal', 'CE', 'BCE', 'RMSE', 'MSE', 'SIMSE', 'MAE'])
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--epochs_num", default=30, type=int)
    parser.add_argument("--optm", default="Adam", type=str, choices=['SGD', 'SAM', 'Adam'])
    parser.add_argument("--bert_lr_rate", default=-1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--lr_decrease", default='plateau', type=str, choices=['multi_step', 'step', 'exp', 'plateau'])
    parser.add_argument("--lr_decrease_iter", default='50', type=str)
    parser.add_argument("--lr_decrease_rate", default=0.1, type=float)
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    args = parse_args()
    print(args)
