import os
import random
import time
import logging
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from RR_Mixer import RR_Mixer_Module
from DataLoaderUniversal import get_data_loader
from Parameters import parse_args
from Utils import topk_,to_gpu,SAM,rmse,SIMSE
from sklearn.metrics import accuracy_score, f1_score
from transformers import RobertaTokenizer

random.seed(42)
torch.manual_seed(42)            # 为CPU设置随机种子
torch.cuda.manual_seed(42)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(42)   # 为所有GPU设置随机种子

# Roberta
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

def get_labels_from_datas(datas, opt):
    if 'SDK' in opt.dataset:
        return datas[3:-1]
    else:
        return datas[3:]


def get_loss_label_from_labels(labels, opt):
    if opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50','absa']:
        if opt.task == 'regression':
            labels = labels[0]
        elif opt.task == 'classification' and opt.num_class==3:
            labels = labels[1]
        elif opt.task == 'classification' and opt.num_class==7:
            labels = labels[2]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return labels

def get_outputs_from_datas(model, t_data, a_data, v_data, opt):
    sentences = [" ".join(sample) for sample in t_data]
    rbt_details = roberta_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)
    rbt_sentences = to_gpu(torch.LongTensor(rbt_details["input_ids"]))
    rbt_sentence_att_mask = to_gpu(torch.LongTensor(rbt_details["attention_mask"]))
    rbt_sentence_details_list = [rbt_sentences, rbt_sentence_att_mask]
    target = [" ".join(sample) for sample in a_data]
    rbt_details_target = roberta_tokenizer.batch_encode_plus(target, add_special_tokens=True, padding=True)
    rbt_target = to_gpu(torch.LongTensor(rbt_details_target["input_ids"]))
    rbt_target_att_mask = to_gpu(torch.LongTensor(rbt_details_target["attention_mask"]))
    rbt_target_details_list = [rbt_target, rbt_target_att_mask]
    outputs = model(rbt_sentence_details_list, rbt_target_details_list, v_data)
    return outputs


def get_loss_from_loss_func(outputs, labels, loss_func, opt):
    predictions = outputs
    task_loss = loss_func[0]
    labels= torch.as_tensor(labels[0]).to(device).long()
    # Get loss from predictions
    if opt.loss in ['CEL','RMSE', 'MAE', 'MSE', 'SIMSE']:
        loss = task_loss(predictions, labels.long())
    else:
        raise NotImplementedError
    return loss


def get_score_from_result(predictions_corr, labels_corr, opt):
    if opt.task == 'classification':
        if opt.num_class == 1:
            predictions_corr = np.int64(predictions_corr.reshape(-1,) > 0)
        else:
            _, predictions_corr = topk_(predictions_corr, 1, 1)
        predictions_corr, labels_corr = predictions_corr.reshape(-1,), labels_corr.reshape(-1,)
        acc = accuracy_score(labels_corr, predictions_corr)
        f1 = f1_score(labels_corr, predictions_corr, average='weighted')
        return {
            str(opt.num_class)+'-cls_acc': acc,
            str(opt.num_class)+'-f1': f1
        }
    else :
        raise NotImplementedError

def train(train_loader, model, loss_func,optimizer, opt):
    model.train()
    running_loss, predictions_corr, targets_corr = 0.0, [], []
    # weight_decay = 0.001
    for iter, datas in enumerate(train_loader):
        t_data, a_data, v_data = datas[0], datas[1], datas[2].cuda().float()
        labels = get_labels_from_datas(datas, opt) # Get multiple labels

        outputs = get_outputs_from_datas(model, t_data, a_data, v_data, opt) # Get multiple outputs
        loss = get_loss_from_loss_func(outputs, labels, loss_func, opt)  # Get loss
        optimizer.zero_grad()
        loss.backward()
        if opt.gradient_clip > 0:
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], opt.gradient_clip)
        optimizer.step()
        running_loss += loss.item()

        with torch.no_grad():
            predictions_corr += outputs.cpu().numpy().tolist()
            targets_corr += labels[0].cpu().numpy().tolist()
    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr)
    train_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict

    return running_loss/len(train_loader), train_score



def val(val_loader, model, loss_func, opt):
    model.eval()
    running_loss, predictions_corr, targets_corr = 0.0, [], []
    with torch.no_grad():
        for _, datas in enumerate(val_loader):
            t_data, a_data, v_data = datas[0], datas[1], datas[2].cuda().float()
            labels = get_labels_from_datas(datas, opt) # Get multiple labels
            outputs = get_outputs_from_datas(model, t_data, a_data, v_data, opt) # Get multiple outputs
            loss = get_loss_from_loss_func(outputs, labels, loss_func, opt)  # Get loss
            running_loss += loss.item()

            predictions_corr += outputs.cpu().numpy().tolist()
            targets_corr += labels[0].cpu().numpy().tolist()

    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr)
    valid_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict

    return running_loss/len(val_loader), valid_score


def get_optimizer(opt, model):
    if opt.bert_lr_rate <= 0:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        def get_berts_params(model):
            results = []
            for p in model.named_parameters():
                if 'bert' in p[0] and p[1].requires_grad:
                    results.append(p[1])
            return results
        def get_none_berts_params(model):
            results = []
            for p in model.named_parameters():
                if 'bert' not in p[0] and p[1].requires_grad:
                    results.append(p[1])
            return results
        params = [
            {'params': get_berts_params(model), 'lr': float(opt.learning_rate) * opt.bert_lr_rate},
            {'params': get_none_berts_params(model), 'lr': float(opt.learning_rate)},
        ]
    if opt.optm == "Adam":
        optimizer = torch.optim.Adam(params, lr=float(opt.learning_rate), weight_decay=opt.weight_decay)
    elif opt.optm == "SGD":
        optimizer = torch.optim.SGD(params, lr=float(opt.learning_rate), weight_decay=opt.weight_decay, momentum=0.9 )
    elif opt.optm == "SAM":
        optimizer = SAM(params, torch.optim.Adam, lr=float(opt.learning_rate), weight_decay=opt.weight_decay,)
    else:
        raise NotImplementedError

    if opt.lr_decrease == 'step':
        opt.lr_decrease_iter = int(opt.lr_decrease_iter)
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)
    elif opt.lr_decrease == 'multi_step':
        opt.lr_decrease_iter = list((map(int, opt.lr_decrease_iter.split('-'))))
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)
    elif opt.lr_decrease == 'exp':
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_decrease_rate)
    elif opt.lr_decrease == 'plateau':
        mode = 'min' # if opt.task == 'regression' else 'max'
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=int(opt.lr_decrease_iter), factor=opt.lr_decrease_rate,)
    else:
        raise NotImplementedError
    return optimizer, lr_schedule


def get_loss(opt):
    if opt.loss == 'CEL':
        loss_func = nn.CrossEntropyLoss()
    elif opt.loss == 'RMSE':
        loss_func = rmse
    elif opt.loss == 'MAE':
        loss_func = torch.nn.L1Loss()
    elif opt.loss == 'MSE':
        loss_func = torch.nn.MSELoss(reduction='mean')
    elif opt.loss == 'SIMSE':
        loss_func = SIMSE()
    else:
        raise NotImplementedError

    return [loss_func]


def build_message(epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score):
    msg = "Epoch:[{:3.0f}]".format(epoch + 1)

    msg += " ||"
    msg += " TrainLoss:[{0:.3f}]".format(train_loss)
    for key in train_score.keys():
        msg += " Train_" + key + ":[{0:6.3f}]".format(train_score[key])

    msg += " ||"
    msg += " ValLoss:[{0:.3f}]".format(val_loss)
    for key in val_score.keys():
        msg += " Val_" + key + ":[{0:6.3f}]".format(val_score[key])

    return msg
def log_tf_board(writer, epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score, lr_schedule):
    writer.add_scalar('Train/Epoch/Loss', train_loss, epoch)
    writer.add_scalar('Valid/Epoch/Loss', val_loss, epoch)
    writer.add_scalar('test/Epoch/Loss' , test_loss, epoch)
    for key in train_score.keys():
        writer.add_scalar('Train/Epoch/'+key, train_score[key], epoch)
    for key in val_score.keys():
        writer.add_scalar('Valid/Epoch/'+key, val_score[key], epoch)
    for key in test_score.keys():
        writer.add_scalar('test/Epoch/' + key, test_score[key], epoch)
    try:
        writer.add_scalar('Lr',  lr_schedule.get_last_lr()[-1], epoch)
    except:
        pass

if __name__ == '__main__':
    opt = parse_args()
    epoch =opt.epochs_num

    writer = SummaryWriter(os.path.join(opt.log_path))
    best_model_name_val = os.path.join(opt.ckpt_path, opt.task_name, "best_model_val.pth.tar")

    train_dataloader, val_dataloader ,test_dataloader= get_data_loader(opt)
    model = RR_Mixer_Module(opt).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer, lr_schedule = get_optimizer(opt, model)
    loss_fn = get_loss(opt)
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []

    min_acc = 0
    logging.log(msg="Start training...", level=logging.DEBUG)
    time_start = time.time()
    print("训练开始时间",time_start)
    for t in range(epoch):
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer,opt)
        val_loss, val_acc = val(val_dataloader, model, loss_fn,opt)
        print('====epoch:', t, '\n',
              '====train_loss:', train_loss,
              '====train_acc:', train_acc, '\n',
              '=====val_loss:', val_loss,
              '====val_acc:', val_acc, '\n'
              )
        test_loss, test_acc= val(test_dataloader, model, loss_fn, opt)
        print('=====test_loss:', test_loss,
              '=====test_acc:', test_acc, '\n')

        msg = build_message(t, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
        logging.log(msg=msg, level=logging.DEBUG)
        log_tf_board(writer, t, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, lr_schedule)

        loss_train.append(train_loss)
        acc_train.append(float(train_acc['3-cls_acc']))
        loss_val.append(val_loss)
        acc_val.append(float(val_acc['3-cls_acc']))

        # 保存最好的模型权重
        if float(test_acc['3-cls_acc'])> min_acc:
            folder = "sava_model"
            if not os.path.exists(folder):
                os.makedirs(folder)
            min_acc = float(test_acc['3-cls_acc'])
            print(f"save best model,第{t}轮")
            torch.save(model.state_dict(), 'sava_model/best_model.pth')
        # 保存最后一轮得到权重文件
        if t == epoch - 1:
            torch.save(model.state_dict(), 'sava_model/last_model.pth')
        time_epoch=time.time()
        print(f"第{t}轮总计用时：",time_epoch-time_start)
    writer.close()
    time_end = time.time()
    time_sum = time_end - time_start
