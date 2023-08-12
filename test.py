import os
import random
import time
import logging
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from hireMLP_v_patch import HireMLP
from DataLoaderUniversal_S2 import get_data_loader
from Parameters_S2 import parse_args
from Utils_S2 import  topk_, to_gpu,SAM,rmse,SIMSE,calc_metrics
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
from DataLoaderLocal_S2 import mosi_r2c_7, pom_r2c_7, r2c_2, r2c_7
from transformers import  BertTokenizer,RobertaTokenizer, RobertaModel
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D

random.seed(42)
torch.manual_seed(42)            # 为CPU设置随机种子
torch.cuda.manual_seed(42)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(42)   # 为所有GPU设置随机种子

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
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
####bert
    # bert_details = bert_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)
    # bert_sentences = to_gpu(torch.LongTensor(bert_details["input_ids"]))
    # bert_sentence_types = to_gpu(torch.LongTensor(bert_details["token_type_ids"]))
    # bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_details["attention_mask"]))
    # bert_sentence_details_list = [bert_sentences, bert_sentence_types, bert_sentence_att_mask]
#####robert
    rbt_details = roberta_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)
    rbt_sentences = to_gpu(torch.LongTensor(rbt_details["input_ids"]))
    rbt_sentence_att_mask = to_gpu(torch.LongTensor(rbt_details["attention_mask"]))
    rbt_sentence_details_list = [rbt_sentences, rbt_sentence_att_mask]
########################################################
    target = [" ".join(sample) for sample in a_data]
    rbt_details_target = roberta_tokenizer.batch_encode_plus(target, add_special_tokens=True, padding=True)
    rbt_target = to_gpu(torch.LongTensor(rbt_details_target["input_ids"]))
    rbt_target_att_mask = to_gpu(torch.LongTensor(rbt_details_target["attention_mask"]))
    rbt_target_details_list = [rbt_target, rbt_target_att_mask]

    outputs,embedding = model(rbt_sentence_details_list, rbt_target_details_list, v_data)
    return outputs,embedding


def get_loss_from_loss_func(outputs, labels, loss_func, opt):
    predictions = outputs
    task_loss = loss_func[0]
    labels= torch.as_tensor(labels[0]).to(device).long()
    # Get loss from predictions
    if opt.loss in ['CEL','RMSE', 'MAE', 'MSE', 'SIMSE']:
        # loss = task_loss(predictions.reshape(-1, ), labels.reshape(-1, ))
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
    elif opt.task == 'regression':
        predictions_corr, labels_corr = predictions_corr.reshape(-1,), labels_corr.reshape(-1,)
        mae = mean_absolute_error(labels_corr, predictions_corr)
        corr, _ = pearsonr(predictions_corr, labels_corr )

        if opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50']:
            if 'mosi' in opt.dataset:
                predictions_corr_7 = [mosi_r2c_7(p) for p in predictions_corr]
                labels_corr_7 = [mosi_r2c_7(p) for p in labels_corr]
            else:
                predictions_corr_7 = [r2c_7(p) for p in predictions_corr]
                labels_corr_7 = [r2c_7(p) for p in labels_corr]

            predictions_corr_2 = [r2c_2(p) for p in predictions_corr]
            labels_corr_2 = [r2c_2(p) for p in labels_corr]
            acc_7 = accuracy_score(labels_corr_7, predictions_corr_7)
            acc_2 = accuracy_score(labels_corr_2, predictions_corr_2)
            f1_2 = f1_score(labels_corr_2, predictions_corr_2, average='weighted')
            f1_7 = f1_score(labels_corr_7, predictions_corr_7, average='weighted')

            return {
                'mae': mae,
                'corr': corr,
                '7-cls_acc': acc_7,
                '2-cls_acc': acc_2,
                '7-f1': f1_7,
                '2-f1': f1_2,
            }
        elif opt.dataset in ['absa']:

            predictions_corr_2 = [r2c_2(p) for p in predictions_corr]
            labels_corr_2 = [r2c_2(p) for p in labels_corr]
            # acc_7 = accuracy_score(labels_corr_7, predictions_corr_7)
            acc_2 = accuracy_score(labels_corr_2, predictions_corr_2)
            f1_2 = f1_score(labels_corr_2, predictions_corr_2, average='weighted')
            # f1_7 = f1_score(labels_corr_7, predictions_corr_7, average='weighted')

            return {
                '3-cls_acc': acc_2,
                '3-f1': f1_2,
            }
        elif opt.dataset in ['pom_SDK', 'pom']:
            predictions_corr_7 = [pom_r2c_7(p) for p in predictions_corr]
            labels_corr_7 = [pom_r2c_7(p) for p in labels_corr]
            acc_7 = accuracy_score(labels_corr_7, predictions_corr_7)
            f1_7 = f1_score(labels_corr_7, predictions_corr_7, average='weighted')
            return {
                'mae': mae,
                'corr': corr,
                '7-cls_acc': acc_7,
                '7-f1': f1_7,
            }
        elif opt.dataset in ['mmmo', 'mmmov2']:
            predictions_corr_2 = [int(p>=3.5) for p in predictions_corr]
            labels_corr_2 = [int(p>=3.5) for p in labels_corr]
            acc_2 = accuracy_score(labels_corr_2, predictions_corr_2)
            f1_2 = f1_score(labels_corr_2, predictions_corr_2, average='weighted')

            return {
                'mae': mae,
                'corr': corr,
                '2-cls_acc': acc_2,
                '2-f1': f1_2,
            }
        else:
            raise NotImplementedError
    else :
        raise NotImplementedError


#####absa_train
def train(train_loader, model, loss_func,optimizer, opt):
    model.train()
    running_loss, predictions_corr, targets_corr = 0.0, [], []

    for iter, datas in enumerate(train_loader):
        # count=(iter+1)*opt.batch_size
        t_data, a_data, v_data = datas[0], datas[1], datas[2].cuda().float()
        labels = get_labels_from_datas(datas, opt) # Get multiple labels
        # targets = get_loss_label_from_labels(labels, opt).cuda() # Get target from labels

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
       #####每个epoch输出损失信息
        # print('iter:', count,'loss:',running_loss/count, )

    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr)
    train_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict

    return running_loss/len(train_loader), train_score



def val(val_loader, model, opt):
    dir = r'D:\lilili\HireMLPJL\Modify_Similarity\log\top_k_d_common=256\2015_patch(100)_robert_dropout-0.1_256\model\best_model.pth'
    model.load_state_dict(torch.load(dir))
    model.eval()
    predictions_corr, targets_corr,embedding_corr = [], [],[]
    with torch.no_grad():
        for _, datas in enumerate(val_loader):
            t_data, a_data, v_data = datas[0], datas[1], datas[2].cuda().float()
            labels = get_labels_from_datas(datas, opt) # Get multiple labels
            # targets = get_loss_label_from_labels(labels, opt).cuda() # Get target from labels
            outputs,embedding_out = get_outputs_from_datas(model, t_data, a_data, v_data, opt) # Get multiple outputs
            predictions_corr += outputs.cpu().numpy().tolist()
            targets_corr += labels[0].cpu().numpy().tolist()
            embedding_corr += embedding_out.cpu().numpy().tolist()
    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr)
    valid_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict

    return predictions_corr,targets_corr,embedding_corr, valid_score


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
        optimizer = torch.optim.Adam(params, lr=float(opt.learning_rate), weight_decay=0.0)
    elif opt.optm == "SGD":
        optimizer = torch.optim.SGD(params, lr=float(opt.learning_rate), weight_decay=0.0, momentum=0.9 )
    elif opt.optm == "SAM":
        optimizer = SAM(params, torch.optim.Adam, lr=float(opt.learning_rate), weight_decay=0.0,)
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

def list_to_array(x):
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(x)], axis=1)
    return dff.fillna(0).values.T.astype(int)

# 开始训练
if __name__ == '__main__':
    opt = parse_args()
    print(device)

###获取数据集
    train_dataloader, val_dataloader ,test_dataloader= get_data_loader(opt)
    model = HireMLP(opt).to(device)

##获取模型参数
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    time_start = time.time()
    print("训练开始时间",time_start)
    output,labels,embedding, test_acc= val(test_dataloader, model, opt)

    embedding = list_to_array(embedding)

    embedding = np.array(embedding)

    tsne = TSNE(n_components=2)
    features = tsne.fit_transform(embedding)
    label_name = [ 'negative', 'neutral','positive',]
    colors = [ 'red','yellow', 'blue']

    for feature, labelindex in zip(features, labels):
        plt.scatter(feature[0], feature[1], color=colors[int(labelindex)], s=2)


    for i in range(3):
        plt.scatter(feature[0], feature[1],color=colors[i],label=label_name[i], s=1)
        plt.legend(markerscale=5)
    #     plt.legend(prop = {'size':8})
    plt.title('TWITTER-15',font={'family':'Arial', 'size':18})


    plt.savefig("TWITTER_15_Represention.png", dpi=1024, bbox_inches="tight")
    plt.show()

    print(
          '=====test_acc:', test_acc, '\n')


