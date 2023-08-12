import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
import pandas as pd
DATA_PATH = r'XXXXXX'

def whether_type_str(data):
    return "str" in str(type(data))

# absa Structure
absa_l_features = ["robert"]
absa_a_features = ["robert"]
absa_v_features = ["ViT"]

def multi_collate_absa(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.Tensor([sample[3] for sample in batch]).reshape(-1, ).float()

    if whether_type_str(batch[0][0][0]):
        sentences = [sample[0].tolist() for sample in batch]
    else:
        sentences = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch], padding_value=0).transpose(0, 1)

    if whether_type_str(batch[0][1][0]):
        acoustic = [sample[1].tolist() for sample in batch]
    else:
        acoustic = pad_sequence([torch.FloatTensor(sample[1]) for sample in batch], padding_value=0).transpose(0, 1)

    visual = pad_sequence([torch.FloatTensor(sample[2]) for sample in batch], padding_value=0).transpose(0, 1)

    lengths = torch.LongTensor([sample[0].shape[0] for sample in batch])
    return sentences, acoustic, visual, labels, lengths

def get_absa_dataset(mode='train', text='bert', audio='bert', video='resnet',normalize=[True, True, True]):
    with open(os.path.join(DATA_PATH, 'jl_' + mode + '.tsv'), 'rb') as f:
        data = pd.read_table(f, delimiter="\t")

    assert text in absa_l_features
    assert audio in absa_a_features
    assert video in absa_v_features
    l_features=[]
    for i in data["#3 String"]:
        i = np.array(i.split(" "))
        l_features.append(i)
    a_features=[]
    for data_ in data["#4 String"]:
        data_ = np.array(data_.split(" "))
        a_features.append(data_)
#####读取图片特征
    v_features = []
    for image_name in data["#2 ImageID"]:
        res_path = r"D:\lilili\HireMLPJL\dataset\2015_vit_feature"
        res_pt_name = image_name.replace('.jpg', '.pt')
        image_path = os.path.join(res_path, res_pt_name)
        image = torch.load(image_path)
        image = image.squeeze(dim=0)
        image = image.cpu().detach().numpy()  ###在cpu上将tensor转为ndarry
        v_features.append(image)
    v_features = np.array(v_features)
    labels = np.array([data_ for data_ in data["#1 Label"]])
    return l_features, a_features, v_features, labels

class AbsaDataset(Dataset):
    def __init__(self, mode, dataset='mosi', text='bert', audio='bert', video='resnet', normalize=[True, True, True]):
        assert mode in ['test', 'train', 'valid']
        assert dataset in ['mosei', 'mosi', 'pom','absa']

        self.dataset = dataset
        if dataset == 'absa':
            self.l_features, self.a_features, self.v_features, self.labels = get_absa_dataset(mode=mode, text=text, audio=audio, video=video, normalize=normalize)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if self.dataset == 'absa':
            return self.l_features[index], self.a_features[index], self.v_features[index], self.labels[index]
        else:
            raise NotImplementedError
            
    def __len__(self):
        return len(self.labels)
