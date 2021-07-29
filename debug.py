import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np


def file_load(txt_path):
    
    data_path = []    
    f = open(txt_path , 'r')
    while True:
        line = f.readline()
        if not line: break
        data_path.append(line[:-1])
    f.close()
    return data_path

def emg_sampler(emg,target_length):
    
    sampled_emg = emg[::4]
    if len(sampled_emg) >= target_length:
        return sampled_emg[:target_length]
    else:
        output = np.zeros((target_length,8))
        output[:len(sampled_emg)] = sampled_emg
        return output
    
class Myodata(Dataset):
    def __init__(self, train=True):
        super().__init__()

        self.max_len = 870
        """
        opt_data : 'train', 'validation'
        
        """
        if train == True:
            self.file_list = file_load('train.txt')
        else:
            self.file_list = file_load('test.txt')
                    
    def __getitem__(self, index):
        
        info = self.file_list[index].split(' ')
        acc = (pd.read_csv(info[1],header=None).dropna(axis=1)).to_numpy(dtype=np.float)   ## 가 끔씩 계측값는데 잡히는 애들이 있음
        gyr = pd.read_csv(info[2],header=None).dropna(axis=1).to_numpy(dtype=np.float)
        emg = pd.read_csv(info[0],header=None).dropna(axis=1).to_numpy(dtype=np.float)
        emg = emg_sampler(emg,len(acc))
        
        x = np.concatenate((emg,acc,gyr),axis=1)
        pad_len = self.max_len - len(x)
        try:
            self.x_data = torch.tensor(np.pad(x,pad_width=((pad_len,0), (0, 0)))).T
        except:
            print('file error',info)
        self.y_data = torch.tensor(int(info[3])).float()
        return self.x_data, self.y_data

    def __len__(self):
        return len(self.file_list)


    
    

# TODO : 길이 맞게 패딩하고
# 모델 돌려보기
# 기존 cnv + lstm 참조해서

data = Myodata()
i = 0
for i,(x,y) in enumerate(data):
    if i % 100 == 0:
        print(i)