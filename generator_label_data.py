import os, sys
import torch
import random
from config import Config

path = sys.path[0]

config = Config(TrainOrVal = 'train')
img_name_file = open(config.image_name, encoding = 'utf-8')
img_name = img_name_file.readlines()
is_label_data = [0] * len(img_name)
is_label_data = torch.tensor(is_label_data)
is_label_data[:int(1 * len(is_label_data))] = 1
is_label_data = is_label_data.cpu().tolist()
random.shuffle(is_label_data)

file = open(os.path.join(path, 'data/{}/is_label_data_100.txt'.format(config.dataset)), mode = 'w')
for i in range(len(is_label_data)):
    file.write(str(img_name[i]).split('\n')[0] + ' ' + str(is_label_data[i]) + '\n')