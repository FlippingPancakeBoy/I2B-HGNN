
from opt import *
from utils import dataloader

# 初始化参数
opt = OptInit().initialize()
dl = dataloader(opt)
raw_features, y, nonimg, phonetic_score = dl.load_data()
# torch.save(raw_features, f'./data/{opt.dataset}_{opt.atlas}/save_tensor/raw_features.pt')
# torch.save(y, f'./data/{opt.dataset}_{opt.atlas}/save_tensor/y.pt')
torch.save(nonimg, f'nonimg4.pt')
torch.save(phonetic_score, f'phonetic_score4.pt')