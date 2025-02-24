import os
import re
import scipy.io
from opt import *
from utils import get_ids, get_subject_score
# 指定文件夹路径
opt = OptInit().initialize()
folder_path = opt.data_folder

ids = get_ids()
dict = get_subject_score(ids, "Image Data ID")

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 判断是否为.mat文件
    if file_name.endswith('.mat'):
        # 从文件名中提取编号
        num = re.findall(r'\d{3}_S_\d{4}', file_name)[0]
        # 判断字典中是否有对应编号
        if num in dict:
            # 读取.mat文件
            data = scipy.io.loadmat(os.path.join(folder_path, file_name))
            # 构造新文件名
            new_file_name = file_name.replace(num, dict[num])
            # 保存为新文件名
            scipy.io.savemat(os.path.join(folder_path, new_file_name), data)
            # 删除原文件
            os.remove(os.path.join(folder_path, file_name))

