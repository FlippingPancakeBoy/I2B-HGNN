import numpy as np
from sklearn.model_selection import StratifiedKFold
from node import get_node_feature
import csv
import os
from scipy.spatial import distance
from opt import *


opt = OptInit().initialize() # 参数命名空间..
data_folder = opt.data_folder
id_file = "id.txt"
phenotypic_file = "noimgs.csv"
# 非图像信息
if opt.dataset == 'ABIDE':
    scores = [opt.sites, opt.genders, opt.ages, opt.handedness]
elif opt.dataset == 'ADNI':
    scores = [opt.ages, opt.genders]


def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()
    return ((dataset - mean) / std).astype(dtype)


def intensityNormalisationFeatureScaling(dataset, dtype):
    max = dataset.max()
    min = dataset.min()

    return ((dataset - min) / (max - min)).astype(dtype)


class dataloader:
    def __init__(self, args, raw_features=None, y=None, pd_dict={}):
        self.args = args
        self.seed = args.seed
        self.node_ftr_dim = args.node_ftr_dim
        self.num_classes = args.num_classes
        self.num_subjects = args.num_subjects
        self.labels = args.labels
        self.ages = args.ages
        self.genders = args.genders
        self.sites = args.sites
        self.handedness = args.handedness

        # 导入数据
        self.raw_features = raw_features
        self.y = y
        self.pd_dict = pd_dict

    def load_data(self):
        """导入数据"""
        # 获取被试ID
        subject_IDs = get_ids(self.num_subjects)
        # 获取被试标签
        labels = map_values_to_ints(get_subject_score(subject_IDs, score=self.labels))
        # 人群图结点个数
        num_nodes = len(subject_IDs)
        # 获取被试非影像数据(ABIDE数据集时，只使用genders和sites，两个非图像信息)
        ages = get_subject_score(subject_IDs, score=self.ages)
        genders = map_values_to_ints(get_subject_score(subject_IDs, score=self.genders)) # 返回的genders是一个字典，包含被试ID和对应的性别编码。
        sites = map_values_to_ints(get_subject_score(subject_IDs, score=self.sites))
        handednesses = map_values_to_ints(get_subject_score(subject_IDs, score=self.handedness))

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int32)
        site = np.zeros([num_nodes], dtype=np.int32)
        handedness = np.zeros([num_nodes], dtype=np.int32)

        # 获取raw_features(每个被试的个体图，x:fc矩阵，edge_index, edge_atrr), y, y_onehot, phonetic_data, phonetic_score
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
            site[i] = sites[subject_IDs[i]]
            handedness[i] = handednesses[subject_IDs[i]]
            age[i] = float(ages[subject_IDs[i]])

        self.y = y
        self.y_onehot = y_onehot
        self.raw_features = get_node_feature(self.args) # 每个被试的个体图特征数据组成的列表
        # 创建一个数组用于存储非影像数据（如性别和站点信息）
        phonetic_data = np.zeros([num_nodes, 4], dtype=np.float32)
        phonetic_data[:, 0] = gender
        phonetic_data[:, 1] = site
        phonetic_data[:, 2] = age
        phonetic_data[:, 3] = handedness
        # 将非影像信息存储在字典中（pd_dict）
        self.pd_dict[self.genders] = np.copy(phonetic_data[:, 0])
        self.pd_dict[self.sites] = np.copy(phonetic_data[:, 1])
        self.pd_dict[self.ages] = np.copy(phonetic_data[:, 2])
        self.pd_dict[self.handedness] = np.copy(phonetic_data[:, 3])
        phonetic_score = self.pd_dict

        return self.raw_features, self.y, phonetic_data, phonetic_score

    def data_split(self, n_folds):
        """用于将数据集划分为多个交叉验证的折"""
        skf = StratifiedKFold(n_splits=n_folds, random_state=self.seed, shuffle=True)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_inputs(self, nonimg, embeddings, phonetic_score):
        """利用结点嵌入特征和影像特征构建人群图，边索引和边属性"""
        self.node_ftr = np.array(embeddings.detach().cpu().numpy())
        # 被试结点个数
        n = self.node_ftr.shape[0]
        # 完全无环图的边数
        num_edge = n*(1+n)//2 - n

        # 构建人群图,edge_index为边索引，edgenet_input为边嵌入(两个被试之间的非影像特征的拼接),aff_score为af_adj的上三角元素列表
        edge_index_site = np.zeros([2, num_edge], dtype=np.int64)
        edge_index_sex = np.zeros([2, num_edge], dtype=np.int64)
        edge_index_age = np.zeros([2, num_edge], dtype=np.int64)
        edge_index_handedness = np.zeros([2, num_edge], dtype=np.int64)

        # pd_ftr_dim=2 是两个被试同一个非图像信息，方便后面确定两个被试在这一个非图像信息下的相关性
        # edge_input_site = np.zeros([num_edge, 2], dtype=np.float32)
        # edge_input_sex = np.zeros([num_edge, 2], dtype=np.float32)
        # edge_input_age = np.zeros([num_edge, 2], dtype=np.float32)
        # edge_input_handedness = np.zeros([num_edge, 2], dtype=np.float32)

        aff_score_site = np.zeros(num_edge, dtype=np.float32)
        aff_score_sex = np.zeros(num_edge, dtype=np.float32)
        aff_score_age = np.zeros(num_edge, dtype=np.float32)
        aff_score_handedness = np.zeros(num_edge, dtype=np.float32)

        # 相当于一个分数，每个同构图节点之间的分数
        aff_adj_site, aff_adj_sex, aff_adj_age, aff_adj_handedness = get_static_affinity_adj(self.node_ftr, phonetic_score)

        flatten_ind = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_index_sex[:, flatten_ind] = [i, j]
                edge_index_site[:, flatten_ind] = [i, j]
                edge_index_age[:, flatten_ind] = [i, j]
                edge_index_handedness[:, flatten_ind] = [i, j]

                # edge_input_sex[flatten_ind] = (nonimg[i][0], nonimg[j][0])
                # edge_input_site[flatten_ind] = (nonimg[i][1], nonimg[j][1])
                # edge_input_age[flatten_ind] = (nonimg[i][2], nonimg[j][2])
                # edge_input_handedness[flatten_ind] = (nonimg[i][3], nonimg[j][3])

                # 相当于人群图相似度得分
                aff_score_site[flatten_ind] = aff_adj_site[i][j]
                aff_score_sex[flatten_ind] = aff_adj_sex[i][j]
                aff_score_age[flatten_ind] = aff_adj_age[i][j]
                aff_score_handedness[flatten_ind] = aff_adj_handedness[i][j]

                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind_site = np.where(aff_score_site > 0.5)[0]
        keep_ind_sex = np.where(aff_score_sex > opt.threshold)[0]
        keep_ind_age = np.where(aff_score_age > opt.threshold)[0]
        keep_ind_handedness = np.where(aff_score_handedness > opt.threshold)[0]

        edge_index_site = edge_index_site[:, keep_ind_site]
        # edge_input_site = edge_input_site[keep_ind_site]
        edge_index_sex = edge_index_sex[:, keep_ind_sex]
        # edge_input_sex = edge_input_sex[keep_ind_sex]
        edge_index_age = edge_index_age[:, keep_ind_age]
        # edge_input_age = edge_input_age[keep_ind_age]
        edge_index_handedness = edge_index_handedness[:, keep_ind_handedness]
        # edge_input_handedness = edge_input_handedness[keep_ind_handedness]

        return edge_index_site, edge_index_sex, edge_index_age, edge_index_handedness


def get_subject_score(subject_list, score):
    """获取被试表型信息"""
    # score是要获取的特定表型信息的列名。键为被试ID（key），值为特定表型信息（score）
    scores_dict = {}

    phenotype = os.path.join(data_folder, phenotypic_file)
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row[opt.key] in subject_list:
                scores_dict[row[opt.key]] = row[score]
    return scores_dict


def get_ids(num_subjects=None):
    """获取被试ID"""
    subject_IDs = np.genfromtxt(os.path.join(data_folder, id_file), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


def create_affinity_graph_from_scores(scores, pd_dict):
    '''只根据不同的表型特征，创建的四个相似性图（储存各个被试之间的相似非图像信息）'''
    # 方阵信息表示被试之间的相似性。这种相似性可以根据不同的表型特征来定义，比如年龄或智商的相似性，或者其他表型特征的相同性。
    num_nodes = len(pd_dict[scores[0]])

    # 构建两张非图像信息的值的图，分别反应两种非图像信息在被试之间的相似性，异构图
    graph_site = np.zeros((num_nodes, num_nodes))    # 全零矩阵，即邻接矩阵
    graph_sex = np.zeros((num_nodes, num_nodes))
    graph_age = np.zeros((num_nodes, num_nodes))
    graph_handedness = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]
        # 若特征为年龄/FIQ,则比较被试间的绝对差阈值，超过阈值边值为1，否则为0
        if l in [opt.sites]:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        if label_dict[k] == label_dict[j]:
                            graph_site[k, j] += 1
                            graph_site[j, k] += 1
                    except ValueError:  # missing label
                        pass

        if l in [opt.genders]:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        if label_dict[k] == label_dict[j]:
                            graph_sex[k, j] += 1
                            graph_sex[j, k] += 1
                    except ValueError:  # missing label
                        pass

        if l in [opt.handedness]:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        if label_dict[k] == label_dict[j]:
                            graph_handedness[k, j] += 1
                            graph_handedness[j, k] += 1
                    except ValueError:  # missing label
                        pass

        if l in [opt.ages]:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if (val <= 5):
                            graph_age[k, j] += 1
                            graph_age[j, k] += 1
                    except ValueError:  # missing label
                        pass

    return graph_site, graph_sex, graph_age, graph_handedness


def get_static_affinity_adj(features, pd_dict):
    """图像信息 * 非图像信息graph →→→ 四个融合了图像和非图像信息的同构图"""
    # pd_affinity是一个graph，里面是被试之间通过两个非图像数据得到的数值，间接反应相似性
    pd_affinity_site, pd_affinity_sex, pd_affinity_age, pd_affinity_handedness = create_affinity_graph_from_scores(scores, pd_dict)
    distv = distance.pdist(features, metric='correlation')      # 特征数据的方阵
    dist = distance.squareform(distv)  
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))

    adj_site = pd_affinity_site * feature_sim
    adj_sex = pd_affinity_sex * feature_sim
    adj_age = pd_affinity_age * feature_sim
    adj_handedness = pd_affinity_handedness * feature_sim

    return adj_site, adj_sex, adj_age, adj_handedness

def get_adni_id():
    """获取被试表型信息，从表型文件中提取被试的 ID，并将这些 ID 写入到一个文本文件中"""
    scores_dict = {}

    phenotype = os.path.join(data_folder, phenotypic_file)
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scores_dict[row[opt.key]] = row[opt.key]
    l = []
    for v in scores_dict.keys():
        l.append(v)
    with open(r"./data/ADNI_aal/id.txt", "w") as f:
        for item in l:
            f.write("%s\n" % item)


def map_values_to_ints(dict):
    """将表型数据的字符串数据映射为整数"""
    # eg.可以将 "男" 映射为 0，"女" 映射为 1
    value_to_int = {}
    int_list = []
    curr_int = 0
    for v in dict.values():
        if v not in value_to_int:
            value_to_int[v] = curr_int
            curr_int += 1
        int_list.append(value_to_int[v])
    int_dict = {}
    for k in dict.keys():
        int_dict[k] = value_to_int[dict[k]]
    return int_dict


if __name__ == '__main__':
    get_adni_id()
