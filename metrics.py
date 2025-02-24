from math import log10
import torch
import numpy as np 
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch.nn.functional as F
from scipy.special import softmax
import scipy.stats


# 计算峰值信噪比，用于衡量图像的质量
def PSNR(mse, peak=1.):
	return 10 * log10((peak ** 2) / mse)


# 用于计算和存储平均值和当前值。它有 reset 和 update 两个方法，用于重置和更新状态。
# 在使用 AverageMeter 时，我们需要先创建一个对象，如 losses，
# 然后在每个 batch 训练完成后，通过调用 update() 方法更新统计量。最后，通过调用 avg() 方法获取平均值
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


# 计算模型预测的类别索引与真实标签是否相等，最终返回一个由0.0和1.0组成的浮点数数组，其中1.0表示预测正确，0.0表示预测错误。
def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)


# AUC模型评价指标
def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out


# 计算预测和标签之间的 PRF，即精确率、召回率和 F1 分数。它用于衡量多分类模型的性能
def prf(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    pred_lab= np.argmax(preds, 1)
    p, r, f, s = precision_recall_fscore_support(labels, pred_lab, average='micro')
    return [p, r, f]


# 计算预测和真值之间的数值分数，即 FP、FN、TP 和 TN。它用于计算混淆矩阵的元素
def numeric_score(pred, gt):
    FP = np.float64(np.sum((pred == 1) & (gt == 0)))
    FN = np.float64(np.sum((pred == 0) & (gt == 1)))
    TP = np.float64(np.sum((pred == 1) & (gt == 1)))
    TN = np.float64(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


# 计算预测和标签之间的敏感度和特异度，即 TP / (TP + FN) 和 TN / (TN + FP)。它用于衡量二分类模型的灵敏性和特异性
def metrics(preds, labels):
    preds = np.argmax(preds, 1) 
    FP, FN, TP, TN = numeric_score(preds, labels)
    sen = TP / (TP + FN + 1e-10)  # recall sensitivity
    spe = TN / (TN + FP + 1e-10)

    return sen, spe

