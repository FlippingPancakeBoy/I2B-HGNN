import torch.nn
import os
from opt import *
from utils import dataloader
from metrics import accuracy, auc, prf, metrics
from layer import H_gnn
from tensorboardX import SummaryWriter

def train():
    acc = 0
    best_eval_acc = 0
    for epoch in range(opt.num_iter):
        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            node_logits, mi_loss, hg_loss = model(raw_features)
        loss_cla = loss_fn(node_logits[train_ind], labels[train_ind])
        loss = loss_cla + opt.mi_weight * mi_loss + opt.hg_weight * hg_loss
        if opt.log_save:
            writer.add_scalar('train\tloss', loss.item(), epoch)
        loss.backward()
        optimizer.step()
        correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

        model.eval()
        with torch.set_grad_enabled(False):
            node_logits, mi_loss, hg_loss = model(raw_features)
        loss_eval_cla = loss_fn(node_logits[test_ind], labels[test_ind])
        loss_eval = loss_eval_cla + opt.mi_weight_eval * mi_loss + opt.hg_weight_eval * hg_loss
        if opt.log_save:
            writer.add_scalar('test\tloss', loss_eval.item(), epoch)
        logits_test = node_logits[test_ind].detach().cpu().numpy()
        correct_test, acc_test = accuracy(logits_test, y[test_ind])
        if opt.log_save:
            writer.add_scalar('test\tacc', acc_test, epoch)
        test_sen, test_spe = metrics(logits_test, y[test_ind])
        auc_test = auc(logits_test, y[test_ind])
        prf_test = prf(logits_test, y[test_ind])

        if acc_test > best_eval_acc:
            best_eval_acc = acc_test
            best_epoch_stats = {
                'epoch': epoch,
                'train_loss': loss.item(),
                'train_acc': acc_train.item(),
                'eval_loss': loss_eval.item(),
                'eval_acc': acc_test.item(),
                'eval_spe': test_spe
            }
            
        if (epoch + 1) % 5 == 0:
            print("Best result until epoch {}: \tEpoch: {},\ttrain loss: {:.5f},\ttrain acc: {:.5f}, \teval loss{:.5f} ,\teval acc{:.5f} ,"
                  "\teval spe: {:.5f}".format(
                epoch + 1,
                best_epoch_stats['epoch'],
                best_epoch_stats['train_loss'],
                best_epoch_stats['train_acc'],
                best_epoch_stats['eval_loss'],
                best_epoch_stats['eval_acc'],
                best_epoch_stats['eval_spe']
            ))

        if acc_test > acc:
            acc = acc_test
            correct = correct_test
            aucs[fold] = auc_test
            prfs[fold] = prf_test
            sens[fold] = test_sen
            spes[fold] = test_spe
            if (opt.ckpt_path != '') and opt.model_save:
                if not os.path.exists(opt.ckpt_path):
                    os.makedirs(opt.ckpt_path)
                torch.save(model.state_dict(), fold_model_path)
                print("{} Saved model to:{}".format("\u2714", fold_model_path))

    accs[fold] = acc
    corrects[fold] = correct

    print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))



def evaluate():
    print("  Number of testing samples %d" % len(test_ind))
    print('  Start testing...')
    model.load_state_dict(torch.load(fold_model_path))
    model.eval()
    # 用加载的模型参数对测试数据进行前向传播
    node_logits, mi = model(raw_features)
    # 将logits转换为 numpy 数组，并且仅保留测试集的部分
    logits_test = node_logits[test_ind].detach().cpu().numpy()
    corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
    sens[fold], spes[fold] = metrics(logits_test, y[test_ind])
    aucs[fold] = auc(logits_test, y[test_ind])
    prfs[fold] = prf(logits_test, y[test_ind])
    print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))


def save_result():
    with open(f'./result/{opt.dataset}_{opt.atlas}/result.txt', 'a') as f:
        print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)), file=f)
        print("=> Average test sen in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)), file=f)
        print("=> Average test spe in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)), file=f)
        print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)), file=f)
        print("=> Average test sensitivity {:.4f}({:.4f}), specificity {:.4f}({:.4f}), F1-score {:.4f}({:.4f})"
              .format(se, se_var, sp, sp_var, f1, f1_var), file=f)


if __name__ == '__main__':
    # 初始化参数
    opt = OptInit().initialize()
    raw_features = torch.load(f'./data/{opt.dataset}_{opt.atlas}/save_tensor/raw_features.pt')
    y = torch.load(f'./data/{opt.dataset}_{opt.atlas}/save_tensor/y.pt')
    nonimg = torch.load(f'./data/{opt.dataset}_{opt.atlas}/save_tensor/nonimg4.pt')
    phonetic_score = torch.load(f'./data/{opt.dataset}_{opt.atlas}/save_tensor/phonetic_score4.pt')
    dl = dataloader(opt, raw_features, y, phonetic_score)

    # k折交叉验证
    n_folds = opt.folds
    cv_splits = dl.data_split(n_folds)
    # 训练和评估模型
    corrects = np.zeros(n_folds, dtype=np.int32) 
    accs = np.zeros(n_folds, dtype=np.float32) 
    sens = np.zeros(n_folds, dtype=np.float32) 
    spes = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds, 3], dtype=np.float32)
    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold)) 
        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        model = H_gnn(opt, nonimg, phonetic_score).to(opt.device)
        print(model)
        loss_fn =torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)
        if opt.log_save:
            writer = SummaryWriter(f'./{opt.dataset}_{opt.atlas}_log/{fold}')
        if opt.train == 1:
            train()
        elif opt.train == 0:
            evaluate()
    print("\r\n========================== Finish ==========================")
    se, sp, f1 = np.mean(prfs, axis=0)
    se_var, sp_var, f1_var = np.var(prfs, axis=0)
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)))
    print("=> Average test sen in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)))
    print("=> Average test spe in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)))
    print("=> Average test sensitivity {:.4f}({:.4f}), specificity {:.4f}({:.4f}), F1-score {:.4f}({:.4f})"
          .format(se, se_var, sp, sp_var, f1, f1_var))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))
    save_result()
