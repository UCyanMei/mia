import numpy as np

import torch
from rich.progress import track
import fire
import logging
from rich.logging import RichHandler
from pytorch_lightning import seed_everything
import components
from typing import Type, Dict
from itertools import chain
from model import UNet
from dataset_utils import load_member_data
from torchmetrics.classification import BinaryAUROC, BinaryROC
from sklearn import metrics  # 添加 sklearn.metrics 导入

# ------------------------ roc计算函数 ---------------------------
def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    min_conf = min(member_scores.min(), nonmember_scores.min()).item()
    max_conf = max(member_scores.max(), nonmember_scores.max()).item()

    FPR_list = []
    TPR_list = []

    for threshold in torch.arange(min_conf, max_conf, (max_conf - min_conf) / n_points):
        TP = (member_scores < threshold).sum()
        TN = (nonmember_scores >= threshold).sum()
        FP = (nonmember_scores < threshold).sum()
        FN = (member_scores >= threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > max_asr:
            max_asr = ASR.item()
            max_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    # auc = metrics.auc(FPR_list, TPR_list)
    return max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold
#-------------------------------------------------------------------

def get_FLAGS():

    def FLAGS(x): return x
    FLAGS.T = 1000
    FLAGS.ch = 128
    FLAGS.ch_mult = [1, 2, 2, 2]
    FLAGS.attn = [1]
    FLAGS.num_res_blocks = 2
    FLAGS.dropout = 0.1
    FLAGS.beta_1 = 0.0001
    FLAGS.beta_T = 0.02

    return FLAGS


def get_model(ckpt, WA=True):
    FLAGS = get_FLAGS()
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(ckpt)

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model


class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        return self.model(xt, t=t)


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "SecMI": components.SecMIAttacker,
    "PIA": components.PIA,
    "naive": components.NaiveAttacker,
    "PIAN": components.PIAN,
}


DEVICE = 'cuda'


@torch.no_grad()
def main(checkpoint='checkpoint/ckpt_cifar10.pt',
         dataset='cifar10',
         attacker_name="PIA",
         attack_num=30, interval=10,
         seed=0):
    seed_everything(seed)

    FLAGS = get_FLAGS()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("loading model...")
    model = get_model(checkpoint, WA = True).to(DEVICE)
    model.eval()

    logger.info("loading dataset...")
    if dataset == 'cifar10':
        _, _, train_loader, test_loader = load_member_data(dataset_name='cifar10', batch_size=64,
                                                           shuffle=False, randaugment=False)
    if dataset == 'TINY-IN':
        _, _, train_loader, test_loader = load_member_data(dataset_name='TINY-IN', batch_size=64,
                                                           shuffle=False, randaugment=False)

    attacker = attackers[attacker_name](
        torch.from_numpy(np.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T)).to(DEVICE), interval, attack_num, EpsGetter(model), lambda x: x * 2 - 1)

    logger.info("attack start...")

    members, nonmembers = [], []
    for member, nonmember in track(zip(train_loader, chain(*([test_loader]))), total=len(test_loader)):
        member, nonmember = member[0].to(DEVICE), nonmember[0].to(DEVICE)

        members.append(attacker(member))
        nonmembers.append(attacker(nonmember))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]

    member = members[0]
    nonmember = nonmembers[0]

    auroc = [BinaryAUROC().cuda()(
        torch.cat([member[i] / max([member[i].max().item(),nonmember[i].max().item()]),
                   nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()])]),
        torch.cat([torch.zeros(member.shape[1]).long(),
                   torch.ones(nonmember.shape[1]).long()]).cuda()).item()
        for i in range(member.shape[0])]
    # tpr_fpr = [BinaryROC().cuda()(
    #     torch.cat([1 - nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()]),
    #                1 - member[i] / max([member[i].max().item(), nonmember[i].max().item()])]),
    #     torch.cat([torch.zeros(member.shape[1]).long(),
    #                torch.ones(nonmember.shape[1]).long()]).cuda()) for i in range(member.shape[0])]
    tpr_fpr = [BinaryROC().cuda()(
        torch.cat([member[i] / max([member[i].max().item(),nonmember[i].max().item()]),
                   nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()])]),
        torch.cat([torch.zeros(member.shape[1]).long(),
                   torch.ones(nonmember.shape[1]).long()]).cuda())
        for i in range(member.shape[0])]
    tpr_fpr_1 = [i[1][(i[0] < 0.01).sum() - 1].item() for i in tpr_fpr]
    cp_auroc = auroc[:]
    cp_auroc.sort(reverse=True)
    cp_tpr_fpr_1 = tpr_fpr_1[:]
    cp_tpr_fpr_1.sort(reverse=True)

    # --------------------------- 计算成功率 -----------------------------
    asr = []
    thresholds = []

    for i in range(member.shape[0]):
        member_scores = member[i]
        nonmember_scores = nonmember[i]

        # 计算ASR和最佳阈值
        max_asr, fpr_list, tpr_list, max_threshold = roc(member_scores, nonmember_scores)

        # custom_auroc.append(auc)
        asr.append(max_asr)
        thresholds.append(max_threshold.item())
    # ------------------------------------------------------------------

    # 打印结果
    print('auc', auroc)
    print('tpr @ 1% fpr', cp_tpr_fpr_1)
    print('attack success rates (ASR)', asr)
    print('best thresholds', thresholds)
    print('best ASR', max(asr))


if __name__ == '__main__':
    fire.Fire(main)
