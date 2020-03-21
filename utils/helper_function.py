
from itertools import chain
from global_config import *
from torch.autograd import Variable
import os
import random


def set_device(logger, id=-1):
    logger.info('*' * 100)

    if id == -1:
        tmp_file_name = 'tmp%s' % (random.random())
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >%s' % (tmp_file_name))
        memory_gpu = [int(x.split()[2]) for x in open(tmp_file_name, 'r').readlines()]
        id = np.argmax(memory_gpu)
        os.system('rm %s' % (tmp_file_name))

    logger.info('process runs on gpu %d', id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)
    logger.info('*' * 100)
    return id


def param_refine(param, old_param):
    refine_dict = ['vocab_size', 'vocab_embedding_dim', 'hidden_len', 'feature_dim', 'rnn_layer',
                   'video_sample_rate']
    for key in refine_dict:
        param[key] = old_param[key]


def model_initialize(model, logger):
    for name, param in model.named_parameters():
        if param.dim() == 2:  # weights
            logger.info('initialized with kaiming normal: %s', name)
            torch.nn.init.kaiming_normal(param)
        elif param.dim() == 1:  # bias
            logger.info('initialized with constant 0: %s', name)
            torch.nn.init.constant(param, 0)


def hidden_transpose(hidden):
    """
    :param hidden: (~, batch_size, hidden_dim)
    :return:
        (batch_size, ~ * hidden_dim)
    """
    return hidden.transpose(0, 1).contiguous().view(hidden.size(1), -1)


def se2cw(se):
    """
    :param se: (batch, 2)
    :return:
        cw: (batch, 2)
    """
    s, e = se.chunk(2, dim=1)  # batch, 1
    return torch.cat([(s + e) / 2, e - s], dim=1)
    # _w = s - e
    # _w_delta = torch.zeros_like(_w) + DELTA
    # return torch.cat([(s + e) / 2, torch.max(_w, _w_delta)], dim=1)


def cw2se(cw):
    """
    :param cw: range in (0,1)
    :return:
    """

    c, w = cw.chunk(2, dim=1)  # batch, 1
    # _w_delta = torch.zeros_like(w) + DELTA
    # _w = torch.max(w, _w_delta)
    s, _ = torch.cat([c - w / 2, torch.zeros_like(c)], dim=1).max(1)
    e, _ = torch.cat([c + w / 2, torch.ones_like(c)], dim=1).min(1)
    return torch.stack([s, e], dim=1)


def compute_iou(seg1, seg2):
    """
    :param seg1: batch, 2 in (s, e) format
    :param seg2: batch, 2 in (s, e) format
    :return:
        iou: batch
    """
    seg1_s, seg1_e = seg1.chunk(2, dim=1)  # batch, 1
    seg2_s, seg2_e = seg2.chunk(2, dim=1)  # batch, 1
    min_end, _ = torch.cat([seg1_e, seg2_e], dim=1).min(1)  # batch
    max_end, _ = torch.cat([seg1_e, seg2_e], dim=1).max(1)
    min_beg, _ = torch.cat([seg1_s, seg2_s], dim=1).min(1)
    max_beg, _ = torch.cat([seg1_s, seg2_s], dim=1).max(1)
    intersection = min_end - max_beg
    intersection, _ = torch.stack([intersection, torch.zeros_like(intersection)], dim=1).max(1)  # batch
    union = max_end - min_beg  # batch
    iou = intersection / (union + DELTA)  # batch
    return iou


def temporal_segment_merge(ts1, ts2):
    """
    :param ts1:  (2,) se format
    :param ts2:  (2,) se format
    :return:
    """
    # import pdb; pdb.set_trace()
    s = torch.min(ts1[0], ts2[0])
    e = torch.max(ts1[1], ts2[1])
    # s, _ = torch.cat([ts1[0], ts2[0]], dim=0).min(0)
    # e, _ = torch.cat([ts1[1], ts2[1]], dim=0).max(0)
    return torch.stack([s, e], dim=0)


def refine_temporal_segment(ts_old, ts_new, ts_gather_idx, iou_thres):
    """
    :param ts_old:          (batch, 2), cw format (0, 1)
    :param ts_new:          (batch, 2), cw format (0, 1)
    :param ts_gather_idx:   (batch, )
    :return:
        ts_refined: (new_batch, 2)
        ts_gather_idx_refined, (new_batch, )
    """
    ts_new_se = cw2se(ts_new)
    ts_old_se = cw2se(ts_old)
    ts_refined = dict()
    ts_new_gather_idx = list()

    for idx1, idx2 in enumerate(ts_gather_idx.cpu().data.numpy().tolist()):
        if compute_iou(ts_new_se[idx1].unsqueeze(0), ts_old_se[idx1].unsqueeze(0)).mean().cpu().item() > 0.2:
            continue
        if idx2 not in ts_refined:
            # ts_refined[idx2] = [ts_new_se[idx1], ]  # add the first one as default
            ts_refined[idx2] = [Variable(FloatTensor([0, 1-DELTA])), ]  # add [0, 1] to avoid empty set
            ts_new_gather_idx.append(idx2)
        else:
            ts_temp1 = torch.stack(ts_refined[idx2], dim=0)
            ts_temp2 = ts_new_se[idx1].unsqueeze(0).expand_as(ts_temp1)
            ious = compute_iou(ts_temp1, ts_temp2)  # batch
            max_iou, max_iou_idx = ious.max(0)
            max_iou, max_iou_idx = max_iou.cpu().item(), max_iou_idx.cpu().item()
            if max_iou > iou_thres:
                ts_refined[idx2][max_iou_idx] = temporal_segment_merge(ts_new_se[idx1], ts_refined[idx2][max_iou_idx])
            else:
                ts_refined[idx2].append(ts_new_se[idx1])
                ts_new_gather_idx.append(idx2)

    return se2cw(torch.stack(list(chain(*ts_refined.values())), dim=0)), Variable(LongTensor(ts_new_gather_idx))


def gen_random_ts(batch_size, n_rand):
    """
    return Random FlatTensor (batch_size, 2)
    """
    linear_transform = torch.range(0, 1, 1.0 / n_rand)[:n_rand].unsqueeze(1)

    initial_random_c = torch.rand(n_rand, 1)
    initial_random_w = torch.rand(n_rand, 1)
    initial_random = torch.cat([initial_random_c, initial_random_w], dim=1).repeat(batch_size, 1)
    return initial_random.cuda()
