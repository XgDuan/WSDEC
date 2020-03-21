
import sys
from itertools import chain
from global_config import *

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helper_function import *

sys.path.append('./third_party/densevid_eval/coco-caption')

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class CaptionEvaluator(object):

    def __init__(self, rtranslator, params):
        self.scorers = [
            Meteor(), Rouge()]
        self.rtranslator = rtranslator
        self.params = params

    def evaluate(self, gts, res):
        scores = []
        for scorer in self.scorers:
            _, score = scorer.compute_score(gts, res)
            scores.append(score)
        return scores[0] + scores[1]

    def build_loss(self, sl_conf, video_feat, video_len, video_mask, sent_gd, model_cg):
        """
        :param sl_conf:         (batch, n_anchor)
        :param sl_gather_idx:   (batch, )
        :param video_feat:      (batch, ~, ~)
        :param video_len:       (batch, 2)
        :param video_mask:      (batch, ~, 1)
        :param model_cg:
        :return:
        """
        initial_anchors = self.params['anchor_list']
        n_anchors = len(initial_anchors)

        batch_size = video_feat.size(0)
        ts_seq = Variable(FloatTensor(initial_anchors).repeat(batch_size, 1))
        ts_gather_idx = Variable(LongTensor(range(batch_size)).unsqueeze(1).repeat(1, n_anchors).view(-1))
        _, sent_pred, sent_len, sent_mask = model_cg.forward(video_feat, video_len, video_mask, ts_seq, ts_gather_idx)

        sent_pred = sent_pred.view(batch_size, n_anchors, -1)

        cur_res = {}
        cur_gts = {}
        for idxi, gts_caption in enumerate(sent_gd):
            cur_gts[idxi] = [remove_nonascii(self.rtranslator.rtranslate(gts_caption.cpu().data.numpy()))]
            for idxj in range(n_anchors):
                cur_res[idxi*n_anchors+idxj] = [remove_nonascii(self.rtranslator.rtranslate(sent_pred[idxi, idxj].cpu().data.numpy()))]

        res = {i: {j: cur_res[i*n_anchors+j] for j in range(n_anchors)} for i in range(sent_gd.size(0))}
        gts = {i: {j: cur_gts[i] for j in range(n_anchors)} for i in range(sent_gd.size(0))}

        scores = []
        for i in range(sent_gd.size(0)):
            score = self.evaluate(gts[i], res[i])
            scores.append(score)
        scores = np.exp(10 * np.array(scores))
        scores = scores / np.max(scores)
        # import pdb; pdb.set_trace()
        # print(scores)
        rewards = Variable(torch.from_numpy(scores).float().cuda())
        rewards = rewards - rewards.mean(dim=1, keepdim=True)
        return - (F.log_softmax(sl_conf) * rewards).sum() / sent_gd.size(0)
