import argparse
import sys
import time
from itertools import chain
from global_config import *

from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.model_saver import ModelSaver
from dataset import ANetDataFull, collate_fn
from utils.helper_function import *

from model.caption_generator import CaptionGenerator
from model.sentence_localizer_times import SentenceLocalizer



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
    s, _ = torch.cat([ts1[0], ts2[0]], dim=0).min(0)
    e, _ = torch.cat([ts1[1], ts2[1]], dim=0).max(0)
    return torch.cat([s,e], dim=0)


def eval(model_sl, model_cg, data_loader, logger, saver, params):

    model_sl.eval()
    model_cg.eval()
    _start_time = time.time()
    accumulate_iou = 0
    accumulate_geneneration = 0
    pred_dict = {'version': 'V0',
                 'results':{},
                 'external_data':{
                    'used': True,
                     'details': 'provided C3D feature'
                 },
                 'params': params}

    logger.info('eval gd model:' + '*' *  86)

    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()

        # data pre processing
        video_feat, video_len, video_mask, sent_feat, sent_len, sent_mask, sent_gather_idx, ts_time, _, key = batch_data
        video_feat = Variable(video_feat.cuda())
        video_len = Variable(video_len.cuda())
        video_mask = Variable(video_mask.cuda())
        sent_feat = Variable(sent_feat.cuda())
        sent_len = Variable(sent_len.cuda())
        sent_mask = Variable(sent_mask.cuda())
        sent_gather_idx = Variable(sent_gather_idx.cuda())
        batch_size = video_feat.size(0)
        ts_seq = model_sl.forward_eval(video_feat, video_len, video_mask,
                                       sent_feat, sent_len, sent_mask, sent_gather_idx)

        if params['batch_log_interval'] != -1 and idx % params['batch_log_interval'] == 0:
            logger.info('eval: batch[%04d/%04d], elapsed time=%0.4fs, avg generated_sent: %f',
                        idx, len(data_loader), time.time() - batch_time, ts_seq.size(0) * 1.0 / batch_size)

        accumulate_geneneration += ts_seq.size(0)
        _, video_time_len = video_len.index_select(dim=0, index=sent_gather_idx).chunk(2, dim=1)

        pred_time = (cw2se(ts_seq) * video_time_len)
        accumulate_iou += model_sl.compute_mean_iou(pred_time.data, ts_time.cuda())
        pred_time = pred_time.data.cpu().numpy()
        pred_sent = sent_feat.cpu().data.numpy()

        for idx in range(len(sent_gather_idx)):
            video_key = key[sent_gather_idx.cpu().data[idx]]
            if video_key not in pred_dict['results']:
                pred_dict['results'][video_key] = list()
            pred_dict['results'][video_key].append({
                'sentence': data_loader.dataset.rtranslate(pred_sent[idx]),
                'timestamp': pred_time[idx].tolist(),
                'gt_timestamp': ts_time[idx].numpy().tolist(),
            })

    saver.save_submits(pred_dict, 0)
    logger.info('eval finished, elapsed time:%0.4fs, total_generation: %d, miou: %06.6f',
                time.time() - _start_time, accumulate_geneneration, accumulate_iou / len(data_loader))
    logger.info('*'*80)


def construct_model(params, saver, logger):
    if params['checkpoint'] is None:
        logger.error('checkpoints are required for evaluation')
        exit()
    logger.warn('use checkpoint: %s', params['checkpoint'])
    state_dict_sl, state_dict_cg, params_ = saver.load_model_slcg(params['checkpoint'])

    # params_['anchor_list'] = ANCHOR_LIST
    # params['anchor_list'] = params_['anchor_list']
    # params['regressor_scale'] = params_['regressor_scale']
    # params = params
    for key in params_:
        if 'data' in key:
            continue
        params[key] = params_[key]
    params['batch_size'] = 8
    model_sl = SentenceLocalizer(params['hidden_dim'], params['rnn_layer'], params['rnn_cell'], params['rnn_dropout'],
                                 params['bidirectional'], params['attention_type_sl'], params['regressor_scale'],
                                 params['vocab_size'], params['sent_embedding_dim'], params['video_feature_dim'],
                                 params['fc_dropout'], params['anchor_list'], params['feature_mixer_type'],
                                 params['video_use_residual'], params['sent_use_residual'])

    model_cg = CaptionGenerator(params['hidden_dim'], params['rnn_layer'], params['rnn_cell'], params['rnn_dropout'],
                                params['bidirectional'], params['attention_type_cg'], params['context_type'],
                                params['softmask_scale'], params['vocab_size'], params['sent_embedding_dim'],
                                params['video_feature_dim'], params['video_use_residual'], params['max_cap_length'])

    logger.info('*' * 100)
    sys.stdout.flush()
    print('caption generator' + '*' * 90)
    print(model_cg)
    print('sentence localizer' + '*' * 90)
    print(model_sl)
    sys.stdout.flush()
    logger.info('*' * 100)
    model_sl.load_state_dict(state_dict_sl)
    model_cg.load_state_dict(state_dict_cg)

    return model_sl, model_cg


def main(params):
    logger = logging.getLogger(params['alias'])
    set_device(logger, params['gpu_id'])
    saver = ModelSaver(params, os.path.abspath('./third_party/densevid_eval'))
    torch.manual_seed(params['rand_seed'])
    model_sl, model_cg = construct_model(params, saver, logger)

    model_sl, model_cg = model_sl.cuda(), model_cg.cuda()

    val_set = ANetDataFull(params['val_data'], params['feature_path'],
                           params['translator_path'], params['video_sample_rate'], logger)

    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=True,
                            num_workers=params['num_workers'], collate_fn=collate_fn, drop_last=True)
    print(params)
    eval(model_sl, model_cg, val_loader, logger, saver, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # datasets
    parser.add_argument('--train_data', type=str, default='./data/densecap/train.json',
                        help='training data path')
    parser.add_argument('--val_data', type=str, default='./data/densecap/val_1.json',
                        help='validation data path')
    parser.add_argument('--feature_path', type=str, default='./data/anet_v1.3.c3d.hdf5',
                        help='feature path')
    parser.add_argument('--vocab_size', type=int, default=6000,
                        help='vocabulary size')
    parser.add_argument('--translator_path', type=str, default='./data/translator6000.pkl',
                        help='translator path')

    # model setting
    parser.add_argument('--model', type=str, default="CaptionGenerator",
                        help='the model to be used')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='model checkpoint')
    parser.add_argument('--save_model_interval', type=int, default=1,
                        help='save the model parameters every a certain step')
    parser.add_argument('--max_cap_length', type=int, default=20,
                        help='max captioning length')

    # network setting
    parser.add_argument('--sent_embedding_dim', type=int, default=512)
    parser.add_argument('--video_feature_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden dimension of rnn')
    parser.add_argument('--rnn_cell', type=str, default='gru',
                        help='rnn cell used in the model')
    parser.add_argument('--bidirectional', type=bool, default=False,
                        help='Whether to use bidirectional RNN')
    parser.add_argument('--rnn_layer', type=int, default=2,
                        help='layers number of rnn')
    parser.add_argument('--fc_dropout', type=float, default=0.0,
                        help='dropout')
    parser.add_argument('--rnn_dropout', type=float, default=0.3,
                        help='rnn_dropout')
    parser.add_argument('--video_sample_rate', type=int, default=2,
                         help='video sample rate')
    parser.add_argument('--attention_type_cg', type=str, default='mean',
                        help='attention module used in encoding')
    parser.add_argument('--attention_type_sl', type=str, default='type0',
                        help='attention module used in encoding')
    parser.add_argument('--context_type', type=str, default='clr',
                        help='context type: clr, cl c')
    parser.add_argument('--regressor_scale', type=float, default=0.1,
                        help='regressor scale, used to normalize the regressor head')
    parser.add_argument('--softmask_scale', type=float, default=0.1,
                        help='softmask scale, used to normalize the regressor head')
    parser.add_argument('--feature_mixer_type', type=str, default='type0',
                        help='multi-model feature fusion function')
    parser.add_argument('--video_use_residual', type=bool, default=True)
    parser.add_argument('--sent_use_residual', type=bool, default=False)

    # training setting
    parser.add_argument('--runs', type=str, default='runs',
                        help='folder where models are saved')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='the id of gup used to train the model, -1 means automatically choose the best one')
    parser.add_argument('--optim_method', type=str, default='SGD',
                        help='masks for caption loss')
    parser.add_argument('--caption_loss_mask', type=float, default=1,
                        help='masks for caption loss')
    parser.add_argument('--reconstruction_loss_mask', type=float, default=1,
                        help='masks for caption loss')
    parser.add_argument('--training_epoch', type=int, default=100,
                        help='training epochs in total')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size used to train the model')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clip threshold(not used)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate used to train the model')
    parser.add_argument('--lr_decay_step', type=int, default=5,
                        help='decay learning rate after a certain epochs, UNUSED')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay learning rate by this value every decay step')
    parser.add_argument('--momentum', type=float, default=0.8,
                        help='momentum used in the process of learning')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay, i.e. weight normalization')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='used in data loader(only 1 is supported because of bugs in h5py)')
    parser.add_argument('--batch_log_interval', type=int, default=10,
                        help='log interval')
    parser.add_argument('--batch_log_interval_test', type=int, default=20,
                        help='log interval')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='test interval between training')
    parser.add_argument('--switch_optim_interval', type=int, default=1,
                        help='switch the optimizing target every interval epoch')
    parser.add_argument('--alias', type=str, default='test',
                        help='alias used in model/checkpoint saver')
    parser.add_argument('--soft_mask_function_scale', type=float, default=1,
                        help='alias used in model/checkpoint saver')
    parser.add_argument('--pretrain_epoch', type=int, default=5,
                        help='pretrain with fake01')
    parser.add_argument('--lr_step', type=int, nargs='+', default=[5, 20, 50],
                        help='lr_steps used to decay the learning_rate')

    parser.add_argument('--refine_round', type=int, default=1,
                        help='round used to refine the model')
    parser.add_argument('--iou_thres', type=float, default=0.9,
                        help='overlapping threshold for temporal segment refine')
    parser.add_argument('--n_rand', type=int, default=15,
                        help='number of random sample used for evaluation')
    parser.add_argument('--rand_seed', type=int, default=233,
                        help='random seed used to evaluation')
    params = parser.parse_args()
    params = vars(params)

    main(params)
    print('training finished successfully!')
