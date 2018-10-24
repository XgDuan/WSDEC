import argparse
import sys
import time
from itertools import chain
from global_config import *

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from utils.model_saver import ModelSaver
from dataset import ANetDataFull, ANetDataSample, collate_fn
from model.caption_generator import CaptionGenerator
from model.sentence_localizer_regressor import SentenceLocalizer
from utils.helper_function import *




sys.path.append('./third_party/densevid_eval/coco-caption')



def pretrain_cg(model, data_loader, params, logger, step, optimizer):
    model.train()

    _start_time = time.time()
    accumulate_loss = 0

    logger.info('learning rate:' + '*' *  86)
    logger.info('pretraining with fake proposal')
    for param_group in optimizer.param_groups:
        logger.info('  ' * 7 + '|%s: %s,', param_group['name'], param_group['lr'])
    logger.info('*' * 100)

    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()

        # data pre processing
        video_feat, video_len, video_mask, sent_feat, sent_len, sent_mask, sent_gather_idx, _, ts_seq, _ = batch_data
        video_feat = Variable(video_feat.cuda())
        video_len = Variable(video_len.cuda())
        video_mask = Variable(video_mask.cuda())
        sent_feat = Variable(sent_feat.cuda())
        sent_len = Variable(sent_len.cuda())
        sent_mask = Variable(sent_mask.cuda())
        sent_gather_idx = Variable(sent_gather_idx.cuda())
        ts_seq = Variable(FloatTensor(sent_gather_idx.size(0), 2))
        ts_seq[:, 0] = 0
        ts_seq[:, 1] = 1

        # forward
        video_seq_len, _ = video_len.index_select(dim=0, index=sent_gather_idx).chunk(2, dim=1)
        ts_seq = se2cw(ts_seq)  # normalized in (0, 1) cw format.
        caption_prob, _, _, _ = model.forward(video_feat, video_len, video_mask, ts_seq, sent_gather_idx, sent_feat)

        # backward
        loss = model.build_loss(caption_prob, sent_feat, sent_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statics
        accumulate_loss += loss.cpu().data[0]
        if params['batch_log_interval'] != -1 and idx % params['batch_log_interval'] == 0:
            logger.info('train: epoch[%05d], batch[%04d/%04d], elapsed time=%0.4fs, loss: %06.6f',
                        step, idx, len(data_loader), time.time() - batch_time, loss.cpu().data[0])

    logger.info('epoch [%05d]: elapsed time:%0.4fs, avg loss: %06.6f',
                step, time.time() - _start_time, accumulate_loss / len(data_loader))
    logger.info('*'*100)


def train_cg(model_cg, model_sl, data_loader, params, logger, step, optimizer):
    model_cg.train()
    model_sl.train()

    _start_time = time.time()
    accumulate_loss = 0

    logger.info('learning rate:' + '*' *  86)
    logger.info('training on optimizing caption generator')
    for param_group in optimizer.param_groups:
        logger.info('  ' * 7 + '|%s: %s,', param_group['name'], param_group['lr'])
    logger.info('*' * 100)

    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()

        # data pre processing
        video_feat, video_len, video_mask, sent_feat, sent_len, sent_mask, sent_gather_idx, _, _ , _ = batch_data
        video_feat = Variable(video_feat.cuda())
        video_len = Variable(video_len.cuda())
        video_mask = Variable(video_mask.cuda())
        sent_feat = Variable(sent_feat.cuda())
        sent_len = Variable(sent_len.cuda())
        sent_mask = Variable(sent_mask.cuda())
        sent_gather_idx = Variable(sent_gather_idx.cuda())

        # forward
        # forward with sl
        ts_seq = model_sl.forward_diff(
            video_feat, video_len, video_mask, sent_feat, sent_len, sent_mask, sent_gather_idx)
        # ts_seq = ts_seq + Variable(torch.rand(*ts_seq.size())).cuda() / 100
        ts_seq_noise = Variable(torch.rand(*ts_seq.size()) - 0.5).cuda() / 50 # (-0.01, 0.01)
        ts_seq = ts_seq + ts_seq_noise

        # forward with cg
        caption_prob, caption_pred, caption_len, caption_mask = model_cg.forward(video_feat, video_len, video_mask, ts_seq, sent_gather_idx, sent_feat)
        ts_seq_new = model_sl.forward_diff(
           video_feat, video_len, video_mask, caption_pred.detach(), sent_len, sent_mask, sent_gather_idx)

        # backward
        loss = model_cg.build_loss(caption_prob, sent_feat, sent_mask)  # caption loss
        loss_lgl = ((ts_seq_new - ts_seq.detach())**2).mean() / 100  # the reconstruction loss
        optimizer.zero_grad()
        (loss + loss_lgl).backward()
        torch.nn.utils.clip_grad_norm(model_sl.parameters(), params['grad_clip'], norm_type=2)

        optimizer.step()

        # statics
        accumulate_loss += loss.cpu().data[0]
        if params['batch_log_interval'] != -1 and idx % params['batch_log_interval'] == 0:
            logger.info('train: epoch[%05d], batch[%04d/%04d], elapsed time=%0.4fs, loss: %06.6f, %06.6f',
            # logger.info('train: epoch[%05d], batch[%04d/%04d], elapsed time=%0.4fs, loss: %06.6f',
                        # step, idx, len(data_loader), time.time() - batch_time, loss.cpu().data[0])
                        step, idx, len(data_loader), time.time() - batch_time, loss.cpu().data[0], loss_lgl.cpu().data[0])

    logger.info('epoch [%05d]: elapsed time:%0.4fs, avg loss: %06.6f',
                step, time.time() - _start_time, accumulate_loss / len(data_loader))
    logger.info('*'*100)


def eval(model_sl, model_cg, data_loader, logger, saver, params, step):

    model_sl.eval()
    model_cg.eval()
    _start_time = time.time()
    accumulate_geneneration = 0
    pred_dict = {'version': 'V0',
                 'results':{},
                 'external_data':{
                    'used': True,
                     'details': 'provided C3D feature'
                 },
                 'params': params}

    logger.info('eval gd model:' + '*' *  86)

    initial_anchors = params['anchor_list']
    n_anchors = len(initial_anchors)
    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()

        # data pre processing
        video_feat, video_len, video_mask, _, _, _, _, _, _, key = batch_data
        video_feat = Variable(video_feat.cuda())
        video_len = Variable(video_len.cuda())
        video_mask = Variable(video_mask.cuda())

        batch_size = video_feat.size(0)
        ts_seq = Variable(FloatTensor(initial_anchors).repeat(batch_size, 1))
        ts_gather_idx = Variable(LongTensor(range(batch_size)).unsqueeze(1).repeat(1, n_anchors).view(-1))
        _, sent_pred, sent_len, sent_mask = model_cg.forward(video_feat, video_len, video_mask, ts_seq, ts_gather_idx)
        refine_round = 0

        while refine_round < params['refine_round']:
            new_ts_seq = model_sl.forward_eval(video_feat, video_len, video_mask,
                                               sent_pred, sent_len, sent_mask, ts_gather_idx)
            ts_seq, ts_gather_idx = refine_temporal_segment(ts_seq, new_ts_seq, ts_gather_idx, params['iou_thres'])
            _, sent_pred, sent_len, sent_mask = model_cg.forward(video_feat, video_len, video_mask, ts_seq, ts_gather_idx)
            refine_round += 1

        if params['batch_log_interval'] != -1 and idx % params['batch_log_interval'] == 0:
            logger.info('eval: batch[%04d/%04d], elapsed time=%0.4fs, avg generated_sent: %f',
                        idx, len(data_loader), time.time() - batch_time, ts_seq.size(0) * 1.0 / batch_size)

        accumulate_geneneration += ts_seq.size(0)
        _, video_time_len = video_len.index_select(dim=0, index=ts_gather_idx).chunk(2, dim=1)

        pred_time = (cw2se(ts_seq) * video_time_len).cpu().data.numpy()
        pred_sent = sent_pred.cpu().data.numpy()

        for idx in range(len(ts_gather_idx)):
            video_key = key[ts_gather_idx.cpu().data[idx]]
            if video_key not in pred_dict['results']:
                pred_dict['results'][video_key] = list()
            pred_dict['results'][video_key].append({
                'sentence': data_loader.dataset.rtranslate(pred_sent[idx]),
                'timestamp': pred_time[idx].tolist(),
            })

    saver.save_submits(pred_dict, step)
    logger.info('epoch [%05d]: eval finished, elapsed time:%0.4fs, total_generation: %d',
                step, time.time() - _start_time, accumulate_geneneration)
    logger.info('*'*80)


def construct_model(params, saver, logger):

    params['anchor_list'] = ANCHOR_LIST

    if params['checkpoint'] is not None:
        state_dict_sl, state_dict_cg, params_ = saver.load_model_slcg(params['checkpoint'])
        params['anchor_list'] = params_['anchor_list']

    model_sl = SentenceLocalizer(params['hidden_dim'], params['rnn_layer'], params['rnn_cell'], params['rnn_dropout'],
                                 params['bidirectional'], params['attention_type_sl'], params['regressor_scale'],
                                 params['vocab_size'], params['sent_embedding_dim'], params['video_feature_dim'],
                                 params['fc_dropout'], params['anchor_list'], params['feature_mixer_type'],
                                 params['video_use_residual'], params['sent_use_residual'],
                                 params['pe_video'], params['pe_sent'])

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
    if params['checkpoint'] is not None:
        logger.warn('use checkpoint: %s', params['checkpoint'])
        model_sl.load_state_dict(state_dict_sl)
        model_cg.load_state_dict(state_dict_cg)
    if params['checkpoint_cg'] is not None:
        state_dict_cg, _ = saver.load_model(params['checkpoint_cg'])
        logger.warn('use checkpoint: %s', params['checkpoint_cg'])
        model_cg.load_state_dict(state_dict_cg)
    if params['checkpoint_sl'] is not None:
        state_dict_sl, _ = saver.load_model(params['checkpoint_sl'])
        logger.warn('use checkpoint: %s', params['checkpoint_sl'])
        model_sl.load_state_dict(state_dict_sl)

    return model_sl, model_cg


def main(params):
    logger = logging.getLogger(params['alias'])
    gpu_id = set_device(logger, params['gpu_id'])
    logger = logging.getLogger(params['alias'] + '(%d)'%gpu_id)
    set_device(logger, params['gpu_id'])
    saver = ModelSaver(params, os.path.abspath('./third_party/densevid_eval'))
    model_sl, model_cg = construct_model(params, saver, logger)

    model_sl, model_cg = model_sl.cuda(), model_cg.cuda()

    training_set = ANetDataSample(params['train_data'], params['feature_path'],
                                  params['translator_path'], params['video_sample_rate'], logger)
    val_set = ANetDataFull(params['val_data'], params['feature_path'],
                           params['translator_path'], params['video_sample_rate'], logger)
    train_loader_cg = DataLoader(training_set, batch_size=params['batch_size'], shuffle=True,
                              num_workers=params['num_workers'], collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True,
                            num_workers=params['num_workers'], collate_fn=collate_fn, drop_last=True)


    optimizer_cg = torch.optim.SGD(list(chain(model_sl.get_parameter_group(params), model_cg.get_parameter_group(params))),
                                   lr=params['lr'], weight_decay=params['weight_decay'], momentum=params['momentum'])

    optimizer_cg_n = torch.optim.SGD(model_cg.get_parameter_group(params),
                                     lr=params['lr'], weight_decay=params['weight_decay'], momentum=params['momentum'])

    lr_scheduler_cg = torch.optim.lr_scheduler.MultiStepLR(optimizer_cg,
                                                        milestones=params['lr_step'], gamma=params["lr_decay_rate"])

    saver.save_model(model_sl, 0, {'step': 0, 'model_sl': model_sl.state_dict(), 'model_cg': model_cg.state_dict()})
    # eval(model_sl, model_cg, val_loader, logger, saver, params, -1)
    # train_cg(model_cg, model_sl, train_loader_cg, params, logger, -1, optimizer_cg)
    # train_sl(model_cg, model_sl, train_loader_sl, evaluator, params, logger, -1, optimizer_sl)
    for step in range(params['training_epoch']):
        lr_scheduler_cg.step()

        train_cg(model_cg, model_sl, train_loader_cg, params, logger, step, optimizer_cg)

        # validation and saving
        # if step % params['test_interval'] == 0:
        #     eval(model_sl, model_cg, val_loader, logger, saver, params, step)
        if step % params['save_model_interval'] == 0 and step != 0:
            saver.save_model(model_sl, step, {'step': step, 'model_sl': model_sl.state_dict(), 'model_cg': model_cg.state_dict()})



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
                        help='checkpoint')
    parser.add_argument('--checkpoint_sl', type=str, default=None,
                        help='model checkpoint for sentence localization')
    parser.add_argument('--checkpoint_cg', type=str, default=None,
                        help='model checkpoint for caption generation')
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
    parser.add_argument('--fc_dropout', type=float, default=0.3,
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
    parser.add_argument('--regressor_scale', type=float, default=0.3,
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
    parser.add_argument('--training_epoch', type=int, default=100,
                        help='training epochs in total')
    parser.add_argument('--batch_size', type=int, default=31,
                        help='batch size used to train the model')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clip threshold(not used)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate used to train the model')
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
    parser.add_argument('--pretrain_epoch', type=int, default=4,
                        help='pretrain with fake01')
    parser.add_argument('--lr_step', type=int, nargs='+', default=[1, 2, 20],
                        help='lr_steps used to decay the learning_rate')

    parser.add_argument('--refine_round', type=int, default=1,
                        help='round used to refine the model')
    parser.add_argument('--iou_thres', type=float, default=0.9,
                        help='overlapping threshold for temporal segment refine')
    parser.add_argument('--pe_video', type=int, default=100,
                        help='position embedding size used in video embedding(valid when attention_type_cl=type2)')
    parser.add_argument('--pe_sent', type=int, default=10,
                        help='position embedding size used in sent embedding(valid when attention_type_cl=type2)')
    parser.add_argument('--alter_step', type=int, default=5,
                        help='alternate the training between losses')
    params = parser.parse_args()
    params = vars(params)

    main(params)
    print('training finished successfully!')
