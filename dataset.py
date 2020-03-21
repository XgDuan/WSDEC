from collections import namedtuple, defaultdict
from itertools import chain
import json
import h5py
import pickle

from global_config import *
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    "rearrange the data returned by the batch"

    batch_size = len(batch)
    feature_size = batch[0][0].shape[1]
    feature_list, timestamps_list, caption_list, raw_timestamp, raw_duration, key = zip(*batch)

    max_video_length = max([x.shape[0] for x in feature_list])
    max_caption_length = max(chain(*[[len(caption) for caption in captions] for captions in caption_list ]))
    total_caption_num = sum(chain([len(captions) for captions in caption_list ]))

    video_tensor = torch.FloatTensor(batch_size, max_video_length, feature_size).zero_()
    video_length = torch.FloatTensor(batch_size, 2).zero_()  # true length, sequence length
    video_mask = torch.FloatTensor(batch_size, max_video_length, 1).zero_()

    timestamps_tensor = torch.FloatTensor(total_caption_num, 2).zero_()

    caption_tensor = torch.LongTensor(total_caption_num, max_caption_length).zero_() + EOS_ID
    caption_length = torch.LongTensor(total_caption_num).zero_()
    caption_mask = torch.FloatTensor(total_caption_num, max_caption_length, 1).zero_()
    caption_gather_idx = torch.LongTensor(total_caption_num).zero_()

    total_caption_idx = 0

    for idx in range(batch_size):
        video_len = feature_list[idx].shape[0]
        #video
        video_tensor[idx,:video_len,:] = torch.from_numpy(feature_list[idx])
        video_length[idx,0] = float(video_len)
        video_length[idx,1] = raw_duration[idx]
        video_mask[idx, :video_len, 0] = 1
        # timestamps
        proposal_length = len(timestamps_list[idx])
        timestamps_tensor[total_caption_idx:total_caption_idx+proposal_length, :] = \
            torch.from_numpy(timestamps_list[idx])
        caption_gather_idx[total_caption_idx:total_caption_idx+proposal_length] = idx
        #caption
        for iidx, captioning in enumerate(caption_list[idx]):
            _caption_len = len(captioning)
            caption_length[total_caption_idx+iidx] = _caption_len
            caption_tensor[total_caption_idx+iidx, :_caption_len] = torch.from_numpy(captioning)
            caption_mask[total_caption_idx+iidx, :_caption_len, 0] = 1
        total_caption_idx += proposal_length
    raw_timestamp = torch.FloatTensor(list(chain(*raw_timestamp)))
    # print((video_length[:,0] / video_length[:,1]).mean())

    return (video_tensor, video_length, video_mask,
            caption_tensor, caption_length, caption_mask, caption_gather_idx,
            raw_timestamp, timestamps_tensor, key)


class ANetData(Dataset):

    def __init__(self, caption_file, feature_file, translator_pickle, feature_sample_rate, logger):
        """
        :param caption_file: the location caption stored
        :param feature_file: the location feature stored
        :param shuffle_data: whether the file should be shuffled or not
        :return: video_info_obj: VideoInfo
        """
        super(ANetData, self).__init__()
        self.captioning = json.load(open(caption_file, 'r'))
        self.keys = list(self.captioning.keys())
        logger.info('load captioning file, %d captioning loaded', len(self.keys))

        self.feature_file = h5py.File(feature_file, 'r')
        logger.info('load video feature file, %d video feature obj(%s) loaded',
                    len(self.feature_file.keys()),
                    self.feature_file[self.keys[0]]['c3d_features'][0].shape)
        # we should initialize the file handle in the subprocess (otherwise we will encounter bugs when using more than one workers)
        self.feature_file.close()
        self.feature_file = feature_file

        self.translator = pickle.load(open(translator_pickle, 'r'))
        self.translator['word_to_id'] = defaultdict(lambda: len(self.translator['id_to_word'])-1,
                                                    self.translator['word_to_id'])

        logger.info('load translator, total_vocab: %d', len(self.translator['id_to_word']))

        self.sample_rate = feature_sample_rate

    def __len__(self):
        return len(self.keys)

    def translate(self, sentence):
        sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
        sentence_split = ['<bos>'] + sentence_split + ['<eos>']
        res = np.array([self.translator['word_to_id'][word] for word in sentence_split])
        return res

    def rtranslate(self, sent_ids):
        assert sent_ids[0] == self.translator['word_to_id']['<bos>']
        sent_ids = sent_ids[1:]
        for i in range(len(sent_ids)):
            if sent_ids[i] == self.translator['word_to_id']['<eos>']:
                sent_ids = sent_ids[:i]
                break
        #while len(sent_ids) > 0 and sent_ids[-1] == self.translator['word_to_id']['<eos>']:
        #    sent_ids = sent_ids[:-1]
        return ' '.join([self.translator['id_to_word'][idx] for idx in sent_ids])

    def process_time_step(self, duration, timestamps_list, feature_length):
        res = np.zeros([len(timestamps_list), 2])
        for idx, (start, end) in enumerate(timestamps_list):
            start_, end_ = int(feature_length*start/duration), int(feature_length*end/duration)
            end_ = min(end_, feature_length-1)
            res[idx] = np.array([start_, end_])
        return res

    def __getitem__(self, idx):
        raise NotImplementedError()


class ANetDataFull(ANetData):

    def __init__(self, caption_file, feature_file, translator_pickle, feature_sample_rate, logger):
        super(ANetDataFull, self).__init__(caption_file,
            feature_file, translator_pickle, feature_sample_rate, logger)

    def __getitem__(self, idx):
        if isinstance(self.feature_file, str):
            self.feature_file = h5py.File(self.feature_file, 'r')
        key = str(self.keys[idx])
        feature_obj = self.feature_file[key]['c3d_features']
        feature_obj = feature_obj[::self.sample_rate, :]
        captioning = self.captioning[key]['sentences']

        captioning = [np.array(self.translate(sent)) for sent in captioning]
        timestamps = self.captioning[key]['timestamps']

        duration = self.captioning[key]['duration']
        processed_timestamps = self.process_time_step(duration, timestamps, feature_obj.shape[0])
        return feature_obj, processed_timestamps, captioning, timestamps, duration, key


class ANetDataSample(ANetData):

    def __init__(self, caption_file, feature_file, translator_pickle, feature_sample_rate, logger):
        super(ANetDataSample, self).__init__(caption_file,
            feature_file, translator_pickle, feature_sample_rate, logger)

    def __getitem__(self, idx):
        if isinstance(self.feature_file, str):
            self.feature_file = h5py.File(self.feature_file, 'r')
        key = str(self.keys[idx])
        feature_obj = self.feature_file[key]['c3d_features']
        feature_obj = feature_obj[::self.sample_rate, :]
        captioning = self.captioning[key]['sentences']

        idx = int(np.random.choice(range(len(captioning)), 1))
        captioning = [[np.array(self.translate(sent)) for sent in captioning][idx]]
        timestamps = [self.captioning[key]['timestamps'][idx]]

        duration = self.captioning[key]['duration']
        processed_timestamps = self.process_time_step(duration, timestamps, feature_obj.shape[0])
        return feature_obj, processed_timestamps, captioning, timestamps, duration, key



if __name__ == '__main__':
    import logging
    logging.basicConfig()
    logger = logging.getLogger('dataset')
    logger.setLevel(logging.INFO)
    dataset = ANetData('data/densecap/train.json',
                       'data/anet_v1.3.c3d.hdf5',
                       'data/translator.pkl', 1,
                       logger)
    data_loader = DataLoader(dataset, batch_size=64,
        shuffle=True, num_workers=2, collate_fn=collate_fn)
    for dt in data_loader:
        for tensor in dt[:-1]:
            print(type(tensor), tensor.size())
        print(dt[-1])
        print('*'*80)
    logger.info('test_done')

