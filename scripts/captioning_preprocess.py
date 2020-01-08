import json
import pickle
import argparse
import logging
from collections import Counter, defaultdict
from itertools import chain

logging.basicConfig()


def build_vocab(video_caption_dict, params, logger=None):
    if logger is None:
        logger = logging.getLogger('build_vocab')
        logger.setLevel(logging.INFO)

    all_annoations = [x['sentences'] for x in video_caption_dict.values()]
    max_captioning_number = max([len(x['sentences']) for x in video_caption_dict.values()])
    total_video = len(all_annoations)
    logger.info('total captioned video: %d', total_video)
    all_annoations = list(chain(*all_annoations))
    logger.info('total captioning sentence: %d', len(all_annoations))
    logger.info('average captioning per video: %f; max captioning number: %d',
                len(all_annoations)/float(total_video), max_captioning_number)
    total_sentence = len(all_annoations)
    all_annoations = ' '.join(all_annoations).replace('.', ' . ').replace(',', ' , ').lower().split()
    logger.info('total words: %d', total_sentence)
    logger.info('average word number per sentence: %f', len(all_annoations) / total_sentence)
    words = Counter(all_annoations)
    logger.info('total unique  words: %d', len(words))
    words = {key:val for key, val in words.iteritems() if val > params['word_count_threshold']}
    logger.info('valid word size: %d', len(words))
    count_pairs = sorted(words.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    if params['vocab_size'] != -1 and len(words) > params['vocab_size']:
        words = words[:params['vocab_size'] - 3]  # -3 for oov words, bos, eos

    word_to_id = defaultdict(lambda: -1)
    for i, word in enumerate(words):
        word_to_id[word] = i + 2
    word_to_id['<bos>'] = 0
    word_to_id['<eos>'] = 1
    id_to_word = ('<bos>', '<eos>') + words + ('<oov>',)
    logger.info('total vocab size: %d', len(id_to_word))

    return dict(word_to_id), id_to_word


def main(params):
    video_caption_dict = json.load(open(params['input_json'], 'r'))

    logger = logging.getLogger('captioning_pre process')
    logger.setLevel(logging.INFO)
    word_to_id, id_to_word = build_vocab(video_caption_dict, params, logger)
    translator = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word
    }
    pickle.dump(translator, open(params['out_path'] + str(params['vocab_size']) + '.pkl', 'w'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='data/densecap/train.json',
                        help='anet captioning data')
    parser.add_argument('--out_path', type=str, default='data/translator',
                        help='captioning dict between words and indexes')
    parser.add_argument('--vocab_size', type=int, default=6000,
                        help='vocabulary size used in final model')
    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
