import sys
sys.path.append('./third_party/densevid_eval/coco-caption')

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor

src = {0: [{'caption': "this could be a good time, but not then."}]}
tgt = {0: [{'caption': "this is not good at all, time will say."}]}
src_1 = {0: ["this could be a good time, but not then."]}
tgt_1 = {0: ["this is not good at all, time will say."]}

src = {0: [{'caption': "1 32 3 4 5 111 99 4256 12341 666 555"}]}
tgt = {0: [{'caption': "1 12312 12341 5 7456 8646 99 111 1211 12312 555"}]}
src_1 = {0: ["1 32 3 4 5 111 99 4256 12341 666 555"]}
tgt_1 = {0: ["1 12312 12341 5 7456 8646 99 111 1211 12312 555"]}


tokenizer = PTBTokenizer()
meteor = Meteor()

src_t = tokenizer.tokenize(src)
tgt_t = tokenizer.tokenize(tgt)

score = meteor.compute_score(src_t, tgt_t)
score_1 = meteor.compute_score(src_1, tgt_1)
import pdb; pdb.set_trace()