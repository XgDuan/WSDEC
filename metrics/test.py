import sys
sys.path.append('../third_party/densevid_eval/coco-caption')

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor

src = {0: [{'caption': "this could be a good time, but not then."}]}
tgt = {0: [{'caption': "this is not good at all, time will say."}]}
src_1 = {0: ["this could be a good time, but not then."]}
tgt_1 = {0: ["this is not good at all, time will say."]}

tokenizer = PTBTokenizer()
meteor = Meteor()

src_t = tokenizer.tokenize(src)
tgt_t = tokenizer.tokenize(tgt)

score = meteor.compute_score(src_t, tgt_t)
score_1 = meteor.compute_score(src_1, tgt_1)
import pdb; pdb.set_trace()