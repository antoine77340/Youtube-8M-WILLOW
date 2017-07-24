# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




import os
from collections import defaultdict, Counter
import pickle
import pandas as pd

SUBMIT_PATH = ''
SIGFIGS = 6

def read_models(model_weights, blend=None):
    if not blend:
        blend = defaultdict(Counter)
    for m, w in model_weights.items():
        print(m, w)
        with open(os.path.join(SUBMIT_PATH, m + '.csv'), 'r') as f:
            f.readline()
            for l in f:
                id, r = l.split(',')
                id, r = int(id), r.split(' ')
                n = len(r) // 2
                for i in range(0, n, 2):
                    k = int(r[i])
                    v = int(10**(SIGFIGS - 1) * float(r[i+1]))
                    blend[id][k] += w * v
    return blend


def write_models(blend, file_name, total_weight):
    with open(os.path.join(SUBMIT_PATH, file_name + '.csv'), 'w') as f:
        f.write('VideoID,LabelConfidencePairs\n')
        for id, v in blend.items():
            l = ' '.join(['{} {:{}f}'.format(t[0]
                                            , float(t[1]) / 10 ** (SIGFIGS - 1) / total_weight
                                            , SIGFIGS) for t in v.most_common(20)])
            f.write(','.join([str(id), l + '\n']))
    return None


model_pred = {'test-gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe': 1
                  , 'test-GRU-0002-1200-2': 1
                  , 'test-gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe': 1
                  , 'test-gateddboflf-4096-1024-80-0002-300iter': 1
                  , 'test-softdboflf-8000-1024-80-0002-300iter': 1
                  , 'test-gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe': 1
                  , 'test-lstm-0002-val-150-random': 1
                 }

avg = read_models(model_pred)
write_models(avg, 'WILLOW_submission', sum(model_pred.values()))
