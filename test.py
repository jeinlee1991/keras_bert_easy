import os
import time
import numpy as np

from keras_bert_easy.bert import get_model
from keras_bert_easy.loader import build_pretrained_model
from keras_bert import load_trained_model_from_checkpoint
from bert4keras.models import build_transformer_model

# model = get_model(token_num=20000, seq_len=None)
# model.summary()

if os.name == 'nt':
    pretrained_path = r'D:/work/bert/chinese_L-12_H-768_A-12/'
else:
    pretrained_path = '/home/washington/project/bert/chinese_L-12_H-768_A-12/'
checkpoint_file = os.path.join(pretrained_path, 'bert_model.ckpt')
config_file = os.path.join(pretrained_path, 'bert_config.json')


if __name__=='__main__':
    # ids = np.random.randint(0,1000, size=(1,10))
    ids = np.array([[1,2,3,0,0]])
    segs = np.zeros_like(ids)
    tstart = time.perf_counter()
    model = build_pretrained_model(checkpoint_file=checkpoint_file)
    print('load ckpt time cost: %.4f' % (time.perf_counter() - tstart))
    # model.summary()
    r = model.predict([ids, segs])
    print(r[:,0,0:2])

    # tstart = time.perf_counter()
    # model = load_trained_model_from_checkpoint(config_file, checkpoint_file, seq_len=None)
    # print('load ckpt time cost: %.4f' % (time.perf_counter() - tstart))
    # # model.summary()
    # r = model.predict([ids, segs])
    # print(r[:,0,0:2])

    tstart = time.perf_counter()
    model = build_transformer_model(config_file, checkpoint_file)
    print('load ckpt time cost: %.4f' % (time.perf_counter() - tstart))
    # model.summary()
    r = model.predict([ids, segs])
    print(r[:,0,0:2])