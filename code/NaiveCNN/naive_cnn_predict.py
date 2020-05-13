from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys

import tensorflow as tf


model = sys.argv[1]
if not isfile(model):
    print("no such file")
    exit(1)

classifierLoad = tf.keras.models.load_model(model)

import numpy as np
from keras.preprocessing import image
beg = max(model.rindex('/') + 1, 0)
end = model.index('.')

loc = 'data/'
subdirs = ['C1-P1_Dev', 'C1-P1_Train']

with open('{}_summary.txt'.format(model[beg:end]), "w") as sumfile:
    for dset in subdirs:
        summary = {}
        for k in "ABC":
            summary[k] = {'c': 0, 'm': 0, 't': 0}
            mypath = loc + dset + '/' + k
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            t = len(onlyfiles)
            summary[k]['t'] = t
            print('class {}:'.format(k))
            for fn in tqdm(onlyfiles):
                test_image = image.load_img(join(mypath, fn), target_size = (200,200))
                #test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                result = classifierLoad.predict(test_image)
                f = np.where(result[0] == 1)[0]
                if len(f):
                    p = ['A', 'B', 'C'][f[0]]
                    if p == k:
                        summary[k]['c'] += 1
                else:
                    summary[k]['m'] += 1

        tc = 0; tm = 0; tt = 0;
        for k in "ABC":
            v = lambda a: summary[k][a]
            c = v('c'); m = v('m'); t = v('t')
            tc += c;
            tm += m;
            tt += t;
            l = '{} class {}: {}%, good: {}, bad: {}, miss: {}, total: {}'.format(
                dset, k, c/t * 100, c, t-c-m, m, t)
            print(l)
            sumfile.write(l + '\n')

        l = '{} class {}: {}%, good: {}, bad: {}, miss: {}, total: {}'.format(
            dset, 'total', tc/tt * 100, tc, tt-tc-tm, tm, tt)
        print(l)
        sumfile.write(l + '\n')
