import numpy as np
import argparse
import os
from PIL import Image
import imageio
from shutil import copyfile
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', default='/home3/qianchen/data/VOCdevkit/JPEGImages/', help='The path of the source image')
parser.add_argument("--crfh_sal", default='./result/crf_sal/crf_sal_h', type=str)
parser.add_argument("--crfl_sal", default='./result/crf_sal/crf_sal_l', type=str)
parser.add_argument("--outsaldir", default='./result/final_sal', type=str)
args = parser.parse_args()

img_name_list = np.loadtxt('./train.txt', dtype=np.int32)
all_img = len(img_name_list)
cls_labels_dict = np.load('./voc12/cls_labels.npy', allow_pickle=True).item()

for curimg in range(all_img):
    s = str(int(img_name_list[curimg]))
    img_name = s[:4] + '_' + s[4:]
    print(img_name)
    img_name = str(img_name)


    cur_label = cls_labels_dict[img_name]
    cur_set = set()
    for i in range(len(cur_label)):
        if cur_label[i] == 1:
            cur_set.add(i + 1)
    if len(cur_set) == 1:
        source = os.path.join(args.crfh_sal,'%s.png'%img_name)
        topath = os.path.join(args.outsaldir, '%s.png'%img_name)
        copyfile(source, topath)
        continue
    else:
        predict_file = os.path.join(args.crfh_sal,'%s.png'%img_name)
        ori_sal = np.array(Image.open(predict_file))
        crf1_file = os.path.join(args.crfl_sal, '%s.png'%img_name)
        ori_dict = Counter(ori_sal.flatten())
        crf1_sal = np.array(Image.open(crf1_file))
        crf1_dict = Counter(crf1_sal.flatten())
        
        if crf1_dict[0] > ori_dict[0] * 6:
            ss = np.argwhere(crf1_sal==0)
            for j in ss:
                x = j[0]
                y = j[1]
                ori_sal[x][y] = 0
        
        
        for i in cur_set:
            #if ori_dict[i] == 0 or crf1_dict[i] * 0.5 > ori_dict[i]:
            if crf1_dict[i] > ori_dict[i]:
                ss = np.argwhere(crf1_sal==i)
                for j in ss:
                    x = j[0]
                    y = j[1]
                    ori_sal[x][y] = i
        



        imageio.imwrite(os.path.join(args.outsaldir, img_name + '.png'), ori_sal)




                