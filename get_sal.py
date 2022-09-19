import numpy as np
import argparse
import imutils
import os
from PIL import Image
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--oricrfdir', default='./result/crf', help='The path of the source cam')
parser.add_argument('--imgdir', default='./VOC2012/JPEGImages/', help='The path of the source image')
parser.add_argument("--sal_path", default='saliency_map_train_aug', type=str) 
parser.add_argument("--out_sal", required=True, type=str)
parser.add_argument("--theta", default=0.5, type=float)
parser.add_argument("--alpha", default=0, type=float)
parser.add_argument("--beta", default=1, type=float)
args = parser.parse_args()

img_name_list = np.loadtxt('./train.txt', dtype=np.int32)
all_img = len(img_name_list)

for curimg in range(all_img):
    s = str(int(img_name_list[curimg]))
    img_name = s[:4] + '_' + s[4:]
    print(img_name)
    oricam_path = os.path.join(args.oricrfdir, img_name + '.npy')

    crf_dict = np.load(oricam_path, allow_pickle=True).item()
    h, w = list(crf_dict.values())[0].shape
    tensor = np.zeros((21, h, w), np.float32)
    for key in crf_dict.keys():
        tensor[key] = crf_dict[key]
    if args.out_sal is not None:
        sal_file = os.path.join(args.sal_path, img_name + '.png')
        sal_map = np.array(Image.open(sal_file))
        sal_map = np.where(sal_map <= args.theta * 100, 1, 0)
        tensor = np.where(tensor < args.alpha, -2, tensor) #tensor中得分小于alpha的输出-2，其余保持tensor
        tensor = np.where(tensor > args.beta, 2, tensor) #tensor中得分大于beta的输出2，其余保持tensor
        tensor[0, :, :] = sal_map
    pred = np.argmax(tensor, axis=0)
    imageio.imwrite(os.path.join(args.out_sal, img_name + '.png'), pred.astype(np.uint8))

    