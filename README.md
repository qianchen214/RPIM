# RPIM

This repository contains the pytorch codes and trained models of RPIM(Region-based Pixels Integration Mechanism for Weakly Supervised Semantic Segmentation).



## Env

Our model was trained with Python 3.9, PyTorch 1.10.0 and 4 GeForce RTX 2080 Ti with 11GB memory. 

### Other Dependencies

```
pip install -r requirements.txt
```



## Pretrain Models

We provide the [pretrained weights](https://drive.google.com/drive/folders/1UKCekBdWpp07C5arlGKHyVYSnBDr25fz?usp=sharing) for CAM.

We train the [Deeplab v2 Network](https://github.com/kazuto1011/deeplab-pytorch ) using the generated [pseudo-labels](https://drive.google.com/drive/folders/1fgzROcresg-pgn_oOaeot-El8hjtBlTG?usp=sharing).



## Dataset

[PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) is the main dataset we used to do the experiments. 

We use functions in OpenCV to get [superpixels](https://drive.google.com/drive/folders/1-7V0rX0HKkdw87O9PFzuRBjVc-30caAa?usp=sharing) and [Saliency map](https://drive.google.com/file/d/1ENS6jR6EUIDtxWsYwgwZ9YELSp0tFQ0k/view) from [EDAM](https://github.com/allenwu97/EDAM) in post-processing method.



## Usage

link the dataset to your folder

```
ln -s /home/data/VOCdevkit VOC2012
ln -s /home/data/super super
```

### Train

```
python train_RPIM.py 
```



### Inference and Post-process

The result folder should be:

```
result
	> crf
	> crf_sal
	> final_sal
```



```
python infer_RPIM.py --weights ./resnet38_RPIM.pth --out_cam result/crf

python get_sal.py --oricrfdir ./result/crf/crf_1.0 --out_sal ./result/crf_sal/crf_sal_l --theta 0.2 --alpha 0.000001 --beta 0.99999

python get_sal.py --oricrfdir ./result/crf/crf_4.0 --out_sal ./result/crf_sal/crf_sal_h --theta 0.3 --alpha 0.0000002 --beta 0.9999

python sal_hl.py
```



### Eval

```
python evaluation.py --comment "[your comment]"
```



## Acknowledgement

We use [SEAM](https://github.com/YudeWang/SEAM) as our baseline and our post-processing method is based on [EDAM](https://github.com/allenwu97/EDAM). Thanks for their work.

