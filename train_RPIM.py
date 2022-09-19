
import numpy as np
import torch
import random
import cv2
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
from time import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import copy
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_RPIM", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", default="ilsvrc-cls_rna-a1_cls1000_ep-0001.params", type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))
    
    model = getattr(importlib.import_module(args.network), 'Net')()

    print(model)

    tblogger = SummaryWriter(args.tblog_dir)	

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_refine')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1,scale_factor=scale_factor,mode='bilinear',align_corners=True) 
            N,C,H,W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            cam1, cam_rv1 = model(img1)
            oriallcam = cam_rv1
            label1 = F.adaptive_avg_pool2d(cam1, (1,1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1*label)[:,1:,:,:])
            cam1 = F.interpolate(visualization.max_norm(cam1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label

            cam2, cam_rv2 = model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1,1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2*label)[:,1:,:,:])
            cam2 = visualization.max_norm(cam2)*label
            cam_rv2 = visualization.max_norm(cam_rv2)*label

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:,1:,:,:], label[:,1:,:,:])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:,1:,:,:], label[:,1:,:,:])

            ns,cs,hs,ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:,1:,:,:]-cam2[:,1:,:,:]))
            cam1[:,0,:,:] = 1-torch.max(cam1[:,1:,:,:],dim=1)[0]
            cam2[:,0,:,:] = 1-torch.max(cam2[:,1:,:,:],dim=1)[0]

            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)#*eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)#*eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            def superpixel(oricam, img_name, label, negthres, upthres, args):
                file_path = os.path.join('super', str(img_name) + '.pkl')
                img_path = voc12.data.get_img_path(img_name, args.voc12_root)
                img = np.asarray(Image.open(img_path))
                orig_img_size = img.shape[:2]
                oricam = F.upsample(oricam[:,:,:].unsqueeze(0), orig_img_size, mode='bilinear', align_corners=False)[0]
                oricam = oricam.cpu().detach().numpy() * label.clone().view(20, 1, 1).numpy()
                
                with open(file_path, 'rb') as f:
                    obj = pickle.loads(f.read())
                    nregion = obj[0]
                    spix = obj[1]
                    neighbors = obj[2]
                    inpoint = obj[3]
                f.close()

                sum_cam = oricam
                sum_cam[sum_cam < 0] = 0
                cam_max = np.max(sum_cam, (1,2), keepdims=True)
                cam_min = np.min(sum_cam, (1,2), keepdims=True)
                sum_cam[sum_cam < cam_min+1e-5] = 0
                high_res = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)
                oricam = copy.deepcopy(high_res)

                valid_cat = torch.nonzero(label)[:, 0]
                handle_map = copy.deepcopy(high_res)
                max_region = 1000
                single_loss = 0.0
                nclass = len(valid_cat)
                for key in valid_cat:
                    count = 0
                    countsum = 0.0
                    key = key.item()
                    neighbor = copy.deepcopy(neighbors)
                    while True:
                        max_index = np.unravel_index(handle_map[key].argmax(), handle_map[key].shape)
                        if handle_map[key][max_index[0]][max_index[1]] < upthres:
                            break
                        cur_region = spix[max_index[0]][max_index[1]]
                        if max_region == 1000 or cur_region not in neighbor[max_region]:  
                            cur_score = handle_map[key][max_index[0]][max_index[1]]
                            max_region = cur_region
                        negsum = 0.0
                        negcount = 0
                        upcount = 0
                        for j in inpoint[cur_region]:
                            x, y = j.split(',', 1)
                            x = int(x)
                            y = int(y)
                            if handle_map[key][x][y] < negthres:
                                negsum = negsum + handle_map[key][x][y]
                                negcount = negcount + 1
                            if handle_map[key][x][y] >= upthres:
                                upcount = upcount + 1
                            handle_map[key][x][y] = 0.0
                        if negcount > len(inpoint[cur_region]) * 0.7:
                            negavg = negsum / negcount
                            for j in inpoint[cur_region]: 
                                x, y = j.split(',', 1)
                                x = int(x)
                                y = int(y)
                                if high_res[key][x][y] > negavg:
                                    count += 1
                                    countsum += abs(high_res[key][x][y] - negavg)
                                    high_res[key][x][y] = negavg
                            max_region = 1000
                        elif upcount > len(inpoint[cur_region]) * 0.7:
                            for j in inpoint[cur_region]:
                                count += 1
                                x, y = j.split(',', 1)
                                x = int(x)
                                y = int(y)
                                countsum += abs(high_res[key][x][y] - cur_score)
                                high_res[key][x][y] = cur_score
                            max_region = cur_region
                        else:
                            max_region = 1000
                    if count == 0:
                        nclass = nclass - 1
                    else:
                        single_loss += countsum / count

                if nclass == 0:
                    single_loss = 0.0
                else:
                    single_loss = single_loss / nclass
                return single_loss


            if ep > 0:
                batch_loss = 0.0
                upthres = 0.75 - 7e-5* (optimizer.global_step - 1)
                negthres = 3.5e-5 * (optimizer.global_step - 1) - 0.075
                for ibatch in range(args.batch_size):
                    mylabel = pack[2][ibatch]
                    oricam = oriallcam[ibatch]
                    img_name = pack[0][ibatch]
                    sucam = oricam[1:, :, :]
                    single_loss = superpixel(sucam, img_name, mylabel, negthres, upthres, args)
                    batch_loss = batch_loss + single_loss

                loss_refine = batch_loss / args.batch_size
            
            else:
                loss_refine = 0.0

            loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1 + loss_rvmin2)/2 
            loss = loss_cls + loss_er + loss_ecr + loss_refine

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item(), 'loss_refine': loss_refine})

            if (optimizer.global_step - 1) % 50 == 0:

                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f %.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_refine'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()
                loss_dict = {'loss':loss.item(), 
                             'loss_cls':loss_cls.item(),
                             'loss_er':loss_er.item(),
                             'loss_ecr':loss_ecr.item(),
                             'loss_refine':loss_refine}
                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)

        else:
            print('')
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')

         
