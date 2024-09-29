import json
import os,warnings
import time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from st_models import teacherTimm, studentTimm, teacherMM, studentMM, teacherTimm1, studentTimm1
import random
from data_texbat import MVtexbat, MMtexbat
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt

# 禁用所有UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''双模态知识蒸馏 数据集： GNSS'''

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class st():
    def __init__(self, data_type,ds_name):
        set_seed(42)#999

        super(st, self).__init__()
        self.data_path = "/root/autodl-tmp/datasets/TEXBAT"
        self.distillType = 'st'
        self.backbone = 'resnet34'
        self.out_indice = [2, 3]
        self.obj = 'TEXBAT'
        self.data_type = data_type
        self.ds_name = ds_name
        self.phase = 'train'  # test  train
        self.save_path = "/root/autodl-tmp/results"
        self.epochs = 30  # 50
        self.batch_size = 16
        self.lr = 0.001 # 0.0035
        self.img_size = 224
        self.crop_size = 224
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm = True  # normalize features before loss calculation

        self.validation_ratio = 0.1
        self.img_resize = self.img_size
        self.img_cropsize = self.crop_size

        # self.teacher = teacherTimm(backbone_name=self.backbone).to(self.device)
        # self.student = studentTimm(backbone_name=self.backbone).to(self.device)

        self.teacher = teacherTimm1(backbone_name=self.backbone).to(self.device)
        self.student = studentTimm1(backbone_name=self.backbone).to(self.device)

        # self.teacher = teacherMM().to(self.device)
        # self.student = studentMM().to(self.device)


        self.model_dir = self.save_path+ "/models" + "/" + self.obj
        os.makedirs(self.model_dir, exist_ok=True)

        if self.obj == "TEXBAT":
            train_dataset = MMtexbat(data_type=self.data_type,
                                     resize_shape=[self.img_resize, self.img_resize],
                                     crop_size=[self.img_cropsize, self.img_cropsize],
                                     phase='train',ds=self.ds_name
                                     )
            img_nums = len(train_dataset)
            valid_num = int(img_nums * self.validation_ratio)
            train_num = img_nums - valid_num
            train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)



        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr * 10,
                                                             epochs=self.epochs,
                                                             steps_per_epoch=len(self.train_loader))

    def loadWeights(self, model, model_dir, alias):
        try:
            checkpoint = torch.load(os.path.join(model_dir, alias))
        except:
            raise Exception("Check saved model path.")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model


    def infer(self, heatmap,spec,obs):
        features_t = self.teacher(heatmap,spec,obs)
        features_s = self.student(heatmap,spec,obs)
        return features_s, features_t

    @torch.no_grad()
    def cal_anomaly_maps(self, fs_list, ft_list, out_size, norm):
        anomaly_map = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            fs_norm = F.normalize(fs, p=2) if norm else fs
            ft_norm = F.normalize(ft, p=2) if norm else ft

            _,_,h, w = fs.shape

            a_map = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)
            # print(a_map.shape)
            # a,b,c,d = ft_norm.shape
            # a_map = 1 - F.cosine_similarity(fs_norm, ft_norm,0).view(a,b,c,d)
            # print(a_map.shape)
            a_map = a_map.sum(1, keepdim=True)

            a_map = F.interpolate(
                a_map, size=out_size, mode="bilinear", align_corners=False
            )
            anomaly_map += a_map
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

        return anomaly_map

    def computeAUROC(self,scores, gt_list, obj, name="base"):
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print(obj + " image" + str(name) + " ROCAUC: %.3f" % (img_roc_auc))

        plt.plot(img_scores)
        plt.savefig('/root/autodl-tmp/results/imgs/TEXBAT/scores_%s.png'%(self.data_type))
        plt.show()
        return img_roc_auc, img_scores


    def cal_loss(self, fs_list, ft_list, norm):
        t_loss = 0
        N = len(fs_list)
        for i in range(N):
            fs = fs_list[i]#.view(self.batch_size,-1)
            ft = ft_list[i]#.view(self.batch_size,-1)
            _, _, h, w = fs.shape
            # 功能：将某一个维度fs除以那个维度对应的2范数
            fs_norm = F.normalize(fs, p=2) if norm else fs
            ft_norm = F.normalize(ft, p=2) if norm else ft

            f_loss = 0.5 * (ft_norm - fs_norm) ** 2
            # f_loss = 1 - F.cosine_similarity(fs_norm, ft_norm, 0)
            f_loss = f_loss.sum() / (h * w)
            # f_loss = f_loss.sum() / len(fs)
            t_loss += f_loss

        return t_loss / N

    def computeLoss(self, features_s, features_t):
        loss = self.cal_loss(features_s, features_t, self.norm)
        return loss

    def val(self, epoch_bar):
        self.student.eval()
        losses = AverageMeter()
        for sample in self.val_loader:
            # sample = {'heatmap': heatmap,'obs': new_obs, 'spec':spec}
            heatmap = sample['heatmap'].to(self.device)
            obs = sample['obs'].to(self.device)
            spec = sample['spec'].to(self.device)

            with torch.set_grad_enabled(False):
                features_s, features_t = self.infer(heatmap,spec, obs)
                loss = self.computeLoss(features_s, features_t)
                losses.update(loss.item(), heatmap.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student_%s_%s.pth"%(self.data_type,self.ds_name)))

    @torch.no_grad()
    def test(self):
        test_res_list = {}
        self.student = self.loadWeights(self.student, self.model_dir, "student_%s_%s.pth"%(self.data_type,self.ds_name))
        kwargs = ({"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {})
        test_dataset = MMtexbat(
            data_type=self.data_type,
            resize_shape=[self.img_resize, self.img_resize],
            crop_size=[self.img_cropsize, self.img_cropsize],
            phase='test',ds=self.ds_name
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        scores = []
        test_imgs = []
        gt_list = []
        progressBar = tqdm(test_loader)
        for sample in test_loader:
            # sample = {'heatmap': heatmap,'obs': new_obs, 'spec':spec}
            heatmap = sample['heatmap'].to(self.device)
            obs = sample['obs'].to(self.device)
            # spec = sample['spec'].to(self.device)
            label = sample['has_anomaly'].to(self.device)

            # test_imgs.extend(heatmap.cpu().numpy())

            gt_list.extend(label.cpu().numpy())
            spec = 0
            with torch.set_grad_enabled(False):
                # The shape of features_s and features_t  are
                features_s, features_t = self.infer(heatmap,spec,obs)
                score = self.cal_anomaly_maps(features_s, features_t, self.img_cropsize, self.norm)
                progressBar.update()
            scores.append(score)
        print('len(gt_list) = ', len(gt_list))
        progressBar.close()
        scores = np.asarray(scores)
        gt_list = np.asarray(gt_list)
        # gt_list[-5:]=1
        img_roc_auc, img_scores = self.computeAUROC(scores, gt_list, self.obj, " " + self.distillType)
        test_path = test_dataset.image_paths
        test_res_list['img_scores'] = img_scores.tolist()
        test_res_list['test_path'] = test_path
        # test_res_list['scores'] = scores.tolist()
        test_res_list['label'] = gt_list.tolist()

        with open('/root/autodl-tmp/results/valSet_result_texbat_%s.json'%self.ds_name, 'w', encoding='utf8') as f:
            json.dump(test_res_list, f, ensure_ascii=False)
        return img_roc_auc

    def train(self):
        print("training " + self.obj)
        self.student.train()
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.epochs, desc="Training", unit="batch")
        hist = []

        for _ in range(1, self.epochs + 1):
            losses = AverageMeter()

            for sample in self.train_loader:
                # sample = {'heatmap': heatmap,'obs': new_obs, 'spec':spec}
                heatmap = sample['heatmap'].to(self.device)
                obs = sample['obs'].to(self.device)
                spec = sample['spec'].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    features_s, features_t = self.infer(heatmap,spec,obs)
                    loss = self.computeLoss(features_s, features_t)

                    losses.update(loss.sum().item(), heatmap.size(0))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    hist.append(loss.item())

                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()

            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

            epoch_time.update(time.time() - start_time)
            start_time = time.time()
        epoch_bar.close()
        plt.plot(hist)
        plt.savefig('/root/autodl-tmp/results/imgs/TEXBAT/trainingLoss_%s_%s.png' % (self.data_type,self.ds_name))
        plt.close()

        print("Training end.")


if __name__ == "__main__":
    trainer = st(data_type='MM_texbat2',ds_name='ds6')
    # trainer.train()
    trainer.test()
