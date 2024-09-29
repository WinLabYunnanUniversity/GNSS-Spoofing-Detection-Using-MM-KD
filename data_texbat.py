import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
import numpy as np
import glob, json


class MMtexbat(Dataset):
    print('Loading TEXBAT ... ')
    def __init__(self, data_type, resize_shape=None, crop_size=None, phase="train"):
        with open('/root/autodl-tmp/MM-GNSS/dataset/TEXBAT/obs_normalizer.json', 'r') as f0:
            self.normalizer = json.load(f0)

        self.data_type = data_type
        if phase == "train":
            with open('/root/autodl-tmp/dataset2/TEXBAT/heatmap_train.json', "r", encoding="utf-8") as f:
                self.images = json.load(f)
            self.image_paths = self.images

            with open('/root/autodl-tmp/dataset2/TEXBAT/obs_train.json', "r", encoding="utf-8") as f1:
                self.obs = json.load(f1)
            self.obs_paths = self.obs

            with open('/root/autodl-tmp/dataset2/TEXBAT/spec_train.json', "r", encoding="utf-8") as f:
                self.spec = json.load(f)
                self.spec_paths = self.spec

        else:
            with open('/root/autodl-tmp/dataset2/TEXBAT/heatmap_test.json', "r", encoding="utf-8") as f:
                self.images = json.load(f)
                self.image_paths = self.images

            with open('/root/autodl-tmp/dataset2/TEXBAT/obs_test.json', "r", encoding="utf-8") as f1:
                self.obs = json.load(f1)
            self.obs_paths = self.obs

            with open('/root/autodl-tmp/dataset2/TEXBAT/spec_test.json', "r", encoding="utf-8") as f:
                self.spec = json.load(f)
                self.spec_paths = self.spec

        self.resize_shape = resize_shape
        if (crop_size == None):
            crop_size = resize_shape[0]
        self.transform_img = T.Compose(
            [T.CenterCrop(crop_size), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.phase = phase

    def __len__(self):
        if self.phase == "test":
            return len(self.images)
        else:
            return len(self.image_paths)

    def transform_image(self, image_path):

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32) / 255.0

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = np.asarray(self.transform_img(torch.from_numpy(image)))

        if self.phase == "test":
            return image
        else:
            return image

    def __getitem__(self, idx):

        if self.phase == "test":
            if torch.is_tensor(idx):
                idx = idx.tolist()
            heatmap_path = self.images[idx]
            spec_path = self.spec[idx]
            obs_path = self.obs[idx]
            with open(obs_path, "r", encoding="utf-8") as f22:
                obs = json.load(f22)

            heatmap = self.transform_image(heatmap_path)
            spec = self.transform_image(spec_path)

            normalizer_min = np.array(self.normalizer['min'])
            normalizer_max = np.array(self.normalizer['max'])

            nor_obs = (obs - normalizer_min) / (normalizer_max - normalizer_min)
            nor_obs = np.nan_to_num(nor_obs)

            new_obs = torch.tensor(nor_obs).view(1, 32, 6)

            if '/0/' in heatmap_path:
                has_anomaly = np.array([0], dtype=np.int64)
            else:
                has_anomaly = np.array([1], dtype=np.int64)
            sample = {'heatmap': heatmap,'obs': new_obs, 'spec':spec,'has_anomaly': has_anomaly, 'idx': idx}

        else:
            idx = torch.randint(0, len(self.image_paths), (1,)).item()

            heatmap = self.transform_image(self.images[idx])
            spec = self.transform_image(self.spec[idx])
            obs_ipath = self.obs[idx]
            with open(obs_ipath, "r", encoding="utf-8") as f22:
                obs = json.load(f22)

            normalizer_min = np.array(self.normalizer['min'])
            normalizer_max = np.array(self.normalizer['max'])

            nor_obs = (obs - normalizer_min) / (normalizer_max - normalizer_min)
            nor_obs = np.nan_to_num(nor_obs)

            new_obs = torch.tensor(nor_obs).view(1, 32, 6)

            sample = {'heatmap': heatmap,'obs': new_obs, 'spec':spec}
        return sample
