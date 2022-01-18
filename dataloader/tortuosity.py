from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
import random
import warnings
import csv
import numpy as np
import numbers

warnings.filterwarnings('ignore')


def load_dataset(root_dir, train="train"):
    images_path = []
    segs_path = []
    rois_path = []
    class_id = []
    if train == "train":
        sub_dir = 'training/train'
    elif train == "val":
        sub_dir = 'training/val'
    elif train == "test":
        sub_dir = 'test'
    with open(os.path.join(root_dir, sub_dir + ".csv"), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for item in reader:
            img_name, seg_name, roi_name, id = item
            img_path = os.path.join(root_dir, sub_dir, img_name)
            seg_path = os.path.join(root_dir, sub_dir, seg_name)
            roi_path = os.path.join(root_dir, sub_dir, roi_name)
            segs_path.append(seg_path)
            images_path.append(img_path)
            rois_path.append(roi_path)
            class_id.append(int(id))
    return images_path, segs_path, rois_path, class_id


class MyData(Dataset):
    def __init__(self,
                 root_dir,
                 train="train",
                 rotate=45,
                 flip=True,
                 random_crop=True,
                 size=304):

        self.root_dir = root_dir
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.resize = size
        self.random_crop = random_crop
        self.transform_train = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        self.transform_roi = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        self.images_path, self.segs_path, self.roi_path, self.class_id = load_dataset(self.root_dir, self.train)

    def __len__(self):
        return len(self.images_path)

    def RandomCrop(self, img, seg, roi, crop_size):
        if isinstance(crop_size, numbers.Number):
            crop_size = (int(crop_size), int(crop_size))
        else:
            crop_size = crop_size
        w, h = img.size
        th, tw = crop_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img = F.crop(img, i, j, h, w)
        seg = F.crop(seg, i, j, h, w)
        roi = F.crop(roi, i, j, h, w)
        return img, seg, roi

    def RandomHorizonalFlip(self, img, seg, roi, p):
        if random.random() < p:
            img = F.hflip(img)
            seg = F.hflip(seg)
            roi = F.hflip(roi)
        return img, seg, roi

    def RandomVerticalFlip(self, img, seg, roi, p):
        if random.random() < p:
            img = F.vflip(img)
            seg = F.vflip(seg)
            roi = F.vflip(roi)
        return img, seg, roi

    def RandomRotation(self, img, seg, roi, degrees, p):
        if random.random() < p:
            angle = random.uniform(-degrees, degrees)
            img = F.rotate(img, angle)
            seg = F.rotate(seg, angle)
            roi = F.rotate(roi, angle)
        return img, seg, roi

    def RandomColorJitter(self, img, seg, roi, p):
        if random.random() < p:
            brightness = random.uniform(0.2, 0.8)
            contrast = random.uniform(0.2, 0.8)
            saturation = random.uniform(0.2, 0.8)
            img = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)(img)
            seg = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)(seg)
            roi = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)(roi)
        return img, seg, roi

    def RandomSpNoise(self, img, seg, roi, p=0.25, mean=0, var=0.005):
        if random.random() < p:
            img = np.asarray(img)
            seg = np.asarray(seg)
            roi = np.asarray(roi)
            img = np.array(img / 255, dtype=float)
            seg = np.array(seg / 255, dtype=float)
            roi = np.array(roi / 255, dtype=float)
            noise = np.random.normal(mean, var ** 0.5, img.shape)
            img = img + noise
            seg = seg + noise
            roi = roi + noise
            if img.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            img = np.clip(img, low_clip, 1.0)
            img = np.uint8(img * 255)
            img = Image.fromarray(np.uint8(img))
            seg = np.clip(seg, low_clip, 1.0)
            seg = np.uint8(seg * 255)
            seg = Image.fromarray(np.uint8(seg))
            roi = np.clip(roi, low_clip, 1.0)
            roi = np.uint8(roi * 255)
            roi = Image.fromarray(np.uint8(roi))
        return img, seg, roi

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        seg_path = self.segs_path[idx]
        roi_path = self.roi_path[idx]
        img_id = self.class_id[idx]
        image = Image.open(img_path).convert("L")
        seg = Image.open(seg_path)
        roi = Image.open(roi_path)

        if self.train is "train":
            image, seg, roi = self.RandomSpNoise(image, seg, roi)
            image, seg, roi = self.RandomRotation(image, seg, roi, self.rotate, 0.6)
            image, seg, roi = self.RandomColorJitter(image, seg, roi, 0.25)
            image, seg, roi = self.RandomHorizonalFlip(image, seg, roi, 0.5)
            image, seg, roi = self.RandomVerticalFlip(image, seg, roi, 0.5)
            image, seg, roi = self.RandomCrop(image, seg, roi, crop_size=self.resize)
            image = self.transform_train(image)
            seg = self.transform_train(seg)
            roi = self.transform_roi(roi)
        else:
            image = self.transform_test(image)
            seg = self.transform_test(seg)
            roi = self.transform_roi(roi)
        images = {"img": image,
                  "seg": seg,
                  "roi": roi,
                  }
        img_class = {"img_id"  : img_id,
                     "img_name": self.images_path[idx]}
        return images, img_class
