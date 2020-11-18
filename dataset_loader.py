import os
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

from torchvision import transforms
from skimage import io
from torch.utils.data import Dataset
from transformers import Rescale, ToTensor


class CheckDataset(Dataset):
    """Check dataset."""

    def __init__(self, ann_dir, im_dir, transform=None):
        """
        :param ann_dir: ann directory
        :param im_dir: image directory
        :param transform: transformation
        """
        self.anns = []
        self.ann_dir = ann_dir
        self.im_dir = im_dir
        self.transform = transform

        for ann in os.listdir(self.ann_dir):
            self.anns.append(ann)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.im_dir, self.anns[idx][:-5])
        image = io.imread(img_name)
        heatmap = create_word_heatmap(parse_regions(os.path.join(self.ann_dir, self.anns[idx])), image.shape)
        sample = {'image': image, 'heatmap': heatmap}

        if self.transform:
            sample = self.transform(sample)

        return sample


def parse_regions(dir_to_file):
    indexes = []
    with open(dir_to_file) as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if key != 'text':
                if isinstance(value, list):
                    for element in value:
                        if isinstance(element, dict):
                            cur_d = {}
                            if 'text' in element.keys() and 'regions' in element.keys():
                                cur_d['text'] = element['text']
                                cur_d['regions'] = element['regions']
                            indexes.append(cur_d)

                elif isinstance(value, dict):
                    if 'text' in value.keys() and 'regions' in value.keys():
                        indexes.append({'text': value['text'],
                                        'regions': value['regions']})
        return indexes


def create_word_heatmap(landmarks, image_shape):
    final_landmark_image = np.zeros(image_shape[0:2])
    for d in landmarks:
        for four_p in d['regions']:
            kpts_x = []
            kpts_y = []
            for p in four_p:
                kpts_x.append(p['x'])
                kpts_y.append(p['y'])

            # get extremes
            x_min, x_max = min(kpts_x), max(kpts_x)
            y_min, y_max = min(kpts_y), max(kpts_y)

            # deal with border cases
            if x_min == x_max:
                x_max = x_min + 2
            if y_min == y_max:
                y_max = y_min + 2
            if y_max > final_landmark_image.shape[0] - 2:
                y_max = final_landmark_image.shape[0] - 1
            if y_min > final_landmark_image.shape[0] - 2:
                y_min = final_landmark_image.shape[0] - 3
            if x_max > final_landmark_image.shape[1] - 2:
                x_max = final_landmark_image.shape[1] - 1
            if x_min > final_landmark_image.shape[1] - 2:
                x_min = final_landmark_image.shape[1] - 3

            # apply perspective transition
            cur_res = perspective_trans(x_max-x_min, y_max-y_min)

            # add result to the heatmap
            final_landmark_image[y_min: y_max, x_min: x_max] += cur_res

    return final_landmark_image


def create_2d_gaussian():
    x, y = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 0.5, 0.0
    return np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) / (2.0 * np.pi * sigma**2)


def perspective_trans(dx, dy):
    # get standart gaussian distrib
    gaus = create_2d_gaussian()
    g_h, g_w = gaus.shape[0], gaus.shape[1]

    # create coordinates for perspective transformation
    start_point = np.float32([[0,0], [g_h, 0], [0, g_w], [g_h, g_w]])
    end_point = np.float32([[0,0], [dx, 0], [0, dy], [dx, dy]])

    # get perspective matrix
    matrix = cv2.getPerspectiveTransform(start_point, end_point)

    return cv2.warpPerspective(gaus, matrix, (dx, dy))


if __name__ == '__main__':

    d = CheckDataset('C:/Users/denis/Desktop/probation/train/ann/',
                     'C:/Users/denis/Desktop/probation/train/images/',
                     transforms.Compose([Rescale((256, 256)), ToTensor()]))

    for i in range(len(d)):
        sample = d[i]
        print(i, sample['image'].shape, sample['heatmap'].shape)


