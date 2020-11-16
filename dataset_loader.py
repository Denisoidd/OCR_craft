import os
import json
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


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
        print(img_name)
        image = cv2.imread(img_name)
        landmarks = parse_regions(os.path.join(self.ann_dir, self.anns[idx]))
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


def parse_regions(dir_to_file):
    indexes = []
    with open(dir_to_file) as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if key != 'text':
                # print(key)
                if isinstance(value, list):
                    for element in value:
                        if isinstance(element, dict):
                            cur_d = {}
                            if 'index' in element.keys() and 'regions' in element.keys():
                                cur_d['index'] = element['index']
                                cur_d['regions'] = element['regions']
                            if 'data' in element.keys():
                                cur_d['data'] = element['data']
                            indexes.append(cur_d)

                elif isinstance(value, dict):
                    if 'index' in value.keys() and 'regions' in value.keys():
                        indexes.append({'index': value['index'],
                                          'regions': value['regions'],
                                          'data': value['data']})
        return indexes
        # print(sorted(indexes_1, key=lambda k: k['index']))
        # for el in indexes_1:
        #     print(el['data'], end=' ')
        # print()
        # for el in indexes_1:
        #     print(el['regions'], end=' ')
        # print()
        # print()
        # for el in indexes_2:
        #     print(el['data'], end=' ')
        # print()
        # for el in indexes_2:
        #     print(el['regions'], end=' ')

        # print(sorted(indexes_2, key=lambda k: k['index']))


if __name__ == '__main__':
    d = CheckDataset('C:/Users/denis/Desktop/probation/train/ann/', 'C:/Users/denis/Desktop/probation/train/images/')
    # parse_regions('C:/Users/denis/Desktop/probation/train/ann/X00016469620.jpg.json')
    for i in range(len(d)):
        sample = d[i]

        print(i, sample['image'].shape, sample['landmarks'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        plt.imshow(sample['image'])
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        if i == 3:
            plt.show()
            break