import os

from skimage import io
from dataset_loader import parse_regions


def create_gt_map(ann_dir, im_dir, gt_res_map_dir):
    resize_s = 256
    anns = []
    for ann in os.listdir(ann_dir):
        anns.append(ann)

    points_for_folder = []
    for i in range(len(anns)):
        # get image shape to resize landmarks after
        img_name = os.path.join(im_dir, anns[i][:-5])
        image = io.imread(img_name)
        h, w = image.shape[:2]

        landmarks = parse_regions(os.path.join(ann_dir, anns[i]))

        points_for_image = []
        for d in landmarks:
            points_for_object = []
            for list_of_points in d['regions']:
                cur_l = []
                for i, point in enumerate(list_of_points):
                    if i == 0 or i == 2:
                        cur_l.append(point['x'] * resize_s / w)
                        cur_l.append(point['y'] * resize_s / h)
                points_for_object.append(cur_l)
            if points_for_object not in points_for_image:
                points_for_image.append(points_for_object)
        points_for_folder.append(points_for_image)

        # save pred data for mAP
        filename, file_ext = os.path.splitext(os.path.basename(img_name))
        with open(gt_res_map_dir + '/' + filename + '.txt', 'w') as f:
            for phrase in points_for_image:
                for word in phrase:
                    s = 'text ' + str(word[0]) + ' ' + str(word[1]) + ' ' + str(word[2]) + ' ' + str(word[3]) + '\n'
                    f.write(s)


if __name__ == '__main__':
    # create gt map
    create_gt_map('C:/Users/denis/Desktop/probation/test/ann/',
                  'C:/Users/denis/Desktop/probation/test/images/',
                  'input/ground-truth')
