import numpy as np
import os
import cv2
import dlib
from tqdm import tqdm
from A2.lab2_lamdmarks import rect_to_bb


def prepare_cartoon_data2(images_dir, labels_path, img_name_columns, labels_columns, img_size = 50, train=True):
    labels_file = open(labels_path, 'r')
    lines = labels_file.readlines()
    image_label = {line.split('\t')[img_name_columns].rstrip() : int(line.split('\t')[labels_columns]) for line in lines[1:]}
    # get all image paths
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    if train:
        without_glasses_image = []
        print('start find image without glasses in {}....'.format(images_dir))
        for i in tqdm(range(len(image_paths))):
            image_path = image_paths[i]
            img = cv2.imread(image_path)
            x, y, w, h = 240, 180, 50, 50
            img = img[x:x + w, y:y + h]
            x, y, w, h = 16, 26, 20, 20

            hsv = cv2.cvtColor(img[x:x+w, y:y+h], cv2.COLOR_RGB2HSV)
            mean = hsv[:, :, 2].mean()
            if mean > 50:
                without_glasses_image.append(image_path)
        print('total image without glasses: {} total:{} rate:{}'.format(len(without_glasses_image), len(image_paths), len(without_glasses_image)/len(image_paths)))
        image_paths = without_glasses_image
        
    all_imgs = []
    all_labels = []
    print('start extracting feature of images in ', images_dir)
    i = 0
    for image_path in tqdm(image_paths):
        i += 1
        # if i> 100:
        #     break
        img = cv2.imread(image_path)
        img = cv2.resize(src=img, dsize=(img_size, img_size),  interpolation=cv2.INTER_LANCZOS4) #reduce image size
        img_name = image_path.split('/')[-1]
        

        all_imgs.append(img)
        all_labels.append(image_label[img_name])
    total_len = len(all_imgs)
    all_imgs = np.array(all_imgs).reshape((total_len, -1))
    all_labels = np.array(all_labels)
    return all_imgs, all_labels