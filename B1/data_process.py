import numpy as np
import os
import cv2
from tqdm import tqdm

def get_features(img):
    
    return None
def prepare_cartoon_data(images_dir, labels_path, img_name_colunms, labels_colunms, img_size = 50):
    labels_file = open(labels_path, 'r')
    lines = labels_file.readlines()
    image_label = {line.split('\t')[img_name_colunms].rstrip() : int(line.split('\t')[labels_colunms]) for line in lines[1:]}
    # get all image paths
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    all_imgs = []
    all_labels = []
    print('start extracting feature of images in ', images_dir)
    i = 0
    for image_path in tqdm(image_paths):
        i += 1
        # if i> 100:
        #     break
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src=img, dsize=(img_size, img_size),  interpolation=cv2.INTER_LANCZOS4)
        img_name = image_path.split('/')[-1]
        # features = get_features(img)
        # if features is not None:
        all_imgs.append(img)
        all_labels.append(image_label[img_name])
    total_len = len(all_imgs)
    all_imgs = np.array(all_imgs).reshape((total_len, -1))
    all_labels = np.array(all_labels)
    return all_imgs, all_labels