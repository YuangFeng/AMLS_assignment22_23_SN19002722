import numpy as np
import os
import cv2
import dlib
from tqdm import tqdm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('B2/shape_predictor_68_face_landmarks.dat')
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    # resized_image = image.astype('uint8')

    gray = image.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [136])

    return dlibout

def prepare_cartoon_data2(images_dir, labels_path, img_name_colunms, labels_colunms, img_size = 50):
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
        img = cv2.resize(src=img, dsize=(img_size, img_size),  interpolation=cv2.INTER_LANCZOS4)
        img_name = image_path.split('/')[-1]
        

        all_imgs.append(img)
        all_labels.append(image_label[img_name])
    total_len = len(all_imgs)
    all_imgs = np.array(all_imgs).reshape((total_len, -1))
    all_labels = np.array(all_labels)
    print(all_imgs.shape)
    return all_imgs, all_labels