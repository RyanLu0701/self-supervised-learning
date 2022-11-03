import tqdm
import cv2
import numpy as np
import os

def readfile_train(PATH):

    jpg_list = os.listdir(path=PATH)

    num = len(jpg_list)

    x = np.zeros((num , 128, 128, 3), dtype=np.uint8)

    count = 0

    for  file in  tqdm.tqdm(jpg_list):

        img = cv2.imread(os.path.join(PATH, file))

        x[count, :, :] =cv2.resize(img, (128, 128))

        count += 1

    return x


def readfile_test(PATH):

    dir_list = os.listdir(path=PATH)

    x = np.zeros((500 , 128, 128, 3), dtype=np.uint8)
    y = np.zeros(500  , dtype= np.uint8)
    count = 0

    for i in dir_list:

        jpg_list = os.listdir(path = f"{PATH}/{i}" )


        for file in  tqdm.tqdm(jpg_list):

            img = cv2.imread(os.path.join(f"{PATH}/{i}", file))

            x[count, :, :] =cv2.resize(img, (128, 128))

            y[count] = i

            count += 1

    return x,y

