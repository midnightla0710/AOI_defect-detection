import os
import cv2
import numpy as np
from PIL import Image

# 讀取中文路徑圖檔(圖片讀取為BGR)
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

def transfer_to_channels():
    for i in os.listdir(src_dir_name):
        subfolder = src_dir_name + i
        for j in os.listdir(subfolder):
            print(subfolder+'/'+ j)
            img = cv_imread(subfolder+'/'+ j)
            # 轉成3通道灰階圖(以3通道灰階圖訓練模型)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            img = np.asarray(img)
            cv2.imencode('.png', img)[1].tofile(subfolder+'/'+ j)

def transfer_test_to_channels():
    for i in os.listdir(src_dir_name):
        print(src_dir_name + i)
        img = cv_imread(src_dir_name + i)
        # 轉成3通道灰階圖(以3通道灰階圖訓練模型)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        img = np.asarray(img)
        cv2.imencode('.png', img)[1].tofile(src_dir_name + i)

if __name__ == '__main__':
    # 將train資料集轉換成3通道
    src_dir_name = './AOI/split_train_and_vali_balance/train/'
    transfer_to_channels()

    # 將vali資料集轉換為3通道
    src_dir_name = './AOI/split_train_and_vali_balance/vali/'
    transfer_to_channels()

    # 將test資料集轉換成3通道
    src_dir_name = './AOI/three_channels_test_images/'
    transfer_test_to_channels()
    print('轉換完成')