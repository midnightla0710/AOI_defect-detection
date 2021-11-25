from tensorflow.keras.preprocessing import image
from keras.models import load_model
import pandas as pd
import numpy as np
import csv
import os

from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Activation

#定義激活函數hard_swish
def hard_swish(x):
    return x * (K.relu(x + 3., max_value = 6.) / 6.)

# 讀取csv，將ID轉換成list
def id_list():
    with open(csv_path, mode='r') as inp:
        reader = csv.reader(inp)
        ids = [rows[0] for rows in reader]
        del ids[0]
    return  ids

# 讀取圖片
def read_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(512, 512))
        img = np.expand_dims(img, axis=0)
        return img / 255
    except Exception as e:
        print(img_path, e)

# 預測分類
def predit_result():
    answers = []
    for i in ids:
        print(src_dir_name + i)
        img = read_image(src_dir_name + i)
        # 機率list
        pred = model.predict(img)[0]
        # 預測結果(機率最大值)
        index = np.argmax(pred)
        prediction = labels[index]
        answers.append(prediction)
        print('辨識結果為： {}'.format(prediction))
        print('='*50)
    return answers

# 將資料存入CSV檔
def save_csv():
    for i in ids:
        dataframe = pd.DataFrame({'ID': ids, 'Label': answers})
        dataframe.to_csv("./submit.csv", index=False, encoding='utf-8-sig')
    print('※ 成功儲存submit.csv')

if __name__ == '__main__':
    # 關閉GPU加速功能(建議安裝無GPU版本，縮短初始化時間)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 檔案路徑
    csv_path = './AOI/test.csv'
    src_dir_name = './AOI/test_images/'
    model_path = './model/MobileNetV3_v5/MobileV3_v5.h5'

    # Label、ID列表
    labels = [i for i in range(6)]
    ids = id_list()

    # 新增hard_swish激活函數
    get_custom_objects().update({'hard_swish': Activation(hard_swish)})

    # 載入模型
    model = load_model(model_path)

    # 預測分類
    answers = predit_result()

    # 儲存答案
    save_csv()