# --coding:utf-8--
import os
import random
import shutil
import csv

# 讀取csv，將ID與label轉換成dict
def label_dict():
    with open(csv_path, mode='r') as inp:
        reader = csv.reader(inp)
        label_dict = {rows[0]: rows[1] for rows in reader}
        del label_dict['ID']
    print('※ 成功讀取ID與label之csv檔')
    return  label_dict

# 圖片依照label(0~5)移動到對應的資料夾
def detect_classfier(label_dict):
    for i in os.listdir(src_dir_name):
        if i.endswith('.png') and i in label_dict:
            try:
                os.mkdir(src_dir_name + label_dict[i])
            except FileExistsError:
                pass
            shutil.move(src_dir_name + i, src_dir_name + label_dict[i] + '/' + i)
        else:
            pass
    print('※ train圖片移動到對應資料夾')

# 移動測試集
def move_test_data(label, test_data:list):
    for j in test_data:
        try:
            os.mkdir(target_dir_name + label)
        except FileExistsError:
            pass
        shutil.move(src_dir_name + label + '/' + j, target_dir_name + label + '/' + j)

# 分成train與vali
def test_train_split():
    try:
        os.mkdir(target_dir_name)
    except FileExistsError:
        pass
    for i in os.listdir(src_dir_name):
        # 每個類別的照片數
        dir_length = len(os.listdir(src_dir_name + i))
        # 比例抽樣
        test_num = round(test_size * dir_length)
        test_data = random.sample(os.listdir(src_dir_name + i), k = test_num)
        move_test_data(i, test_data)
        print('   類別{}，移動完成'.format(i))
    print('※ train與vali分配完成')

if __name__ == '__main__':
    ###分成訓練集跟資料集###
    src_dir_name = './AOI/split_train_and_vali_balance/train/'
    target_dir_name = './AOI/split_train_and_vali_balance/vali/'
    csv_path = './AOI/train.csv'
    test_size = 0.2

    # 讀取csv，將label轉成dict
    label_dict = label_dict()
    # 圖片依照label(0~5)移動到對應的資料夾
    detect_classfier(label_dict)
    # 分成train跟vali
    test_train_split()