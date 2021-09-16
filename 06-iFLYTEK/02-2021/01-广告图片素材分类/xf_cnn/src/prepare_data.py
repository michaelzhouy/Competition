# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 11:58 上午
# @Author  : Michael Zhouy
import pandas as pd
import glob
import os
import codecs
import shutil
from glob import glob
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


label_id_name_dict = [str(i) for i in range(137)]


def split_train_valid(input_dir, output_train_dir, output_valid_dir):
    if not os.path.exists(input_dir):
        print(input_dir, 'is not exist')
        return

    # 1. 检查图片和标签的一一对应
    # label_file_paths = glob(os.path.join(input_dir, '*.txt'))
    # valid_img_names = []
    # valid_labels = []
    # for file_path in label_file_paths:
    #     with codecs.open(file_path, 'r', 'utf-8') as f:
    #         line = f.readline()
    #     line_split = line.strip().split(', ')
    #     img_name = line_split[0]
    #     label_id = line_split[1]
    #     if os.path.exists(os.path.join(input_dir, img_name)):
    #         valid_img_names.append(img_name)
    #         valid_labels.append(int(label_id))
    #     else:
    #         print('error', img_name, 'is not exist')
    label_file_paths = glob('../data/训练集/*/*')
    train_df = pd.DataFrame({'path': label_file_paths})
    train_df['label'] = train_df['path'].apply(lambda x: int(x.split('/')[-2]))

    # 2. 使用 StratifiedShuffleSplit 划分训练集和验证集，可保证划分后各类别的占比保持一致
    # TODO，数据集划分方式可根据您的需要自行调整
    sss = StratifiedShuffleSplit(n_splits=1, test_size=500, random_state=0)
    sps = sss.split(train_df['path'], train_df['label'])
    for sp in sps:
        train_index, val_index = sp

    # 3. 创建 output_train_dir 目录下的所有标签名子目录
    for id in label_id_name_dict:
        if not os.path.exists(os.path.join(output_train_dir, id)):
            os.mkdir(os.path.join(output_train_dir, id))

    # 4. 将训练集图片拷贝到 output_train_dir 目录
    for index in train_index:
        file_path = label_file_paths[index]
        with codecs.open(file_path, 'r', 'utf-8') as f:
            gt_label = f.readline()
        img_name = gt_label.split(',')[0].strip()
        id = gt_label.split(',')[1].strip()
        shutil.copy(os.path.join(input_dir, img_name), os.path.join(output_train_dir, id, img_name))

    # 5. 创建 output_val_dir 目录下的所有标签名子目录
    for id in label_id_name_dict:
        if not os.path.exists(os.path.join(output_valid_dir, id)):
            os.mkdir(os.path.join(output_valid_dir, id))

    # 6. 将验证集图片拷贝到 output_val_dir 目录
    for index in val_index:
        file_path = label_file_paths[index]
        with codecs.open(file_path, 'r', 'utf-8') as f:
            gt_label = f.readline()
        img_name = gt_label.split(',')[0].strip()
        id = gt_label.split(',')[1].strip()
        shutil.copy(os.path.join(input_dir, img_name), os.path.join(output_valid_dir, id, img_name))

    print('total samples: %d, train samples: %d, val samples:%d'
          % (len(train_df), len(train_index), len(val_index)))
    print('end')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='data prepare')
    parser.add_argument('--input_dir', required=True, type=str, help='input data dir')
    parser.add_argument('--output_train_dir', required=True, type=str, help='output train data dir')
    parser.add_argument('--output_valid_dir', required=True, type=str, help='output validation data dir')
    args = parser.parse_args()
    if args.input_dir == '' or args.output_train_dir == '' or args.output_valid_dir == '':
        raise Exception('You must specify valid arguments')
    if not os.path.exists(args.output_train_dir):
        os.makedirs(args.output_train_dir)
    if not os.path.exists(args.output_valid_dir):
        os.makedirs(args.output_valid_dir)
    split_train_valid(args.input_dir, args.output_train_dir, args.output_valid_dir)
