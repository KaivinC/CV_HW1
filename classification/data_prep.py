import csv
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import argparse

def analyze():  # analyze the data distribution for prepare_val
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'datasets')
    labels = pd.read_csv(DATA_PATH + '/' + 'training_labels.csv')
    label = labels['label'].tolist()
    ids = labels['id'].tolist()
    cat_to_label = {}
    label_to_cat = {}
    label_name = []
    analyze_array = [[] for i in range(196)]
    id_to_label = {}
    for i in label:
        if i not in label_name:
            label_name.append(i)
    for idx, name in enumerate(label_name):
        cat_to_label[idx] = name
    for idx, name in enumerate(label_name):
        label_to_cat[name] = idx
    for idx, id_ in enumerate(ids):
        id_to_label[id_] = label[idx]
    for idx, name in enumerate(label):
        index = label_to_cat[name]
        analyze_array[index].append(ids[idx])

    return analyze_array, id_to_label, label_to_cat, cat_to_label


def prepare_val(training_data_ratio):  # prepare validation
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'datasets/anno')
    if not os.path.exists(os.path.join(DATA_PATH)):
        os.mkdir(os.path.join(DATA_PATH))

    train_file = open(os.path.join(DATA_PATH, "train.txt"), 'w')
    val_file = open(os.path.join(DATA_PATH, "test.txt"), 'w')
    val = []
    train = []

    analyze_array, id_to_label, label_to_cat, cat_to_label = analyze()

    for i in analyze_array:
        for j in range(len(i)):
            # randomly split train/val
            choice = random.sample(range(0, len(i)), math.ceil(
                len(i) * (1-training_data_ratio)))
            if j in choice:
                val.append(["%06d" % i[j] + ".jpg", ",",
                            str(label_to_cat[id_to_label[i[j]]] + 1), "\n"])
            else:
                train.append(["%06d" % i[j] + ".jpg", ",",
                              str(label_to_cat[id_to_label[i[j]]] + 1), "\n"])
    random.shuffle(val)
    random.shuffle(train)

    for i in range(len(val)):
        val_file.writelines(val[i])
    for i in range(len(train)):
        train_file.writelines(train[i])

# make test file name data
def make_test_csv():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'datasets/testing_data')
    file_list = os.listdir(DATA_PATH)
    file = open(os.path.join(BASE_PATH, 'datasets/anno/anno.txt'), 'w')
    for i in file_list:
        file.write(i + "\n")
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_ratio', '-t', default=0.8)
    args = parser.parse_args()

    prepare_val(args.training_data_ratio)
    make_test_csv()