import data_prep
import argparse
import os
import time
import glob

# Convert testing result to kaggle submition format


def convert_result(path):
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'datasets/testing_data')

    with open(os.path.join(path, "store.txt"), "r") as file:
        file_lsit = os.listdir(DATA_PATH)
        result = open(os.path.join(BASE_PATH, "result.csv"), "w")
        analyze_array, id_to_label, label_to_cat, cat_to_label = data_prep.analyze()
        result.write("id,label\n")
        for i in file:
            result.write(str(int(file_lsit[int(i.split(",")[1])].split(
                ".")[0])) + "," + cat_to_label[int(i.split(",")[0])] + "\n")

# K-folder


def voting(k, path):
    print("Use k-folder")
    BASE_PATH = path
    file = open(os.path.join(BASE_PATH, "store.txt"), "w")

    tmp = [{} for i in range(5000)]
    for filepath in glob.glob(os.path.join(BASE_PATH, "*.txt")):
        file0 = open((filepath), "r")
        for i in range(5000):
            line0 = file0.readline().split(",")

            if line0[0] not in tmp[i].keys():
                tmp[i][line0[0]] = 0
            else:
                tmp[i][line0[0]] += 1

    for i in range(5000):
        ans = max(tmp[i], key=tmp[i].get)
        file.write(ans + "," + str(i) + "\n")

    file.close()
    convert_result(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_folder', '-k', default=False)
    parser.add_argument('--path', '-p', default=None)
    args = parser.parse_args()

    if(args.k_folder):
        voting(args.k_folder, args.path)
    else:
        convert_result(args.path)
