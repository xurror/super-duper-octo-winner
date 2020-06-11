import os
import random
import pandas as pd

import os

def split(datadir='src/data/open/'):
    set_size = len(os.listdir(datadir))

    test_ratio = int(set_size * 0.25)

    print(set_size)
    print(test_ratio)

    unique_sequence = random.sample(list(range(set_size)), len(list(range(set_size))))

    test_set = unique_sequence[:test_ratio]

    count = 0
    for i in test_set:
        os.rename(datadir+'open.'+str(i)+".jpg", 'src/data/test/open/open' + '.' + str(count) + ".jpg")
        count += 1
        

def spilt_features(df=None, file_index=None, features_dir="vgg_face7/", data_dir="data/"):
    train_drunk_count = 0
    train_sober_count = 0

    test_drunk_count = 0
    test_sober_count = 0

    val_drunk_count = 0
    val_sober_count = 0

    count = 0

    for i in range(df.shape[0]):
        set_split = df["train"][i]
        category = df["Drunk"][i].lower()
        npy_file = (df[file_index][i])[:-3] + "npy"

        if set_split == "train":
            if category == "drunk":
                try:
                    os.rename(features_dir + npy_file, data_dir + "train/" + "drunk_" + str(train_drunk_count) + ".npy")
                    train_drunk_count += 1
                except FileNotFoundError:
                    count += 1
            elif category == "sober":
                try:
                    os.rename(features_dir + npy_file, data_dir + "train/" + "sober_" + str(train_sober_count) + ".npy")
                    train_sober_count += 1
                except FileNotFoundError:
                    count += 1
            else:
                raise Exception(f"Unknown Category {category}")

        elif set_split == "test":
            if category == "drunk":
                try:
                    os.rename(features_dir + npy_file, data_dir + "test/" + "drunk_" + str(test_drunk_count) + ".npy")
                    test_drunk_count += 1
                except FileNotFoundError:
                    count += 1
            elif category == "sober":
                try:
                    os.rename(features_dir + npy_file, data_dir + "test/" + "sober_" + str(test_sober_count) + ".npy")
                    test_sober_count += 1
                except FileNotFoundError:
                    count += 1
            else:
                raise Exception(f"Unknown Category {category}")
            
        elif set_split == "val":
            if category == "drunk":
                try:
                    os.rename(features_dir + npy_file, data_dir + "val/" + "drunk_" + str(val_drunk_count) + ".npy")
                    val_drunk_count += 1
                except FileNotFoundError:
                    count += 1
            elif category == "sober":
                try:
                    os.rename(features_dir + npy_file, data_dir + "val/" + "sober_" + str(val_sober_count) + ".npy")
                    val_sober_count += 1
                except FileNotFoundError:
                    count += 1
            else:
                raise Exception(f"Unknown Category {category}")

        else:
            raise Exception(f"Unknown index {set_split}")
    print(count)

def split_data():
    df = pd.read_csv("src/data/train_test_sets/1/split_3585_642_1016.csv")
    cols = [x for x in df.columns]
    file_index = cols[2]
    spilt_features(df, file_index, "src/data/vgg_face7/", "src/data/drunk/")

if __name__ == "__main__":    
    split_data()
    path, dirs, train_files = next(os.walk("src/data/drunk/train"))
    path, dirs, test_files = next(os.walk("src/data/drunk/test"))
    path, dirs, val_files = next(os.walk("src/data/drunk/val"))

    print(len(train_files))
    print(len(test_files))
    print(len(val_files))