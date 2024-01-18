import random

import cv2 as cv
from dog_boxer import DogBoxer
import os
import os.path
from tqdm import tqdm

def processBigDataset(csv_path : str, in_images_path : str, out_dataset_path : str, train_rate : float = 0.8):
    records = {}
    box_finder = DogBoxer()
    logger = open("process_log.txt", "wt")

    # read csv
    table = []
    with open(csv_path, "rt") as f:
        for line in f.readlines():
            table.append(line.split(","))
    # remove first line
    table = table[1:]

    # read our breed rates and print some stats
    breed_rate = {}
    total = 0
    for row in table:
        breed_rate[row[1]] = breed_rate.get(row[1], 0)
        breed_rate[row[1]] += 1
        total += 1

    breed_rate = [(k, v) for k, v in breed_rate.items()]
    breed_rate.sort(key = lambda x : -x[1])
    print("Total", total, "100%")
    for breed in breed_rate:
        print(breed[0], breed[1], str(int(breed[1] / total * 100)) + "%")

    # create directory for train and test dataset
    os.makedirs(out_dataset_path, exist_ok=True)

    train_dataset_path = os.path.join(out_dataset_path, "train")
    test_dataset_path = os.path.join(out_dataset_path, "test")
    os.makedirs(train_dataset_path, exist_ok=True)
    os.makedirs(test_dataset_path, exist_ok=True)

    # for each line in table of dawg
    for line in tqdm(table):
        # load image
        breed = line[1]
        path = os.path.join(in_images_path, line[2])
        if os.path.isfile(path):
            imgs = box_finder.crop_all_dogs(path)
            if len(imgs) == 1:
                rand_val = random.uniform(0, 1)

                # ensure that we have at least 1 image in test and train
                if not os.path.isdir(os.path.join(test_dataset_path, breed)):
                    rand_val = 1
                if not os.path.isdir(os.path.join(train_dataset_path, breed)):
                    rand_val = 0

                out_path = out_dataset_path
                if rand_val >= train_rate:
                    out_path = test_dataset_path
                else:
                    out_path = train_dataset_path
                out_path = os.path.join(out_path, breed)
                os.makedirs(out_path, exist_ok=True)
                cv.imwrite(img=imgs[0], filename=os.path.join(out_path, line[2]))
            else:
                logger.write("WARNING: file " + path + " has " + str(len(imgs)) + " dawgs\n")

if __name__ == "__main__":
    processBigDataset("../dog_breed_photos.csv", "../dog_breed_photos/dog_breed_photos/dog_breed_photos", "dataset2", 0.6)