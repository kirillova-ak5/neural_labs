from dog_boxer import DogBoxer
import os
import os.path
from tqdm import tqdm

def checkDataset(path : str, csv_path : str):
    records = {}
    box_finder = DogBoxer()
    print("started dataset", path)

    for dirpath, dirs, files in os.walk(path):
        print("checking dir", dirpath)
        for f in tqdm(files):
            f = os.path.join(dirpath, f)
            if os.path.isfile(f) and f.endswith("png") or f.endswith("jpg"):
                boxes = box_finder.find_all_dogs_bound_box(f)
                records[f] = len(boxes)

    print("write log")
    with open(csv_path, "wt") as f:
        for record in records:
            f.write(str(record) + "; " + str(records[record]) + "\n")

if __name__ == "__main__":
    checkDataset("../dataset1", "../dataset1/log.csv")
    checkDataset("../dataset2", "../dataset2/log.csv")
    checkDataset("../dataset3", "../dataset3/log.csv")
