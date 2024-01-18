import os
import os.path

def print_dataset_stats(dataset_path : str):
    train_folder = os.path.join(dataset_path, "train")
    test_folder = os.path.join(dataset_path, "test")

    train_dict = {}
    train_total = 0
    for dirpath, dirs, files in os.walk(train_folder):
        for f in files:
            train_dict[dirpath] = train_dict.get(dirpath, 0)
            train_dict[dirpath] += 1
            train_total += 1

    test_dict = {}
    test_total = 0
    for dirpath, dirs, files in os.walk(test_folder):
        for f in files:
            test_dict[dirpath] = test_dict.get(dirpath, 0)
            test_dict[dirpath] += 1
            test_total += 1

    train_dict = [(k, v) for k, v in train_dict.items()]
    train_dict.sort(key=lambda x: -x[1])

    test_dict = [(k, v) for k, v in test_dict.items()]
    test_dict.sort(key=lambda x: -x[1])

    print("TRAIN")
    print("Total", train_total, "100%")
    for i, el in enumerate(train_dict):
        if i == int(len(train_dict) / 2):
            print("Median", end="")
        print(el[0], el[1], str(int(el[1] / train_total * 100)) + "%")

    print("TEST")
    print("Total", test_total, "100%")
    for i, el in enumerate(test_dict):
        if i == int(len(test_dict) / 2):
            print("Median", end="")
        print(el[0], el[1], str(int(el[1] / test_total * 100)) + "%")

    test_dict.sort(key=lambda x: x[0].split("\\")[-1])
    train_dict.sort(key=lambda x: x[0].split("\\")[-1])

    for i in range(len(train_dict)):
        print(train_dict[i][0], test_dict[i][0])


if __name__ == "__main__":
    print_dataset_stats("dataset_unbalansed")