import matplotlib.pyplot as plt
from dog_recognizer import DogRecognizer

if __name__ == "__main__":
    recognizer = DogRecognizer(classes_path="classes.txt", model_path="models/save_only_2_epoch_310.m")
    disp = recognizer.confusion_matrix(test_dataset_dir="dataset/test2")
    disp.plot()
    plt.show()