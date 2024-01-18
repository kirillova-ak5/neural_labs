import cv2
from dog_boxer import DogBoxer
from dog_recognizer import DogRecognizer

class DogFinder:
    _box_finder = None
    _recognizer = None
    _window_name = "DogFinder"
    def __init__(self, model_path : str, classes_path : str = "classes.txt"):
        self._box_finder = DogBoxer()
        self._recognizer = DogRecognizer(classes_path=classes_path, model_path=model_path)

    def run(self, image_path : str):
        # load image
        orig_image = cv2.imread(image_path)
        # find all bound boxes
        bound_boxes = self._box_finder.find_all_dogs_bound_box_on_image(orig_image)
        # crop subimages
        sub_images = []
        for bbox in bound_boxes:
            sub_images.append(orig_image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])

        # transform each subimage to PIL and pass to recognizer
        predictions = []
        for i in range(len(sub_images)):
            sub_images[i] = cv2.cvtColor(sub_images[i], cv2.COLOR_BGR2RGB)
            sub_images[i] = cv2.resize(sub_images[i], (227, 227))
            breed, probs = self._recognizer.predict_from_image(sub_images[i])
            predictions.append([breed, probs[breed]])

        # add all texts to orig image
        for i in range(len(bound_boxes)):
            box = bound_boxes[i]
            breed = predictions[i][0]
            probability = predictions[i][1]
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            cv2.rectangle(orig_image, (x, y), (x + w, y + h), [0, 255, 0], 2)
            text = "{}: {:.4f}".format(breed, probability)
            cv2.putText(orig_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)
            cv2.putText(orig_image, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)

        cv2.namedWindow(self._window_name)

        cv2.imshow(self._window_name, orig_image)
        cv2.waitKey(0)

if __name__ == "__main__":
    finder = DogFinder("models/save_only_2_epoch_310.m")
    finder.run("images/test6.jpg")

