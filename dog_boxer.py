import cv2 as cv
import numpy as np


class DogBoxer:
    _net = None
    _dog_class_index = None
    _output_layers = None

    def __init__(self,
                 network_cfg_path : str = 'yolo3/yolov3.cfg',
                 network_weight_path : str = 'yolo3/yolov3.weights',
                 classes_path : str = 'yolo3/coco.names'):
        self._net = cv.dnn.readNetFromDarknet(network_cfg_path, network_weight_path)
        self._net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self._net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        classes = open(classes_path).read().strip().split('\n')
        for i, name in enumerate(classes):
            if "dog" in name:
                self._dog_class_index = i
                break
        ln = self._net.getLayerNames()
        layers = self._net.getUnconnectedOutLayers()
        self._output_layers = [ln[i - 1] for i in layers]

    # This function get a path to file and will return an array of bound boxes
    # bound box is an array of [x, y, w, h]
    def find_all_dogs_bound_box(self, path : str, min_confidence : float = 0.90):
        # 1) load image from path
        img = cv.imread(path)
        if img is None:
            return []

        # 2) call function for image
        return self.find_all_dogs_bound_box_on_image(img, min_confidence)

    # This function get a path to file and will return an array of images
    def crop_all_dogs(self, path: str, min_confidence: float = 0.90):
        # 1) load image from path
        img = cv.imread(path)
        if img is None:
            return []
        return self.crop_all_dogs_from_image(img, min_confidence)

    # This function get an image(OpenCV style) and will return an array of images
    def crop_all_dogs_from_image(self, image, min_confidence: float = 0.90):
        bboxes = self.find_all_dogs_bound_box_on_image(image, min_confidence)

        result = []
        # 3) crop images
        for bbox in bboxes:
            result.append(image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])
        return result

    # This function get an image(OpenCV style) and will return an array of bound boxes
    # bound box is an array of [x, y, w, h]
    def find_all_dogs_bound_box_on_image(self, image, min_confidence : float = 0.90):
        H, W = image.shape[:2]

        # 1) transform image to blob
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # 2) process input with YOLOv3
        self._net.setInput(blob)
        outputs = self._net.forward(self._output_layers)
        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)

        # 3) filter out only dogs bound boxes
        boxes = []
        confidences = []
        for output in outputs:
            # retrieve scores
            scores = output[5:]
            bound_box = output[:4] * np.array([W, H, W, H])
            confidence = scores[self._dog_class_index]
            if confidence > min_confidence:
                x, y, w, h = bound_box
                p0 = max(int(x - w // 2), 0), max(int(y - h // 2), 0)
                boxes.append([*p0, min(int(w), W - p0[0]), min(int(h), H - p0[1])])
                confidences.append(float(confidence))

        # 4) Merge bound boxes to retrieve best
        indices = cv.dnn.NMSBoxes(boxes, confidences, min_confidence, min_confidence - 0.1)

        # 5) retrieve best ones
        result = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                result.append([x, y, w, h])
        return result


if __name__ == "__main__":
    bboxer = DogBoxer()

    dogs = bboxer.crop_all_dogs("../dog_breed_photos/dog_breed_photos/dog_breed_photos\\6ba0dc39-0a41-4d1f-b30e-af8f8e813a24.jpg")
    
    for i, dog in enumerate(dogs):
        cv.imwrite(img=dog, filename="images/daaaaaaaaaaawg" + str(i) + ".jpg")