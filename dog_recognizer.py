import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    #input is a 3-component image with size
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class DogRecognizer:
    _net = None
    _classes = None
    _transform = None
    def __init__(self, classes_path, model_path = None):
        # load classes
        self._classes = open(classes_path).read().strip().split('\n')
        # create net
        self._net = AlexNet(len(self._classes))
        if torch.cuda.is_available():
            self._net.cuda()
        if not model_path is None:
            self._net.load_state_dict(torch.load(model_path))
            self._net.eval()
        self._transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def generate_classes_file(cls, output_path : str, train_dataset_path:str):
        test_dataset = datasets.ImageFolder(train_dataset_path)
        classes = test_dataset.classes
        with open(output_path, "wt") as f:
            for el in classes:
                f.write(el + "\n")
        return

    # this function will get any image(PIL style) and return tuple
    # first element is name of deducted breed
    # second element is a dict with key=breed, value=prediction
    def predict_from_image(self, image):
        # 1) transform image to valid input
        local_image = self._transform(image)
        self._net.eval()
        # 2) move input to cuda
        if torch.cuda.is_available():
            local_image = local_image.cuda()
        local_image = local_image.unsqueeze(0)
        # 3) run local image through our net
        with torch.no_grad():
            prediction = self._net(local_image)

        # 4) combine in dict with key=breed, value = prediction
        prediction = prediction[0].detach().cpu().numpy()
        result = {}
        top_breed = ""
        for i, key in enumerate(self._classes):
            result[key] = float(prediction[i])
            if top_breed == "" or result[key] > result[top_breed]:
                top_breed = key

        return (top_breed, result)

    def train(self,
              train_dataset_dir : str,
              test_dataset_dir : str,
              batch_size : int = 128,
              learning_rate : float = 0.0001):
        writer = SummaryWriter()
        # 1) load datasets as torchvision.datasets.ImageFolder
        print("loading train dataset...")
        train_dataset = datasets.ImageFolder(train_dataset_dir, self._transform)
        print("loading test dataset...")
        test_dataset = datasets.ImageFolder(test_dataset_dir, self._transform)

        # 2) dataloader from dataset with some batch size
        print("creating train dataloader...")
        train_dataloader = data.DataLoader(
            train_dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
            batch_size=batch_size)
        print("creating test dataloader...")
        test_dataloader = data.DataLoader(
            test_dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
            batch_size=batch_size)

        # 4) Optimizer Adam and lr scheduler
        print("creating optimizer...")
        optimizer = optim.Adam(params=self._net.parameters(), lr=learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # 5) train epochs and save each new best
        epoch = 0
        steps = 0
        while True:
            print("epoch", epoch)
            lr_scheduler.step()
            # train iterations
            self._net.train()
            sum_loss = 0
            iter = 0
            for imgs, classes in train_dataloader:
                if torch.cuda.is_available():
                    imgs, classes = imgs.cuda(), classes.cuda()

                # calculate the loss
                output = self._net(imgs)
                loss = F.cross_entropy(output, classes)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1
                iter += 1
                # add info about 10 iteration
                if steps % 100 == 0:
                    with torch.no_grad():
                        _, preds = torch.max(output, 1)
                        accuracy = torch.sum(preds == classes)
                        writer.add_scalar("Acurracy/train", accuracy.item() / batch_size, steps)
                    writer.add_scalar("Loss/train", loss.item(), steps)
                sum_loss += loss.item()

            # add info about epoch average loss
            writer.add_scalar("EposhAvgLoss/train", sum_loss / max(iter, 1), epoch)

            # test iterations
            self._net.eval()
            sum_loss = 0
            iter = 0
            for imgs, classes in test_dataloader:
                if torch.cuda.is_available():
                    imgs, classes = imgs.cuda(), classes.cuda()

                # calculate the loss
                with torch.no_grad():
                    output = self._net(imgs)
                    loss = F.cross_entropy(output, classes)

                steps += 1
                iter += 1
                # add info about 10 iteration
                if steps % 100 == 0:
                    with torch.no_grad():
                        _, preds = torch.max(output, 1)
                        accuracy = torch.sum(preds == classes)
                        writer.add_scalar("Acurracy/test", accuracy.item() / batch_size, steps)
                    writer.add_scalar("Loss/test", loss.item(), steps)
                sum_loss += loss.item()

            # add info about epoch average loss
            writer.add_scalar("EposhAvgLoss/test", sum_loss / max(iter, 1), epoch)

            if epoch % 10 == 0:
                torch.save(self._net.state_dict(), "models/save_only_2_epoch_" + str(epoch) + ".m")

            epoch += 1

    def confusion_matrix(self,
                         test_dataset_dir : str,
                         batch_size : int = 128):
        print("loading test dataset...")
        test_dataset = datasets.ImageFolder(test_dataset_dir, self._transform)
        print("creating test dataloader...")
        test_dataloader = data.DataLoader(
            test_dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
            batch_size=batch_size)

        self._net.eval()
        y_test = []
        predictions = []
        for imgs, classes in test_dataloader:
            if torch.cuda.is_available():
                imgs, classes = imgs.cuda(), classes.cuda()

            # calculate the loss
            with torch.no_grad():
                output = self._net(imgs)
                classes = classes.detach().cpu().numpy()
                output = output.detach().cpu().numpy()
                for i in range(batch_size):
                    y_test.append(classes[i])
                    pred = np.argmax(output[i])
                    predictions.append(pred)

        cm = confusion_matrix(y_test, predictions)
        cm = cm / cm.astype(np.float32).sum(axis=0)
        return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._classes).plot()



if __name__ == "__main__":
    DogRecognizer.generate_classes_file("classes.txt", "dataset_unbalansed/test")
    recognizer = DogRecognizer("classes.txt")
    recognizer.train("dataset_unbalansed/train", "dataset_unbalansed/test")