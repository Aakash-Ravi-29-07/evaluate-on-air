import torch
import torchvision
import statistics
from Model.DigitModel import *
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class ModelOps:
    def __init__(self, train_batch_size, test_batch_size, learning_rate, momentum, kernel_size, pool_size, num_epochs):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_epochs = num_epochs
        self.num_classes = 10

        if torch.cuda.is_available():
            print("available")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.createDataloaders()

    def createDataloaders(self):
        self.train_dataloader = DataLoader(MNIST("/files/", train=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        )), self.train_batch_size, True)
        self.test_dataloader = DataLoader(MNIST("/files/", False, torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        )), self.test_batch_size, True)

    def createNewModel(self):
        self.model = DigitModel(self.kernel_size, self.pool_size, self.num_classes)
        self.model.to(self.device)
        self.createOptimizer()

    def loadModel(self, modelPath):
        self.model = torch.load(modelPath)
        self.model.to(self.device)

    def saveModel(self, modelPath):
        torch.save(self.model, modelPath)

    def createOptimizer(self):
        self.optimizer = SGD(self.model.parameters(), self.learning_rate, self.momentum)

    def train(self):
        for _ in range(self.num_epochs):
            self.model.train()
            print(f"Train Epoch {_ + 1} -> Avg Loss:", end=" ")
            losses = []
            for batch_idx, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                res = self.model(X)
                loss = F.nll_loss(res, y)
                loss.backward()
                self.optimizer.step()
                # if(batch_idx + 1) % 16 == 0:
                #     print(f"Batch {batch_idx + 1} -> Loss : {loss.item()}")
                losses.append(loss.item())
            print(statistics.mean(losses))
            self.test()

    def test(self):
        self.model.eval()
        total_loss = 0
        correct_pred = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                res = self.model(X)
                loss = F.nll_loss(res, y, size_average=False)
                total_loss += loss.item()
                predictions = res.data.max(1, keepdim=True)[1]
                correct_pred += predictions.eq(y.data.view_as(predictions)).sum()
            loss = total_loss / len(self.test_dataloader.dataset)
            print("Test Avg Loss: {:.4f} Accuracy : {:.4f}%".format(loss, (correct_pred / len(self.test_dataloader.dataset)) * 100 ))

    def predict(self, image):
        with torch.no_grad():
            image = image.to(self.device)
            return self.model(image)

