from Models.ToyModel import MLP
from TrainNet.train import train_EmoNet
from TrainProcess.Template import train_template
from Dataset.DatasetTemplate import DatasetTemplate

if __name__ == '__main__':
    # set up your dataset, epoch train process
    dataset, train_epochs = DatasetTemplate, train_template

    # your model
    net = MLP()

    # indexes of your dataset
    index = list(range(0, 15705))

    # train in EmoNet
    train_EmoNet(index=index, dataset=dataset,
             net=net, train_epochs=train_epochs,
             cl=True, dl=True)