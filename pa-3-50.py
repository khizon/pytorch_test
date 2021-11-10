import torch
# import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import os
import pickle
# import numpy as np
# from PIL import Image
# from matplotlib import pyplot as plt
from tqdm import tqdm
import csv

"""Build the timefunc decorator."""

import time
import functools


def timefunc(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure

class Cifar10(Dataset):

  def __init__(self, data_path, train = True, device = 'cpu'):
    self.data_path = data_path
    self.train = train
    self.device = device

  def __len__(self):
    if self.train:
      return 20000
    else:
      return 5000
  
  def __getitem__(self,index):
    if self.train:
      file = os.path.join(self.data_path, f'data_batch_{(index//10000)+1}')
    else:
      file = os.path.join(self.data_path,'test_batch')
    local_index = index%10000
    data = self.__unpickle(file)
    image = data[b'data'][local_index]
    label = data[b'labels'][local_index]
    # label = F.one_hot(torch.tensor(label), 10)
    # label = torch.tensor(label).float
    image, unnormalized = self.__preprocess(image)
    return image, label, unnormalized

  def __unpickle(self, file):
    with open(file, 'rb') as fo:
      batch = pickle.load(fo, encoding='bytes')
    return batch
  
  def __preprocess(self, image):
    image = torch.from_numpy(image).to(self.device)
    #limit between 0 to 1
    image = image/255
    #reshape to image square shape
    image = torch.reshape(image, (3,32,32))
    # Center crop to 28x28
    image = T.CenterCrop(28)(image) 
    unnormalized = image
    # Normalize the image to mean and std dev
    image = self.__normalize(image)
    return image, unnormalized

  def __normalize(self, image):
    mean = torch.mean(image, dim=[1,2]).tolist()
    std = torch.std(image, dim=[1,2]).tolist()
    return T.Normalize(mean, std)(image)

class Conv_Module(nn.Module):
  def __init__(self, in_channels, out_channels, p=0.0, **kwargs):
    super(Conv_Module, self).__init__()
    
    self.layers = nn.Sequential(
        nn.Dropout(p=p),
        nn.Conv2d(in_channels, out_channels, **kwargs),
        nn.BatchNorm2d(out_channels), 
        nn.ReLU()
    )

  def forward(self, x):
    return self.layers(x)

class Inception_Module(nn.Module):
  def __init__(self, in_channels, out_channels_1, out_channels_2, p):
    super(Inception_Module,self).__init__()

    self.conv1 = Conv_Module(in_channels, out_channels_1, p=p,
                            kernel_size=1, stride=1)
    self.conv2 = Conv_Module(in_channels, out_channels_2, p=p,
                            kernel_size=3, stride=1, padding=1)
    
  def forward(self,x):
    return torch.cat([self.conv1(x), self.conv2(x)], 1)

class Downsample_Module(nn.Module):
  def __init__(self, in_channels, out_channels, p):
    super(Downsample_Module, self).__init__()

    self.conv = Conv_Module(in_channels, out_channels, p=p,
                            kernel_size=3, stride=2)
    self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2)

  def forward(self, x):
    return torch.cat([self.conv(x), self.maxPool(x)],1)
    
class mini_inception(nn.Module):
  def __init__(self, in_channels, out_channels, p=0.0, **kwargs):
    super().__init__()

    self.layers = nn.Sequential(
        Conv_Module(in_channels, 96, p=p, kernel_size=3, stride=1),
        Inception_Module(96,32,32,p),
        Inception_Module(32+32,32,48,p),
        Downsample_Module(32+48,80,p),
        Inception_Module(32+48+80,112,48,p),
        Inception_Module(112+48,96,64,p),
        Inception_Module(96+64,80,80,p),
        Inception_Module(80+80,48,96,p),
        Downsample_Module(48+96,96,p),
        Inception_Module(48+96+96,176,160,p),
        Inception_Module(176+160,176,160,p),
        nn.AvgPool2d(kernel_size=5),
        nn.Flatten(),
        nn.Linear(176+160, 10),
        # nn.ReLU()
    )

  def forward(self, x):
    x = self.layers(x)
    return x

def create_data_loader(dataset, batch_size, train=True):
  return DataLoader(dataset, batch_size=batch_size, shuffle =train)

def train_single_epoch(model, data_loader, loss_fxn, optimizer, device, scheduler):
    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    running_loss = 0
    correct = 0
    total = 0

    for batch, (image, label, unnormalized) in loop:
        image, label = image.to(device), label.to(device)

        # calculate loss
        outputs = model(image)
        loss = loss_fxn(outputs, label)

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler
        scheduler.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs.data,1)
        total += label.size(0)
        correct += (pred == label).sum().item()

        # Update Progress bar
        loop.set_postfix(loss = loss.item(), acc = 100.0 *(correct/total))

    epoch_loss = running_loss/len(data_loader)
    epoch_accuracy = 100.0 * (correct/total)
    return epoch_loss, epoch_accuracy

@timefunc
def train(model, data_loader, loss_fxn, optimizer, scheduler, device, start_epoch, epochs):
  loss = []
  accuracy = []
  
  for epoch in range(start_epoch, epochs):
    start = time.perf_counter()   
    curr_loss, curr_acc = train_single_epoch(model, data_loader, loss_fxn, optimizer, device, scheduler)
    time_elapsed = time.perf_counter() - start
    loss.append(curr_loss)
    accuracy.append(curr_acc)

    # Checkpoint
    if ((epoch+1)%5) == 0:
      checkpoint = {
          'state_dict' : model.state_dict(),
          'optimizer' : optimizer.state_dict(),
          'scheduler' : scheduler.state_dict(),
          'loss' : loss,
          'accuracy': accuracy,
          'epoch': epoch
      }
      torch.save(checkpoint, f"check-50-{epoch}.pth.tar")
      print(f'Epoch: {epoch+1} | Loss: {curr_loss} | Checkpoint saved')
    else:
      print(f'Epoch: {epoch+1} | Loss: {curr_loss} |')

    # Save loss and accuracy values
    if not os.path.exists('metrics_50.csv'):
        with open('metrics_50.csv', 'w'): pass

    with open(r'metrics_50.csv', 'a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow([epoch, curr_loss, curr_acc, time_elapsed])

  print("Finished training")
  return loss, accuracy

if __name__ == "__main__":
    EPOCHS = 80
    DROPOUT_P = 0.5 #50% dropout
    BATCH_SIZE = 16
    start_epoch = 0
    DATA_SET_PATH = 'DATASETS/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f'Using: {device}')

    # Initialize Dataset

    train_set, test_set = Cifar10(DATA_SET_PATH, train=True, device = device), Cifar10(DATA_SET_PATH, train=False, device = device)
    train_dataloader, test_dataloader = create_data_loader(train_set, BATCH_SIZE, train=True), create_data_loader(test_set, BATCH_SIZE, train=False)

    # Initialize Model and Training Objects
    train_dataloader, test_dataloader = create_data_loader(train_set, BATCH_SIZE, train=True), create_data_loader(test_set, BATCH_SIZE, train=False)
    model = mini_inception(3,10,p=DROPOUT_P).to(device)
    loss_fxn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 1, end_factor = 1e-2, total_iters=EPOCHS)

    # Training Loop
    train_continue = False
    if train_continue:
        checkpoint = torch.load('check-30-4.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        start_epoch = checkpoint['epoch'] + 1
    else:
        if os.path.exists('metrics_50.csv'):
            os.remove('metrics_50.csv')
    loss, accuracy = train(model, train_dataloader, loss_fxn, optimizer, scheduler, device, start_epoch, EPOCHS,)